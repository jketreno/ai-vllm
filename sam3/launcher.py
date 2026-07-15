"""Launch Streamlit with Prometheus instrumentation for SAM3 workflows."""

import functools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server


MODEL_LOADED = Gauge("sam3_model_loaded", "Whether the SAM3 model is loaded")
MODEL_LOADS = Counter("sam3_model_loads_total", "SAM3 model loads", ["status"])
MODEL_LOAD_SECONDS = Histogram(
    "sam3_model_load_seconds",
    "SAM3 model load latency",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, float("inf")),
)
ANNOTATIONS = Counter("sam3_annotation_requests_total", "Annotation requests", ["status"])
ANNOTATION_ACTIVE = Gauge("sam3_annotation_active", "Annotations currently running")
ANNOTATION_SECONDS = Histogram(
    "sam3_annotation_duration_seconds",
    "Annotation latency",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, float("inf")),
)
ANNOTATION_INPUT_BYTES = Histogram(
    "sam3_annotation_input_bytes",
    "Input image size",
    buckets=(10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, float("inf")),
)
ANNOTATION_PROMPTS = Histogram(
    "sam3_annotation_prompts", "Prompts per request", buckets=(1, 2, 5, 10, 25, 50, float("inf"))
)
DETECTIONS = Counter("sam3_detections_total", "Detections returned by SAM3")
PIPELINE_OPS = Counter("sam3_pipeline_operations_total", "Pipeline operations", ["operation", "status"])
PIPELINE_SECONDS = Histogram(
    "sam3_pipeline_operation_duration_seconds",
    "Pipeline operation latency",
    ["operation"],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 300, 900, 3600, float("inf")),
)
VIDEO_FRAMES = Counter("sam3_video_frames_extracted_total", "Video frames extracted")
TRAINING_JOBS = Counter("sam3_training_jobs_total", "Training jobs", ["status"])
TRAINING_ACTIVE = Gauge("sam3_training_jobs_active", "Training jobs currently running")
TRAINING_SECONDS = Histogram(
    "sam3_training_duration_seconds",
    "Training job duration",
    buckets=(1, 10, 30, 60, 300, 900, 1800, 3600, 7200, 14400, float("inf")),
)
CUDA_ALLOCATED = Gauge("sam3_cuda_memory_allocated_bytes", "CUDA memory allocated by this process")
CUDA_RESERVED = Gauge("sam3_cuda_memory_reserved_bytes", "CUDA memory reserved by this process")
CUDA_FREE = Gauge("sam3_cuda_memory_free_bytes", "CUDA memory currently free")


def update_cuda_metrics():
    """Update process and device CUDA memory gauges when CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            CUDA_ALLOCATED.set(torch.cuda.memory_allocated())
            CUDA_RESERVED.set(torch.cuda.memory_reserved())
            free, _ = torch.cuda.mem_get_info()
            CUDA_FREE.set(free)
    except Exception:
        pass


def instrument_model_and_annotations():
    """Wrap SAM3 model initialization and image annotation."""
    from managers.annotation_manager import SAM3Annotator

    original_initialize = SAM3Annotator.initialize
    original_annotate = SAM3Annotator.annotate_single_image

    @functools.wraps(original_initialize)
    def initialize(self, *args, **kwargs):
        if self.model is not None:
            return original_initialize(self, *args, **kwargs)
        started = time.monotonic()
        try:
            result = original_initialize(self, *args, **kwargs)
            MODEL_LOADED.set(1)
            MODEL_LOADS.labels("success").inc()
            return result
        except Exception:
            MODEL_LOADED.set(0)
            MODEL_LOADS.labels("error").inc()
            raise
        finally:
            MODEL_LOAD_SECONDS.observe(time.monotonic() - started)
            update_cuda_metrics()

    @functools.wraps(original_annotate)
    def annotate(self, image_path, text_prompts, *args, **kwargs):
        started = time.monotonic()
        ANNOTATION_ACTIVE.inc()
        ANNOTATION_PROMPTS.observe(len(text_prompts))
        try:
            ANNOTATION_INPUT_BYTES.observe(Path(image_path).stat().st_size)
        except OSError:
            pass
        try:
            result = original_annotate(self, image_path, text_prompts, *args, **kwargs)
            ANNOTATIONS.labels("success").inc()
            DETECTIONS.inc(len(result.get("detections", ())))
            return result
        except Exception:
            ANNOTATIONS.labels("error").inc()
            raise
        finally:
            ANNOTATION_ACTIVE.dec()
            ANNOTATION_SECONDS.observe(time.monotonic() - started)
            update_cuda_metrics()

    SAM3Annotator.initialize = initialize
    SAM3Annotator.annotate_single_image = annotate


def wrap_operation(owner, method_name, operation, result_count=None):
    """Instrument one synchronous manager method."""
    descriptor = owner.__dict__[method_name]
    original = getattr(owner, method_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        started = time.monotonic()
        try:
            result = original(*args, **kwargs)
            PIPELINE_OPS.labels(operation, "success").inc()
            if result_count:
                result_count(result)
            return result
        except Exception:
            PIPELINE_OPS.labels(operation, "error").inc()
            raise
        finally:
            PIPELINE_SECONDS.labels(operation).observe(time.monotonic() - started)

    if isinstance(descriptor, staticmethod):
        wrapped = staticmethod(wrapped)
    elif isinstance(descriptor, classmethod):
        wrapped = classmethod(wrapped)
    setattr(owner, method_name, wrapped)


def instrument_pipeline():
    """Instrument video, dataset, augmentation, and project operations."""
    from managers.augmentation_manager import AugmentationManager
    from managers.dataset_manager import DatasetManager
    from managers.project_manager import ProjectManager
    from managers.video_manager import VideoManager

    wrap_operation(VideoManager, "extract_frames", "video_extract", lambda frames: VIDEO_FRAMES.inc(len(frames)))
    wrap_operation(DatasetManager, "save_to_dataset", "dataset_export")
    wrap_operation(AugmentationManager, "generate_augmented_dataset", "augmentation")
    wrap_operation(ProjectManager, "create", "project_create")
    wrap_operation(ProjectManager, "load", "project_load")
    wrap_operation(ProjectManager, "delete_project", "project_delete")


def instrument_training_subprocesses():
    """Track lifecycle of train_worker subprocesses launched by the UI."""
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        command = kwargs.get("args", args[0] if args else ())
        command_text = " ".join(map(str, command)) if not isinstance(command, str) else command
        if "train_worker.py" not in command_text:
            return process
        TRAINING_JOBS.labels("started").inc()
        TRAINING_ACTIVE.inc()
        started = time.monotonic()

        def watch():
            return_code = process.wait()
            TRAINING_ACTIVE.dec()
            TRAINING_SECONDS.observe(time.monotonic() - started)
            TRAINING_JOBS.labels("success" if return_code == 0 else "error").inc()

        threading.Thread(target=watch, daemon=True).start()
        return process

    subprocess.Popen = popen


def main():
    metrics_port = int(os.environ.get("SAM3_METRICS_PORT", "9092"))
    start_http_server(metrics_port)
    instrument_model_and_annotations()
    instrument_pipeline()
    instrument_training_subprocesses()

    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
