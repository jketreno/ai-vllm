"""Platform-specific SAM3 model loading and inference contexts."""

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch


_BUNDLED_SOURCE = Path(__file__).resolve().parent / "sam3"
if (_BUNDLED_SOURCE / "sam3").is_dir():
    sys.path.insert(0, str(_BUNDLED_SOURCE))


@dataclass(frozen=True)
class RuntimeConfig:
    platform: str
    device: str
    dtype: torch.dtype
    precision: str
    resolution: int


def _platform_name(value: str) -> str:
    aliases = {
        "gb10": "gb10",
        "nvidia": "gb10",
        "cuda": "gb10",
        "intel": "intel_arc",
        "intel_arc": "intel_arc",
        "b580": "intel_arc",
        "xpu": "intel_arc",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as error:
        raise RuntimeError(
            "SAM3_PLATFORM must be 'gb10' or 'intel_arc'"
        ) from error


def runtime_config(environ=None) -> RuntimeConfig:
    environ = os.environ if environ is None else environ
    platform = _platform_name(environ.get("SAM3_PLATFORM", "gb10"))
    resolution = int(environ.get("SAM3_RESOLUTION", "1008"))
    if resolution != 1008:
        raise RuntimeError(
            "SAM3_RESOLUTION must be 1008 for the bundled SAM3 checkpoint"
        )
    if platform == "gb10":
        precision = environ.get("SAM3_GB10_PRECISION", "bf16").strip().lower()
        dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
        if precision not in dtypes:
            raise RuntimeError(
                "SAM3_GB10_PRECISION must be 'bf16' or 'fp32'"
            )
        return RuntimeConfig(
            platform,
            "cuda",
            dtypes[precision],
            f"{precision}-weight" if precision == "bf16" else precision,
            resolution,
        )
    return RuntimeConfig(platform, "xpu", torch.float16, "fp16-weight", resolution)


def _device_module(config: RuntimeConfig):
    return getattr(torch, config.device, None)


def validate_device(config: RuntimeConfig) -> None:
    device_module = _device_module(config)
    if device_module is None or not device_module.is_available():
        raise RuntimeError(
            f"SAM3_PLATFORM={config.platform} requires an available "
            f"PyTorch {config.device.upper()} device"
        )


def inference_context(config: RuntimeConfig):
    if config.dtype != torch.float32:
        return torch.autocast(device_type=config.device, dtype=config.dtype)
    return nullcontext()


def reset_peak_memory_stats(config: RuntimeConfig) -> None:
    device_module = _device_module(config)
    if device_module is not None and device_module.is_available():
        device_module.reset_peak_memory_stats()


def memory_snapshot(config: RuntimeConfig) -> dict[str, int]:
    device_module = _device_module(config)
    if device_module is None or not device_module.is_available():
        return {}
    free, total = device_module.mem_get_info()
    return {
        "allocated": device_module.memory_allocated(),
        "reserved": device_module.memory_reserved(),
        "peak_allocated": device_module.max_memory_allocated(),
        "peak_reserved": device_module.max_memory_reserved(),
        "free": free,
        "total": total,
    }


def _move_cached_value(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_cached_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_cached_value(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_cached_value(item, device) for key, item in value.items()
        }
    return value


def _move_unregistered_tensor_caches(model, device):
    """Move tensor caches that the upstream model does not register as buffers."""
    excluded = {"_parameters", "_buffers", "_modules"}
    for module in model.modules():
        for name, value in vars(module).items():
            if name in excluded:
                continue
            moved = _move_cached_value(value, device)
            if moved is not value:
                setattr(module, name, moved)


def _convert_floating_weights(model, dtype):
    """Convert floating parameters without corrupting complex buffers."""
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=dtype)


class PlatformSAM3Annotator:
    """Load the same SAM3 checkpoint with platform-appropriate placement."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model = None
        self.processor = None

    def initialize(self):
        if self.model is not None:
            return self.model, self.processor
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        validate_device(self.config)
        model = build_sam3_image_model(device="cpu")
        if self.config.dtype != torch.float32:
            _convert_floating_weights(model, self.config.dtype)
        model = model.to(device=self.config.device)
        _move_unregistered_tensor_caches(model, self.config.device)
        _device_module(self.config).empty_cache()
        model.eval()
        self.model = model
        self.processor = Sam3Processor(
            model,
            resolution=self.config.resolution,
            device=torch.device(self.config.device),
        )
        return self.model, self.processor
