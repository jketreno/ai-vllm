"""Launch SAM3-Demo with memory-safe lazy model switching."""

import gc
import os
import runpy
import threading

import torch
import transformers


class ModelPool:
    """Keep only one SAM3 model variant resident on the GPU."""

    def __init__(self):
        self.active = None
        self.lock = threading.RLock()

    def activate(self, proxy):
        with self.lock:
            if self.active is proxy and proxy.model is not None:
                return proxy.model
            if self.active is not None:
                self.active.release()
            proxy.load()
            self.active = proxy
            return proxy.model


POOL = ModelPool()


class LazyModel:
    """Delay checkpoint loading until a UI mode first invokes its model."""

    def __init__(self, loader, args, kwargs):
        self.loader = loader
        self.args = args
        self.kwargs = kwargs
        self.to_args = ()
        self.to_kwargs = {}
        self.model = None

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return self

    def load(self):
        if self.model is None:
            model = self.loader(*self.args, **self.kwargs)
            self.model = model.to(*self.to_args, **self.to_kwargs)

    def release(self):
        if self.model is None:
            return
        self.model.to("cpu")
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        return POOL.activate(self)(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(POOL.activate(self), name)


def install_lazy_loaders():
    for class_name in ("Sam3Model", "Sam3TrackerModel", "Sam3VideoModel"):
        model_class = getattr(transformers, class_name)
        loader = model_class.from_pretrained

        def lazy_from_pretrained(*args, _loader=loader, **kwargs):
            return LazyModel(_loader, args, kwargs)

        model_class.from_pretrained = lazy_from_pretrained


def configure_gradio():
    import gradio as gr

    original_launch = gr.Blocks.launch

    def launch(blocks, *args, **kwargs):
        kwargs.setdefault("server_name", "0.0.0.0")
        kwargs.setdefault("server_port", 7860)
        kwargs.setdefault("root_path", os.environ.get("GRADIO_ROOT_PATH", "/sam3-demo"))
        kwargs.setdefault("max_file_size", "1gb")
        kwargs["mcp_server"] = False
        return original_launch(blocks, *args, **kwargs)

    gr.Blocks.launch = launch


if __name__ == "__main__":
    install_lazy_loaders()
    configure_gradio()
    runpy.run_path("/app/app.py", run_name="__main__")
