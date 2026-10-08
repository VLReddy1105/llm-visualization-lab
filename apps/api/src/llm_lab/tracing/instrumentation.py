"""Only this module registers instrumentation; every handle belongs to one run."""

from llm_lab.models.adapters.base import HookTarget


class TraceSession:
    def __init__(self, targets: tuple[HookTarget, ...], collector) -> None:
        self.targets = targets
        self.collector = collector
        self.handles = []

    def __enter__(self):
        try:
            for target in self.targets:
                observe = target.observations

                def callback(module, args, kwargs, output=None, _observe=observe):
                    for observation in _observe(args, kwargs, output):
                        self.collector.observe(observation.spec, observation.tensor)
                    # None preserves every input and output object.
                    return None

                if target.kind == "pre":
                    handle = target.module.register_forward_pre_hook(callback, with_kwargs=True)
                else:
                    handle = target.module.register_forward_hook(callback, with_kwargs=True)
                self.handles.append(handle)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.targets = ()
        self.collector = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def hook_counts(model):
    return tuple(
        (len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules()
    )
