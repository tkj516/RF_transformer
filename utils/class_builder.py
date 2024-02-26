import inspect
from typing import Any, Callable, List, Mapping
from ml_collections import ConfigDict


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class ClassBuilder:
    def __init__(self, register: Mapping[str, Callable]):
        self.register = register

    def is_registered(self, config: List[Any]) -> bool:
        if config[0] not in self.register:
            return False
        return True

    def build_class(self, config: List[Any], **kwargs):
        if not self.is_registered(config):
            raise ValueError(f"{config[0]} has not been registered!")
        if len(config) == 1:
            return self.register[config[0]](**kwargs)
        else:
            return self.register[config[0]](**{**config[1], **kwargs})

    def build_dataclass(self, config: List[Any], **kwargs):
        if not self.is_registered(config):
            raise ValueError(f"{config[0]} has not been registered!")
        if len(config) == 1:
            return ConfigDict(
                {
                    **get_default_args(self.register[config[0]]), 
                    **kwargs
                }
            )
        else:
            return ConfigDict(
                {
                    **get_default_args(self.register[config[0]]), 
                    **config[1], 
                    **kwargs
                }
            )

    def build(self, config: List[Any], **kwargs):
        return self.build_class(config, **kwargs), self.build_dataclass(config)