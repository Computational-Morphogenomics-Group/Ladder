from importlib.metadata import version

from . import data, model, models, module, scripts

__all__ = ["data", "model", "models", "module", "scripts"]

__version__ = version("scladder")
