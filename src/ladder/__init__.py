from importlib.metadata import version

from . import data, models, scripts

__all__ = ["data", "models", "scripts"]

__version__ = version("ladder")
