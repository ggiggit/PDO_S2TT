"""Persistent Delivery Optimization inference package."""

__all__ = ["PDOS2TT"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "PDOS2TT":
        from .model import PDOS2TT

        return PDOS2TT
    raise AttributeError(name)
