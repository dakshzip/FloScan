"""FloScan: property capture to measured floor plan, damage and scope."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("floscan")
except PackageNotFoundError:  # Running from a source tree without installation.
    __version__ = "0+unknown"
