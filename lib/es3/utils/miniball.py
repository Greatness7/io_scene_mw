import platform

system = platform.system()
architecture = platform.machine()

if system == "Linux":
    from .linux import miniball
elif system == "Darwin" and architecture == "x86_64":
    from .macos.x86_64 import miniball
elif system == "Darwin" and architecture == "arm64":
    from .macos.arm64 import miniball
elif system == "Windows":
    from .windows import miniball
else:
    raise ImportError(f"Unsupported system or architecture: {system}/{architecture}")

try:
    compute = miniball.compute  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
except AttributeError as exc:
    raise ImportError(f"Unsupported system or architecture: {system}/{architecture}") from exc
