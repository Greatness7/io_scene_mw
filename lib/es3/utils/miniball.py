import platform

system = platform.system()
architecture = platform.machine()

if system == "Linux":
    from .linux import miniball
elif system == "Darwin":
    if architecture == "x86_64":
        from .macos.x86_64 import miniball
    elif architecture == "arm64":
        from .macos.arm64 import miniball
elif system == "Windows":
    from .windows import miniball

try:
    compute = miniball.compute
except (NameError, AttributeError):
    ImportError(f"Unsupported system or architecture: {system}/{architecture}")
