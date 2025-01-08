import platform

system = platform.system()
architecture = platform.machine()

if system == "Linux":
    from .linux import meshoptimizer
elif system == "Darwin":
    if architecture == "x86_64":
        from .macos.x86_64 import meshoptimizer
    elif architecture == "arm64":
        from .macos.arm64 import meshoptimizer
elif system == "Windows":
    from .windows import meshoptimizer

try:
    optimize = meshoptimizer.optimize
except (NameError, AttributeError):
    ImportError(f"Unsupported system or architecture: {system}/{architecture}")
