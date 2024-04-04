import platform

if platform.system() == 'Linux':
	from .linux import meshoptimizer
elif platform.system() == 'Darwin':
	from .macos import meshoptimizer
elif platform.system() == 'Windows':
	from .windows import meshoptimizer
else:
    raise ImportError('Unsupported OS')

optimize = meshoptimizer.optimize
