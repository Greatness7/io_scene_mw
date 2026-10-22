import dataclasses
import timeit

from .conversion.importer import Importer
from .conversion.options import import_options_from_mapping
from .conversion.runtime import ImportRuntime
from .preferences import Preferences

def load(context, filepath, report=None, scale_correction=None, **config):
    """load a scene from a nif file

    `scale_correction` overrides the add-on preference, for callers that need a fixed scale.
    """

    print(f"Import File: {filepath}")
    time = timeit.default_timer()

    options = import_options_from_mapping(config)
    runtime = ImportRuntime.from_context(context, Preferences.from_context(context))
    if scale_correction is not None:
        runtime = dataclasses.replace(runtime, scale_correction=scale_correction)
    importer = Importer(filepath, options, runtime)
    importer.execute()

    for warning in importer.warnings:
        if report:
            report({"WARNING"}, warning)

    time = timeit.default_timer() - time
    print(f"Import Done: {time:.4f} seconds")

    return {"FINISHED"}
