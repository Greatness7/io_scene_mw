import timeit

from es3.utils.geometry import GeometryError

from . import nif_switch
from .conversion.exporter import Exporter
from .conversion.options import export_options_from_mapping
from .conversion.runtime import ExportRuntime
from .preferences import Preferences

def save(context, filepath, report=None, **config):
    """save the scene to a nif file"""

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    options = export_options_from_mapping(config)
    runtime = ExportRuntime.from_context(context, Preferences.from_context(context))
    exporter = Exporter(filepath, options, runtime)
    try:
        exporter.execute()
    except GeometryError as error:
        # malformed geometry cancels the export like any other failed precondition
        raise nif_switch.SwitchError(str(error)) from error

    for warning in exporter.warnings:
        if report:
            report({"WARNING"}, warning)

    time = timeit.default_timer() - time
    print(f"Export Done: {time:.4f} seconds")

    return {"FINISHED"}
