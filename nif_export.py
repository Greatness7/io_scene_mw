import dataclasses
import timeit

from es3.utils.geometry import GeometryError

from . import nif_node_preview, nif_switch
from .conversion.exporter import Exporter
from .conversion.options import export_options_from_mapping
from .conversion.runtime import ExportRuntime
from .preferences import Preferences

def save(context, filepath, report=None, scale_correction=None, **config):
    """save the scene to a nif file

    `scale_correction` overrides the add-on preference, for callers that need a fixed scale.
    """

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    # a previewed billboard's turn lives in its delta rotation, which the exported matrices include
    if nif_node_preview.unrotate():
        context.view_layer.update()
        nif_node_preview.refresh()  # the next preview tick turns them again

    options = export_options_from_mapping(config)
    runtime = ExportRuntime.from_context(context, Preferences.from_context(context))
    if scale_correction is not None:
        runtime = dataclasses.replace(runtime, scale_correction=1 / scale_correction)
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
