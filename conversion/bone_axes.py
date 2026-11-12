"""Define bone-axis corrections for NIF import and export."""

import numpy as np
import numpy.linalg as la

# Blender provenance: axis_conversion("-X", "Z", "Y", "Z").to_4x4()
IMPORT_BIPED = np.array(
    ((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    dtype="<f",
)

# Blender provenance: axis_conversion("Y", "Z", "-Z", "-Y").to_4x4()
IMPORT_OTHER = np.array(
    ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1)),
    dtype="<f",
)

EXPORT_BIPED = la.inv(IMPORT_BIPED)
EXPORT_OTHER = IMPORT_OTHER


def import_axis_correction(name: str) -> np.ndarray:
    return IMPORT_BIPED if "Bip01" in name else IMPORT_OTHER


def export_axis_correction(name: str) -> np.ndarray:
    return EXPORT_BIPED if "Bip01" in name else EXPORT_OTHER
