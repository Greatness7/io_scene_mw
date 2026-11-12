"""Validate operator option mappings and hold immutable import and export settings.

This module does not depend on bpy. Option names must match the corresponding Blender operator properties.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImportOptions:
    vertex_precision: float = 0.001
    attach_keyframe_data: bool = False
    discard_root_transforms: bool = True
    use_existing_materials: bool = False
    ignore_collision_nodes: bool = False
    ignore_custom_normals: bool = False
    ignore_animations: bool = False


@dataclass(frozen=True, slots=True)
class ExportOptions:
    vertex_precision: float = 0.001
    use_active_collection: bool = False
    use_selection: bool = False
    export_animations: bool = True
    randomize_animations: bool = True
    extract_keyframe_data: bool = False
    preserve_root_transforms: bool = False
    preserve_material_names: bool = True
    strip_numeric_suffixes: bool = False


IMPORT_DEFAULTS = ImportOptions()
EXPORT_DEFAULTS = ExportOptions()

_IMPORT_OPTION_NAMES = frozenset(ImportOptions.__dataclass_fields__)
_EXPORT_OPTION_NAMES = frozenset(ExportOptions.__dataclass_fields__)


def _reject_unknown_options(values: Mapping[str, object], names: frozenset[str], kind: str) -> None:
    unknown = sorted(set(values) - names)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unknown {kind} option(s): {joined}")


def _bool_option(values: Mapping[str, object], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"Option {name} must be a bool, not {type(value).__name__}")
    return value


def _float_option(values: Mapping[str, object], name: str, default: float) -> float:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Option {name} must be a number, not {type(value).__name__}")
    return float(value)


def import_options_from_mapping(values: Mapping[str, object]) -> ImportOptions:
    _reject_unknown_options(values, _IMPORT_OPTION_NAMES, "import")
    return ImportOptions(
        vertex_precision=_float_option(values, "vertex_precision", IMPORT_DEFAULTS.vertex_precision),
        attach_keyframe_data=_bool_option(values, "attach_keyframe_data", IMPORT_DEFAULTS.attach_keyframe_data),
        discard_root_transforms=_bool_option(values, "discard_root_transforms", IMPORT_DEFAULTS.discard_root_transforms),
        use_existing_materials=_bool_option(values, "use_existing_materials", IMPORT_DEFAULTS.use_existing_materials),
        ignore_collision_nodes=_bool_option(values, "ignore_collision_nodes", IMPORT_DEFAULTS.ignore_collision_nodes),
        ignore_custom_normals=_bool_option(values, "ignore_custom_normals", IMPORT_DEFAULTS.ignore_custom_normals),
        ignore_animations=_bool_option(values, "ignore_animations", IMPORT_DEFAULTS.ignore_animations),
    )


def export_options_from_mapping(values: Mapping[str, object]) -> ExportOptions:
    _reject_unknown_options(values, _EXPORT_OPTION_NAMES, "export")
    return ExportOptions(
        vertex_precision=_float_option(values, "vertex_precision", EXPORT_DEFAULTS.vertex_precision),
        use_active_collection=_bool_option(values, "use_active_collection", EXPORT_DEFAULTS.use_active_collection),
        use_selection=_bool_option(values, "use_selection", EXPORT_DEFAULTS.use_selection),
        export_animations=_bool_option(values, "export_animations", EXPORT_DEFAULTS.export_animations),
        randomize_animations=_bool_option(values, "randomize_animations", EXPORT_DEFAULTS.randomize_animations),
        extract_keyframe_data=_bool_option(values, "extract_keyframe_data", EXPORT_DEFAULTS.extract_keyframe_data),
        preserve_root_transforms=_bool_option(values, "preserve_root_transforms", EXPORT_DEFAULTS.preserve_root_transforms),
        preserve_material_names=_bool_option(values, "preserve_material_names", EXPORT_DEFAULTS.preserve_material_names),
        strip_numeric_suffixes=_bool_option(values, "strip_numeric_suffixes", EXPORT_DEFAULTS.strip_numeric_suffixes),
    )
