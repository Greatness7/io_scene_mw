"""Typed intermediate buffers used by the import and export pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generic, TypeVar

from numpy.typing import NDArray

Array = NDArray[Any]
NodeT = TypeVar("NodeT")
MaterialT = TypeVar("MaterialT")


class ImportRole(Enum):
    EMPTY = auto()
    MESH = auto()
    ARMATURE = auto()
    BONE = auto()
    PRUNED = auto()


@dataclass(eq=False)
class ImportNode:
    importer: Any
    source: Any
    matrix_local: Array
    role: ImportRole = ImportRole.EMPTY
    output: Any = None
    _parent: ImportNode | None = None
    children: list[ImportNode] = field(default_factory=list)
    matrix_posed: Array | None = None
    switch_plan: ImportSwitchPlan | None = None


@dataclass(eq=False)
class ImportNullSlot:
    importer: Any
    parent: ImportNode
    output: Any = None


@dataclass(frozen=True, slots=True)
class ImportSwitchPlan:
    owner: ImportNode
    slots: tuple[ImportNode | ImportNullSlot, ...]


@dataclass(frozen=True, slots=True)
class ResolvedImportSkeleton:
    root: ImportNode
    bones: tuple[ImportNode, ...]


@dataclass(frozen=True, slots=True)
class ImportCreationJob:
    target: ImportNode | ImportNullSlot
    role: ImportRole | None
    owned_bones: tuple[ImportNode, ...] = ()


class ExportRole(Enum):
    EMPTY = auto()
    MESH = auto()
    ARMATURE = auto()
    BONE = auto()
    UNSUPPORTED = auto()


@dataclass(eq=False)
class ExportNode:
    exporter: Any
    source: Any
    matrix_local: Array
    role: ExportRole = ExportRole.UNSUPPORTED
    output: Any = None
    _parent: ExportNode | None = None
    children: list[ExportNode] = field(default_factory=list)
    matrix_bind: Array | None = None
    baked_vertex_scale: Array | None = None
    switch_plan: ExportSwitchPlan | None = None


@dataclass(frozen=True, slots=True)
class ExportNullSlot:
    source: Any


@dataclass(frozen=True, slots=True)
class ExportSwitchPlan:
    owner: ExportNode
    slots: tuple[ExportNode | ExportNullSlot, ...]


@dataclass(frozen=True, slots=True)
class ResolvedExportSkeleton:
    armature: ExportNode
    bones: tuple[ExportNode, ...]


@dataclass(frozen=True, slots=True)
class ExportCreationJob:
    target: ExportNode
    role: ExportRole
    owned_bones: tuple[ExportNode, ...] = ()


@dataclass(slots=True)
class ImportSkinBuffers:
    weights: Array


@dataclass(slots=True)
class ImportMorphBuffers:
    targets: Array


@dataclass(slots=True)
class ImportMeshBuffers:
    vertices: Array
    triangles: Array
    normals: Array
    uv_sets: Array
    vertex_colors: Array
    skin: ImportSkinBuffers | None
    morphs: ImportMorphBuffers | None


@dataclass(slots=True)
class ExportSkinBuffers(Generic[NodeT]):
    root: NodeT
    bones: tuple[NodeT, ...]
    weights: Array


@dataclass(slots=True)
class ExportMorphBuffers:
    targets: Array
    keys: tuple[Array, ...]
    interpolations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MaterialPartition(Generic[MaterialT]):
    material: MaterialT | None
    corner_indices: Array | None
