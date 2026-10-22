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


@dataclass(slots=True)
class ResolvedImportSkeleton:
    root: ImportNode
    root_bone: ImportNode
    bones: tuple[ImportNode, ...]
    posed_world: dict[ImportNode, Array]
    rest_world: dict[ImportNode, Array]


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


@dataclass(eq=False, slots=True)
class ExportScope:
    parent: ExportScope | None = None


@dataclass(eq=False)
class ExportNode:
    exporter: Any
    source: Any
    matrix_local: Array
    scope: ExportScope | None = None
    role: ExportRole = ExportRole.UNSUPPORTED
    output: Any = None
    _parent: ExportNode | None = None
    children: list[ExportNode] = field(default_factory=list)
    baked_vertex_scale: Array | None = None
    switch_plan: ExportSwitchPlan | None = None


@dataclass(frozen=True, slots=True)
class ExportNullSlot:
    source: Any


@dataclass(frozen=True, slots=True)
class ExportSwitchPlan:
    owner: ExportNode
    slots: tuple[ExportNode | ExportNullSlot, ...]


@dataclass(slots=True)
class ResolvedExportSkeleton:
    armature: ExportNode
    bones: tuple[ExportNode, ...]
    bind_relative_to_root: dict[ExportNode, Array]
    world_matrices: dict[ExportNode, Array]


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
    bind_relative_to_root: dict[NodeT, Array]


@dataclass(slots=True)
class ExportMorphBuffers:
    targets: Array
    keys: tuple[Array, ...]
    interpolations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExportMorphCapture:
    """Shape-key state read before export evaluation mutes any key.

    `coordinates` holds absolute source-vertex positions `(M, V, 3)`, the basis first, followed by every target
    that is unmuted, keyed, and has an unmuted F-curve. `keys` and `interpolations` align with it.
    """

    coordinates: Array
    keys: tuple[Array, ...]
    interpolations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExportMeshMetadata:
    """Author state of one mesh source, captured before the evaluation scope changes it."""

    ignore_normals: bool
    morphs: ExportMorphCapture | None


@dataclass(frozen=True, slots=True)
class ExportSkinBinding(Generic[NodeT]):
    """The skeleton an exported mesh occurrence binds to; `groups[i]` is the vertex-group index of `bones[i]`."""

    root: NodeT
    groups: tuple[int, ...]
    bones: tuple[NodeT, ...]
    bind_relative_to_root: dict[NodeT, Array]


@dataclass(frozen=True, slots=True)
class MaterialPartition(Generic[MaterialT]):
    material: MaterialT | None
    corner_indices: Array | None


@dataclass(frozen=True, slots=True)
class ExportMeshBuffers(Generic[NodeT, MaterialT]):
    """Owned geometry of one exported mesh occurrence.

    Vertex-aligned fields share one axis of length N (see `es3.utils.geometry.VertexPayload`): positions
    `(N, 3)`, normals and colors `(N, 3)`/`(N, 4)` or empty, UVs `(S, N, 2)`, skin weights `(B, N)`, and morph
    rows `(M, N, 3)`. Before welding N counts triangle corners and `triangles` is `arange(N).reshape(-1, 3)`.
    """

    vertices: Array
    normals: Array
    vertex_colors: Array
    uv_sets: Array
    triangles: Array
    partitions: tuple[MaterialPartition[MaterialT], ...]
    skin: ExportSkinBuffers[NodeT] | None
    morphs: ExportMorphBuffers | None
