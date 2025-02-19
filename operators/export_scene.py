import bpy
from bpy_extras.io_utils import ExportHelper  # type: ignore


class ExportScene(bpy.types.Operator, ExportHelper):
    """Export a Morrowind NIF File"""

    bl_idname = "export_scene.mw"
    bl_options = {"PRESET", "UNDO"}
    bl_label = "Export NIF"

    # -- export helper --

    filename_ext = ".nif"
    filter_glob: bpy.props.StringProperty(default="*.nif", options={"HIDDEN"})

    # -- user config --

    vertex_precision: bpy.props.FloatProperty(
        name="Vertex Precision",
        description="Precision used when optimizing vertices. Respects normals. (Recommended: 0.001)",
        default=0.001,
        min=0.0001,
        max=1000.0,
    )

    use_active_collection: bpy.props.BoolProperty(
        name="Only Collection",
        description="Only export objects from the active collection",
        default=False,
    )

    use_selection: bpy.props.BoolProperty(
        name="Only Selected",
        description="Only export objects that are selected",
        default=False,
    )

    export_animations: bpy.props.BoolProperty(
        name="Export Animations",
        description="Animations will be exported. Uncheck to skip all animation data during export",
        default=True,
    )

    extract_keyframe_data: bpy.props.BoolProperty(
        name="Extract Keyframe Data",
        description=(
            "Extract animations and visuals to corresponding 'x.kf' and 'x.nif' files."
            "\n(e.g. exporting 'base_anim.nif' will create 'xbase_anim.nif' and 'xbase_anim.kf' files)"
        ),
        default=False,
    )

    preserve_root_tranforms: bpy.props.BoolProperty(
        name="Preserve Root Transforms",
        description=(
            "Preserve the root object's transformations by inserting an additional parent node above it."
            " This setting is only applicable if there is a single root object with modified transforms."
            " (Recommended: False)"
        ),
        default=False,
    )

    preserve_material_names: bpy.props.BoolProperty(
        name="Preserve Material Names",
        description=(
            "Preserve material names from the source file. If unchecked materials will be renamed based on the"
            " assigned textures"
        ),
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        from .. import nif_export

        kwargs = self.as_keywords(ignore=("filename_ext", "filter_glob", "check_existing"))
        return nif_export.save(context, **kwargs)

    @staticmethod
    def menu_func_export(menu, context) -> None:
        menu.layout.operator(ExportScene.bl_idname, text="Morrowind (.nif)")

    @staticmethod
    def register() -> None:
        bpy.types.TOPBAR_MT_file_export.append(ExportScene.menu_func_export)

    @staticmethod
    def unregister() -> None:
        bpy.types.TOPBAR_MT_file_export.remove(ExportScene.menu_func_export)
