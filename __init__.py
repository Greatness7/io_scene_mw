bl_info = {
    "name": "Morrowind (.nif)",
    "author": "Greatness7",
    "version": (0, 8, 76),
    "blender": (2, 82, 0),
    "location": "File > Import/Export > Morrowind (.nif)",
    "description": "Import/Export files for Morrowind",
    "wiki_url": "https://blender-morrowind.readthedocs.io/",
    "tracker_url": "https://github.com/Greatness7/io_scene_mw/issues",
    "category": "Import-Export",
}

import sys
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper

PATH = Path(__file__).parent


# ------------
# Make lib contents accessible to other addons or scripts.
item = str(PATH / "lib")
if item not in sys.path:
    sys.path.append(item)

# Support Blender's "Reload Scripts" feature. (hotkey: F8)
for item in PATH.iterdir():
    item = locals().get(item.stem)
    if item is not None:
        import importlib
        importlib.reload(item)

del item
# ------------


# --------
# UPDATING
# --------

class UpdateCheck(bpy.types.Operator):
    """Check if a new version is available on the plugin repository."""

    bl_idname = "preferences.mw_update_check"
    bl_options = {"REGISTER"}

    bl_label = "Check for updates"
    bl_description = "Requires internet connection"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        version, zipball_url = self.get_latest_version_info()

        if version > bl_info["version"]:
            # require manual install on non-patch releases
            # cannot replace native libraries while loaded
            if version[:2] > bl_info["version"][:2]:
                Preferences.update_status = "UPDATE_FORBIDDEN"
                Preferences.update_url = zipball_url
            else:
                Preferences.update_status = "UPDATE_AVAILABLE"
                Preferences.update_url = zipball_url
        else:
            Preferences.update_status = "UPDATE_INSTALLED"
            Preferences.update_url = ""

        return {"FINISHED"}

    @staticmethod
    def get_latest_version_info():
        import ssl
        import json
        import urllib.request

        tags_url = "https://api.github.com/repos/Greatness7/io_scene_mw/tags"

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(tags_url, context=ctx) as response:
            data = response.read()

        latest, *_ = json.loads(data)
        latest_zipball_url = latest["zipball_url"]
        latest_version_tag = latest["name"].split(".")
        latest_version = tuple(map(int, latest_version_tag))

        return latest_version, latest_zipball_url


class UpdateApply(bpy.types.Operator):
    """Download and install the latest version from the plugin repository."""

    bl_idname = "preferences.mw_update_apply"
    bl_options = {"REGISTER"}

    bl_label = "An update is available! Click to install"
    bl_description = "Requires internet connection"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        self.create_backup()
        self.install_files(Preferences.update_url)
        Preferences.update_status = "UPDATE_INSTALLED"
        Preferences.update_url = ""
        return {"FINISHED"}

    def create_backup(self):
        """Make a backup archive of our addon in the parent directory.
            e.g. scripts/addons/io_scene_mw.zip
        """
        import shutil
        shutil.make_archive(PATH, "zip", root_dir=PATH.parent, base_dir=PATH.name)

    def install_files(self, zipball_url):
        import io
        import ssl
        import zipfile as zf
        import urllib.request

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(zipball_url, context=ctx) as response:
            zipball = zf.ZipFile(io.BytesIO(response.read()))

        root = Path(PATH.name)

        for info in zipball.infolist()[1:]:
            parts = Path(info.filename).parts[1:]
            relative_path = root.joinpath(*parts)
            if not relative_path.suffix:
                continue  # directories

            try:
                info.filename = str(relative_path)
                zipball.extract(info, PATH.parent)
            except PermissionError:
                print(f"PermissionError: cannot replace {info.filename}")


class UpdateNotes(bpy.types.Operator):
    bl_idname = "preferences.mw_update_notes"
    bl_options = {"REGISTER"}

    bl_label = "You are up to date"
    bl_description = "Click to open changelog"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        Preferences.update_status = "UPDATE_UNCHECKED"
        Preferences.update_url = ""
        bpy.ops.wm.url_open(url="https://github.com/Greatness7/io_scene_mw/releases/latest")
        return {"FINISHED"}


class UpdateLimit(bpy.types.Operator):
    bl_idname = "preferences.mw_update_limit"
    bl_options = {"REGISTER"}

    bl_label = "An update is available, but requires manual installation"
    bl_description = "Click to open download page"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://blender-morrowind.readthedocs.io/en/latest/getting-started/downloading.html")
        return {"FINISHED"}


# -------------
# TEXTURE PATHS
# -------------

class TexturePath(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        subtype="DIR_PATH",
        name="",
        description="Texture Path",
        default="C:\\Morrowind\\Data Files\\Textures\\",
    )


class TexturePathList(bpy.types.UIList):
    bl_idname = 'PREFERENCES_UL_MW_TexturePathList'

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        is_active = getattr(active_data, active_propname) == index
        checkbox_icon = 'CHECKBOX_HLT' if is_active else 'CHECKBOX_DEHLT'
        layout.prop(item, "name", icon=checkbox_icon, emboss=False)


class TexturePathAdd(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_add"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Add Texture Path"
    bl_description = "Add a new texture path"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        p = Preferences.from_context(context)
        p.texture_paths.add()
        p.texture_paths_active_index += 1
        p.texture_paths.move(len(p.texture_paths) - 1, p.texture_paths_active_index)
        return {"FINISHED"}


class TexturePathRemove(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_remove"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Remove Texture Path"
    bl_description = "Remove the selected texture path"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        p = Preferences.from_context(context)
        p.texture_paths.remove(p.texture_paths_active_index)
        p.texture_paths_active_index = min(p.texture_paths_active_index, len(p.texture_paths) - 1)
        return {"FINISHED"}


class TexturePathMove(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_move"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Move Texture Path"
    bl_description = "Move the selected texture path up or down"

    direction: bpy.props.IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'PREFERENCES'

    def execute(self, context):
        p = Preferences.from_context(context)
        old_index = p.texture_paths_active_index
        new_index = old_index + self.direction
        if len(p.texture_paths) > new_index > -1:
            p.texture_paths.move(old_index, new_index)
            p.texture_paths_active_index = new_index
        return {"FINISHED"}


# -----------
# PREFERENCES
# -----------

class Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    update_status, update_url = "UPDATE_UNCHECKED", ""

    scale_correction: bpy.props.FloatProperty(
        name="Scale Correction",
        description="Adjust the scale of imported/exported objects by a specified multiplier",
        min=0.01,
        max=10.0,
        default=0.01,
    )

    texture_paths: bpy.props.CollectionProperty(type=TexturePath)

    texture_paths_active_index: bpy.props.IntProperty(name="Select Item", options={'SKIP_SAVE'})

    def draw(self, context):
        layout = self.layout

        layout.operator(self.get_update_operator(), icon='URL')
        layout.separator()

        layout.label(text="Import/Export:")
        layout.prop(self, "scale_correction")
        layout.separator()

        layout.label(text="Texture Paths:")
        row = layout.row(align=True)
        row.template_list("PREFERENCES_UL_MW_TexturePathList", "", self, "texture_paths", self, "texture_paths_active_index")

        col = row.column(align=True)
        col.operator("preferences.mw_texture_paths_add", icon='ADD', text="")
        col.operator("preferences.mw_texture_paths_remove", icon='REMOVE', text="")
        col.separator()
        col.operator("preferences.mw_texture_paths_move", icon='TRIA_UP', text="").direction = -1
        col.operator("preferences.mw_texture_paths_move", icon='TRIA_DOWN', text="").direction = 1

    @classmethod
    def get_update_operator(cls):
        if cls.update_status == 'UPDATE_UNCHECKED':
            return "preferences.mw_update_check"
        if cls.update_status == 'UPDATE_AVAILABLE':
            return "preferences.mw_update_apply"
        if cls.update_status == 'UPDATE_INSTALLED':
            return "preferences.mw_update_notes"
        if cls.update_status == 'UPDATE_FORBIDDEN':
            return "preferences.mw_update_limit"

    @classmethod
    def from_context(cls, context):
        return context.preferences.addons[cls.bl_idname].preferences


# -------------
# IMPORT/EXPORT
# -------------

class ImportScene(bpy.types.Operator, ImportHelper):
    """Import a Morrowind NIF File"""

    bl_idname = "import_scene.mw"
    bl_options = {"PRESET", "UNDO"}
    bl_label = "Import NIF"

    # -- import helper --

    filename_ext = ".nif"
    filter_glob: bpy.props.StringProperty(default="*.nif", options={"HIDDEN"})

    # -- user config --

    vertex_precision: bpy.props.FloatProperty(
        name="Vertex Precision",
        description="Precision used when optimizing vertices. (Recommended: 0.001)",
        default=0.001,
        min=0.0001,
        max=1000.0,
    )

    attach_keyframe_data: bpy.props.BoolProperty(
        name="Attach Keyframe Data",
        description=(
            "Attach animations from the corresponding .kf file. Requires file names to be an exact match."
            "\n(e.g. when enabled importing 'xbase_anim.nif' will attach animations from 'xbase_anim.kf')"
        ),
        default=False,
    )

    discard_root_transforms: bpy.props.BoolProperty(
        name="Discard Root Transforms",
        description=(
            "Discard the root object's transformations. In-game root transforms are overwriten with the values"
            " provided by individual cell references. Despite this some meshes do define root transformations,"
            " which can lead to unintended results if accidentally applied before exporting"
        ),
        default=True,
    )

    use_existing_materials: bpy.props.BoolProperty(
        name="Use Existing Materials",
        description=(
            "Re-use existing materials from the blender scene if present (rather than creating new materials)."
            " Pre-existing material names must match the base texture name (without extension) to be eligible."
            " When enabled newly imported materials are automatically renamed to their base texture file name."
        ),
        default=False,
    )

    ignore_collision_nodes: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    ignore_custom_normals: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    ignore_animations: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        print(f"Blender Morrowind Plugin {bl_info['version']}")
        from . import nif_import
        kwargs = self.as_keywords(ignore=("filename_ext", "filter_glob", "check_existing"))
        return nif_import.load(context, **kwargs)


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
        return context.mode == 'OBJECT'

    def execute(self, context):
        print(f"Blender Morrowind Plugin {bl_info['version']}")
        from . import nif_export
        kwargs = self.as_keywords(ignore=("filename_ext", "filter_glob", "check_existing"))
        return nif_export.save(context, **kwargs)


# ------------
# OBJECT PANEL
# ------------

class ObjectPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_MW_ObjectPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "Morrowind"

    def draw(self, context):
        ob = context.active_object
        self.layout.prop(ob.mw, "object_flags")


# ---------------
# DOPESHEET PANEL
# ---------------

class MarkersList(bpy.types.UIList):
    bl_idname = "DOPESHEET_UL_MW_MarkersList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.alignment = 'LEFT'
        row.prop(item, "frame", text="", emboss=False)

        pose_marker_icon = 'PMARKER_SEL' if item.select else 'PMARKER'
        layout.prop(item, "name", text="", icon=pose_marker_icon, emboss=False)

    @staticmethod
    def get_selected(self):
        return self.pose_markers.active_index

    @staticmethod
    def set_selected(self, index):
        for m in self.pose_markers:
            m.select = False
        self.pose_markers.active_index = index
        self.pose_markers.active.select = True


class MarkersListSort(bpy.types.Operator):
    bl_idname = "marker.mw_markers_sort"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Sort Markers"
    bl_description = "Sort markers by their timings"

    def execute(self, context):
        try:
            markers = context.active_object.animation_data.action.pose_markers
        except:
            pass
        else:
            temp = [(m.frame, m.name) for m in markers]
            temp.sort()
            for m, t in zip(markers, temp):
                m.frame, m.name = t
        return {"FINISHED"}


class MarkersPanel(bpy.types.Panel):
    bl_idname = "DOPESHEET_PT_MW_MarkersPanel"
    bl_space_type = 'DOPESHEET_EDITOR'
    bl_region_type = 'UI'
    bl_label = "Morrowind"

    @classmethod
    def poll(cls, context):
        sd = context.space_data
        if sd.type == 'DOPESHEET_EDITOR' and sd.ui_mode == 'ACTION':
            try:
                return bool(context.active_object.animation_data.action)
            except:
                pass

    def draw(self, context):
        space_data = context.space_data

        try:
            action = context.active_object.animation_data.action
        except AttributeError:
            self.layout.template_ID(space_data, "action", new="action.new", unlink="action.unlink")
            return

        self.layout.prop(space_data, "show_pose_markers", text="Show Text Keys")

        # Markers List
        row = self.layout.row()
        row.template_list("DOPESHEET_UL_MW_MarkersList", "", action, "pose_markers", action, "active_pose_marker_index")
        row.enabled = space_data.show_pose_markers

        # Markers Operators
        col = row.column(align=True)
        col.operator("marker.add", icon='ADD', text="")
        col.operator("marker.delete", icon='REMOVE', text="")
        col.separator()
        col.operator("marker.mw_markers_sort", icon='SORTTIME', text="")
        col.enabled = space_data.show_pose_markers


# --------------
# MATERIAL PANEL
# --------------

class MaterialCreateShader(bpy.types.Operator):
    bl_idname = "material.mw_create_shader"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Create Morrowind Shader"
    bl_description = "Convert to (or create) a Morrowind shader"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None) and (ob.type == "MESH")

    def execute(self, context):
        from . import nif_shader
        nif_shader.execute(context.active_object)
        return {"FINISHED"}


class MaterialPanelBase:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    is_active = False

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None) and (ob.type == "MESH")


class MaterialSubPanelBase(MaterialPanelBase):
    bl_parent_id = "MATERIAL_PT_MW_MaterialPanel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return MaterialPanelBase.is_active


class MaterialPanel(bpy.types.Panel, MaterialPanelBase):
    bl_idname = "MATERIAL_PT_MW_MaterialPanel"
    bl_label = "Morrowind"

    def draw(self, context):
        ob = context.active_object
        material = context.material
        try:
            this = material.mw.validate()
            MaterialPanelBase.is_active = True
        except (AttributeError, TypeError):
            MaterialPanelBase.is_active = False
            self.layout.operator("material.mw_create_shader")


class MaterialPanelSettings(bpy.types.Panel, MaterialSubPanelBase):
    bl_idname = "MATERIAL_PT_MW_MaterialPanelSettings"
    bl_label = "Settings"

    def draw(self, context):
        ob = context.active_object
        this = context.material.mw
        self.layout.use_property_split = True
        self.draw_color_section(this, ob)
        self.draw_alpha_section(this, ob)

    def draw_color_section(self, this, ob):
        # Vertex Colors
        column = self.layout.column(align=True)
        column.prop(this, "use_vertex_colors")

        # Material Colors
        column = self.layout.column(align=True)

        # Ambient Color
        # column.prop(this.ambient_input, "default_value", text="Ambient Color")

        # Diffuse Color
        if this.use_vertex_colors:
            column.prop_search(this.vertex_color, "layer_name", ob.data, "vertex_colors", text="Diffuse Color")
        else:
            column.prop(this.diffuse_input, "default_value", text="Diffuse Color")

        # Specular Color
        # column.prop(this.specular_input, "default_value", text="Specular Color")

        # Emissive Color
        column.prop(this.emissive_input, "default_value", text="Emissive Color")

    def draw_alpha_section(self, this, ob):
        # Alpha Blend
        self.layout.prop(this, "use_alpha_blend")

        # Opacity
        column = self.layout.column(align=True)
        column.enabled = (this.use_alpha_blend and not this.use_vertex_colors)
        #
        if this.use_vertex_colors:
            column.prop_search(this.vertex_color, "layer_name", ob.data, "vertex_colors", text="Opacity")
        else:
            column.prop(this.opacity_input, "default_value", text="Opacity", slider=True)

        # Alpha Clip
        self.layout.prop(this, "use_alpha_clip")

        # Threshold
        column = self.layout.column(align=True)
        column.enabled = this.use_alpha_clip
        #
        column.prop(this.material, "alpha_threshold", text="Threshold")


class MaterialPanelTemplate(bpy.types.Panel, MaterialSubPanelBase):

    def draw(self, context):
        ob = context.active_object
        this = context.material.mw
        slot = this.texture_slots[self.bl_label]

        # pretty layout
        self.layout.use_property_split = True

        # require image
        self.layout.template_ID(slot, "image", new="image.new", open="image.open")
        if not slot.image:
            return

        # use mip maps
        self.layout.prop(slot, "use_mipmaps")

        # repeat image
        self.layout.prop(slot, "use_repeat")

        # uv map layer
        self.layout.prop_search(slot, "layer", ob.data, "uv_layers", text="UV Map")
        if slot.layer:
            self.layout.prop(slot.mapping_node.inputs[1], "default_value", text="Location")
            self.layout.prop(slot.mapping_node.inputs[3], "default_value", text="Scale")

    @classmethod
    def new(cls, label):
        cls_name = "MaterialPanel" + label.replace(" ", "")
        cls_dict = dict(bl_idname=f"MATERIAL_PT_MW_{cls_name}", bl_label=label)
        return type(cls)(cls_name, (cls,), cls_dict)


MaterialPanelTextures = (
    MaterialPanelTemplate.new("Base Texture"),
    MaterialPanelTemplate.new("Dark Texture"),
    MaterialPanelTemplate.new("Detail Texture"),
    MaterialPanelTemplate.new("Glow Texture"),
    MaterialPanelTemplate.new("Decal 0 Texture"),
    MaterialPanelTemplate.new("Decal 1 Texture"),
    MaterialPanelTemplate.new("Decal 2 Texture"),
    MaterialPanelTemplate.new("Decal 3 Texture"),
)


# ---------------
# PROPERTY GROUPS
# ---------------

class NiObjectProps(bpy.types.PropertyGroup):
    object_flags: bpy.props.IntProperty(name="Flags", min=0, max=65535, default=2)

class NiObjectBoneProps:
    object_flags = 12


# --------------
# MISC OPERATORS
# --------------

class CreateRadiusSphere(bpy.types.Operator):
    bl_idname = "mesh.mw_create_radius_sphere"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Create Radius Sphere"
    bl_description = "Create a sphere representing the selected meshes radius."

    def execute(self, context):
        for mesh in context.selected_objects:
            try:
                center, radius = self.get_center_radius(mesh.data.vertices)
            except (AttributeError, ValueError):
                continue

            ob = bpy.data.objects.new("Radius", None)
            ob.empty_display_type = 'SPHERE'
            ob.empty_display_size = 1.0
            ob.scale = [radius] * 3
            ob.location = center
            ob.parent = mesh

            context.scene.collection.objects.link(ob)

        return {'FINISHED'}

    @staticmethod
    def get_center_radius(vertices):
        # convert to numpy array
        import numpy as np
        array = np.empty((len(vertices), 3))
        vertices.foreach_get("co", array.ravel())
        # calc center and radius
        center = (array.min(axis=0) + array.max(axis=0)) / 2
        radius = np.linalg.norm(center - array, axis=1).max()
        return center, radius


# --------
# REGISTER
# --------
from . import nif_shader


classes = (
    UpdateCheck,
    UpdateApply,
    UpdateNotes,
    UpdateLimit,
    TexturePathList,
    TexturePath,
    TexturePathAdd,
    TexturePathRemove,
    TexturePathMove,
    Preferences,
    ImportScene,
    ExportScene,
    ObjectPanel,
    MarkersList,
    MarkersListSort,
    MarkersPanel,
    MaterialCreateShader,
    MaterialPanel,
    MaterialPanelSettings,
    *MaterialPanelTextures,
    NiObjectProps,
    CreateRadiusSphere,
    nif_shader.TextureSlot,
    nif_shader.NiMaterialProps,
)


def menu_func_import(self, context):
    self.layout.operator(ImportScene.bl_idname, text="Morrowind (.nif)")


def menu_func_export(self, context):
    self.layout.operator(ExportScene.bl_idname, text="Morrowind (.nif)")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

    bpy.types.Object.mw = bpy.props.PointerProperty(type=NiObjectProps)
    bpy.types.Material.mw = bpy.props.PointerProperty(type=nif_shader.NiMaterialProps)
    bpy.types.PoseBone.mw = NiObjectBoneProps
    bpy.types.Action.active_pose_marker_index = bpy.props.IntProperty(
        name="Active Pose Marker",
        get=MarkersList.get_selected,
        set=MarkersList.set_selected,
        options={'SKIP_SAVE'},
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

    del bpy.types.Object.mw
    del bpy.types.Material.mw
    del bpy.types.PoseBone.mw
    del bpy.types.Action.active_pose_marker_index
