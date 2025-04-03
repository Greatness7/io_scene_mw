import bpy


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
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return MaterialPanelBase.is_active


class MaterialPanel(bpy.types.Panel, MaterialPanelBase):
    bl_idname = "MATERIAL_PT_MW_MaterialPanel"
    bl_label = "Morrowind"

    def draw(self, context):
        material = context.material
        try:
            material.mw.validate()
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
        column = self.layout.column(align=True)

        # Vertex Colors
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
        column.enabled = this.use_alpha_blend and not this.use_vertex_colors
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
    def new(cls, label) -> type:
        cls_name = "MaterialPanel" + label.replace(" ", "")
        cls_dict = dict(bl_idname=f"MATERIAL_PT_MW_{cls_name}", bl_label=label)
        return type(cls)(cls_name, (cls,), cls_dict)


BaseTexturePanel = MaterialPanelTemplate.new("Base Texture")
DarkTexturePanel = MaterialPanelTemplate.new("Dark Texture")
DetailTexturePanel = MaterialPanelTemplate.new("Detail Texture")
GlowTexturePanel = MaterialPanelTemplate.new("Glow Texture")
Decal0TexturePanel = MaterialPanelTemplate.new("Decal 0 Texture")
Decal1TexturePanel = MaterialPanelTemplate.new("Decal 1 Texture")
Decal2TexturePanel = MaterialPanelTemplate.new("Decal 2 Texture")
Decal3TexturePanel = MaterialPanelTemplate.new("Decal 3 Texture")
