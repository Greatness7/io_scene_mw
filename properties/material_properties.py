import bpy

from es3.nif.NiAlphaProperty import AlphaBlendFunction, AlphaTestFunction
from es3.utils.flags import get_bit_field, set_bit_field

from ..nif_shader import NodeTreeWrapper, get_current_mw_shader_tree

# the blend, test and no-sort fields of NiAlphaProperty.flags, and the modes a new material starts with:
# SRC_ALPHA / INV_SRC_ALPHA blending and GREATER testing, both disabled
ALPHA_MODE_MASK = 0x1DFE
ALPHA_FLAGS_DEFAULT = 0x10EC


def enum_items(enum):
    return tuple(
        (item.name, item.name.replace("_", " ").title(), "", item.value)
        for item in enum
    )


# the blend fields are 4 bits wide but only 0-10 are defined; files do use the rest, so they get items too
ALPHA_BLEND_ITEMS = enum_items(AlphaBlendFunction) + tuple(
    (f"INVALID_{value}", f"Invalid ({value})", "Not a defined blend function; kept as imported", value)
    for value in range(len(AlphaBlendFunction), 16)
)
ALPHA_TEST_ITEMS = enum_items(AlphaTestFunction)


def upgrade_materials():
    """Migrate every editable Morrowind material; the panel and export never migrate on their own."""
    for material in bpy.data.materials:
        if material.library is not None:
            continue
        try:
            material.mw.validate().upgrade()
        except TypeError:
            continue  # not a Morrowind material


@bpy.app.handlers.persistent
def upgrade_on_load(*args):
    upgrade_materials()


class TextureProperties(bpy.types.PropertyGroup):

    node_tree: bpy.props.PointerProperty(type=bpy.types.NodeTree)

    nodes = property(lambda self: self.node_tree.nodes)
    links = property(lambda self: self.node_tree.links)

    outputs = property(lambda self: self.id_data.node_tree.nodes["MW Inputs"].outputs)
    shader_inputs = property(lambda self: self.id_data.node_tree.nodes["MW Shader"].inputs)

    texture_node = property(lambda self: self.nodes[self.name])
    mapping_node = property(lambda self: self.get_chain_from_tx(self.texture_node)[2])

    # Texture Image
    def update_image(self, context):
        tx_node, image = self.texture_node, self.image

        if tx_node.image == image:  # image unchanged
            return
        elif image and not tx_node.image:  # image added
            self.update_links()
        elif tx_node.image and not image:  # image removed
            self.remove_links()

        tx_node.image = image

    image: bpy.props.PointerProperty(
        type=bpy.types.Image,
        update=update_image,
    )

    # UV Map Layer
    def update_layer(self, context):
        tx_node, layer = self.texture_node, self.layer

        if layer:
            self.update_chain_from_tx(tx_node, layer)
        else:
            self.remove_chain_from_tx(tx_node)

    layer: bpy.props.StringProperty(
        update=update_layer,
    )

    # Use Mipmaps
    def get_use_mipmaps(self):
        return self.texture_node.interpolation != "Closest"

    def set_use_mipmaps(self, value):
        self.texture_node.interpolation = "Linear" if value else "Closest"

    use_mipmaps: bpy.props.BoolProperty(
        name="Use Mipmaps",
        description="",
        get=get_use_mipmaps,
        set=set_use_mipmaps,
        options=set(),
    )

    # Use Repeat
    def get_use_repeat(self):
        return self.texture_node.extension == "REPEAT"

    def set_use_repeat(self, state):
        self.texture_node.extension = "REPEAT" if state else "CLIP"

    use_repeat: bpy.props.BoolProperty(
        name="Repeat Image",
        description="",
        get=get_use_repeat,
        set=set_use_repeat,
        options=set(),
    )

    # Utility Funcs
    def create_chain(self):
        # create nodes
        uv_node = self.nodes.new("ShaderNodeUVMap")
        vec_sub = self.nodes.new("ShaderNodeVectorMath")
        mapping = self.nodes.new("ShaderNodeMapping")
        vec_add = self.nodes.new("ShaderNodeVectorMath")

        # set defaults
        vec_sub.operation = "SUBTRACT"
        vec_sub.inputs[1].default_value = [0.5] * 3
        vec_add.inputs[1].default_value = [0.5] * 3

        # create links
        self.links.new(uv_node.outputs[0], vec_sub.inputs[0])
        self.links.new(vec_sub.outputs[0], mapping.inputs[0])
        self.links.new(mapping.outputs[0], vec_add.inputs[0])

        return uv_node, vec_sub, mapping, vec_add

    def update_chain_from_tx(self, tx_node, uv_name):
        # find an existing uv node for this uv map
        for uv_node in self.nodes_of_type("ShaderNodeUVMap"):
            if uv_node.uv_map == uv_name:
                break
        else:  # no existing node for uv map found
            uv_node, *_ = self.create_chain()
            uv_node.uv_map = uv_name

        vec_add = self.get_chain_from_uv(uv_node)[-1]
        self.links.new(vec_add.outputs[0], tx_node.inputs[0])

    def remove_chain_from_tx(self, tx_node):
        links = tx_node.inputs[0].links
        if not links:
            # the node chain is already removed
            return

        vec_add = links[0].from_node
        if len(vec_add.outputs[0].links) > 1:
            # there are others using this chain
            self.links.remove(links[0])
        else:
            # we are only link to this uv chain
            for node in self.get_chain_from_tx(tx_node):
                self.nodes.remove(node)

    def update_links(self):
        outputs = self.outputs
        inputs = self.shader_inputs
        links = self.id_data.node_tree.links
        for name in self.active_socket_names():
            try:
                links.new(outputs[name], inputs[name])
            except KeyError:
                pass
        if self.name == "Detail Texture" and "Raw Detail Texture" in inputs:
            links.new(outputs[self.name], inputs["Raw Detail Texture"])

    def remove_links(self):
        inputs = self.shader_inputs
        links = self.id_data.node_tree.links
        names = list(self.active_socket_names())
        if self.name == "Detail Texture" and "Raw Detail Texture" in inputs:
            names.append("Raw Detail Texture")
        for name in names:
            for outer_link in inputs[name].links:
                try:
                    links.remove(outer_link)
                except KeyError:
                    pass

    @staticmethod
    def get_chain_from_uv(uv_node):
        vec_sub = uv_node.outputs[0].links[0].to_node
        mapping = vec_sub.outputs[0].links[0].to_node
        vec_add = mapping.outputs[0].links[0].to_node
        return uv_node, vec_sub, mapping, vec_add

    @staticmethod
    def get_chain_from_tx(tx_node):
        vec_add = tx_node.inputs[0].links[0].from_node
        mapping = vec_add.inputs[0].links[0].from_node
        vec_sub = mapping.inputs[0].links[0].from_node
        uv_node = vec_sub.inputs[0].links[0].from_node
        return uv_node, vec_sub, mapping, vec_add

    def active_socket_names(self):
        outputs = self.texture_node.outputs
        return (link.to_socket.name for o in outputs for link in o.links)

    def nodes_of_type(self, bl_idname):
        return (node for node in self.nodes if node.bl_idname == bl_idname)


class MaterialProperties(bpy.types.PropertyGroup, NodeTreeWrapper):
    material: bpy.props.PointerProperty(type=bpy.types.Material)
    texture_slots: bpy.props.CollectionProperty(type=TextureProperties)
    alpha_flags: bpy.props.IntProperty(
        name="Alpha Flags",
        min=0,
        max=0xFFFF,
        default=ALPHA_FLAGS_DEFAULT,
    )
    shine: bpy.props.FloatProperty()

    @staticmethod
    def register():
        bpy.types.Material.mw = bpy.props.PointerProperty(type=MaterialProperties)
        if upgrade_on_load not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(upgrade_on_load)

    @staticmethod
    def unregister():
        if upgrade_on_load in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(upgrade_on_load)

    @property
    def needs_upgrade(self):
        return "Alpha Test Function" not in self.inputs or "Use Vertex Colors" not in self.inputs

    def upgrade(self):
        """One-time migration for materials using an older generated MW Shader node group."""
        if not self.needs_upgrade:
            return

        try:
            alpha_threshold = self.alpha_threshold
        except KeyError:
            alpha_threshold = getattr(self.material, "alpha_threshold", 133 / 255)

        # older versions ignored the mode fields, so a material authored in Blender stored them as zero,
        # which reads as ONE / ONE / ALWAYS; none of 8037 alpha properties in a 2026-09-23 install scan use it.
        # Only pre-alpha-rendering groups need that repair: raw-color upgrades must not reinterpret
        # valid ONE / ONE / ALWAYS authored in a current group.
        if "Alpha Test Function" not in self.inputs and (self.alpha_flags & ALPHA_MODE_MASK) == 0:
            self.alpha_flags |= ALPHA_FLAGS_DEFAULT

        had_raw_detail = "Raw Detail Texture" in self.inputs
        vertex_source = None
        if "Use Vertex Colors" not in self.inputs:
            for link in self.diffuse_input.links:
                if link.from_node == self.vertex_color:
                    vertex_source = link.from_socket
                    break
        self.shader.node_tree = get_current_mw_shader_tree()
        if vertex_source is not None:
            self.remove_link(self.diffuse_input)
            self.links.new(vertex_source, self.vertex_input)
            self.inputs["Use Vertex Colors"].default_value = 1.0
        if not had_raw_detail:
            detail = self.inputs["Detail Texture"]
            raw_detail = self.inputs["Raw Detail Texture"]
            if detail.is_linked:
                self.links.new(detail.links[0].from_socket, raw_detail)
            else:
                # The historical unlinked detail value is a synthetic raw neutral.
                def decode(c):
                    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                raw_detail.default_value = (*map(decode, detail.default_value[:3]), detail.default_value[3])
        self.alpha_threshold = alpha_threshold
        self.sync_alpha_state()

    # Vertex Colors
    def get_use_vertex_colors(self):
        try:
            # Stale materials are still readable until the explicit/load upgrade runs.
            link = self.inputs.get("Vertex Color", self.diffuse_input).links[0]
        except IndexError:
            return False
        return self.vertex_color == link.from_node

    def set_use_vertex_colors(self, value):
        if value:
            vc = bpy.context.active_object.data.vertex_colors
            self.vertex_color.layer_name = (vc.active or vc.new()).name
            self.links.new(self.vertex_input, self.vertex_color.outputs["Color"])
            self.links.new(self.opacity_input, self.vertex_color.outputs["Alpha"])
        else:
            self.remove_link(self.vertex_input)
            self.remove_link(self.inputs["Diffuse Alpha"])
        self.inputs["Use Vertex Colors"].default_value = float(value)

    use_vertex_colors: bpy.props.BoolProperty(
        name="Vertex Colors",
        description="Use vertex colors rather than material colors.",
        get=get_use_vertex_colors,
        set=set_use_vertex_colors,
        options=set(),
    )

    # Alpha Flags (source of truth)
    def apply_render_method(self):
        blending = bool(self.alpha_flags & 0x0001)
        if bpy.app.version >= (4, 2, 0):
            self.material.surface_render_method = 'BLENDED' if blending else 'DITHERED'
        elif blending:
            self.material.blend_method = 'BLEND'
        elif self.alpha_flags & 0x0200:
            self.material.blend_method = 'CLIP'
        else:
            self.material.blend_method = 'OPAQUE'

    def sync_alpha_state(self):
        blending = bool(self.alpha_flags & 0x0001)
        testing = bool(self.alpha_flags & 0x0200)
        self.alpha_blend_enabled = 1.0 if blending else 0.0
        self.additive_blend_enabled = 1.0 if self.alpha_blend_preview_mode == "ADDITIVE" else 0.0
        self.alpha_clip_enabled_input.default_value = 1.0 if testing else 0.0
        self.alpha_test_function_input.default_value = self.get_test_mode()
        self.apply_render_method()

    @property
    def alpha_blend_preview_mode(self):
        src = self.get_src_blend_mode()
        dst = self.get_dst_blend_mode()
        if src == AlphaBlendFunction.SRC_ALPHA and dst == AlphaBlendFunction.INV_SRC_ALPHA:
            return "STANDARD"
        if src == AlphaBlendFunction.SRC_ALPHA and dst == AlphaBlendFunction.ONE:
            return "ADDITIVE"
        return "UNSUPPORTED"

    # Alpha Blend
    def get_use_alpha_blend(self):
        return (self.alpha_flags & 0x0001) != 0

    def set_use_alpha_blend(self, value):
        if value:
            self.alpha_flags |= 0x0001
        else:
            self.alpha_flags &= ~0x0001
        self.sync_alpha_state()

    use_alpha_blend: bpy.props.BoolProperty(
        name="Enable Blending",
        description="Enable alpha blending",
        get=get_use_alpha_blend,
        set=set_use_alpha_blend,
        options=set(),
    )

    # Alpha Blend Modes
    def get_src_blend_mode(self):
        return get_bit_field(self.alpha_flags, 0x001E, 1)

    def set_src_blend_mode(self, value):
        self.alpha_flags = set_bit_field(self.alpha_flags, value, 0x001E, 1)
        self.sync_alpha_state()

    src_blend_mode: bpy.props.EnumProperty(
        name="Source Blend Mode",
        items=ALPHA_BLEND_ITEMS,
        get=get_src_blend_mode,
        set=set_src_blend_mode,
        options=set(),
    )

    def get_dst_blend_mode(self):
        return get_bit_field(self.alpha_flags, 0x01E0, 5)

    def set_dst_blend_mode(self, value):
        self.alpha_flags = set_bit_field(self.alpha_flags, value, 0x01E0, 5)
        self.sync_alpha_state()

    dst_blend_mode: bpy.props.EnumProperty(
        name="Destination Blend Mode",
        items=ALPHA_BLEND_ITEMS,
        get=get_dst_blend_mode,
        set=set_dst_blend_mode,
        options=set(),
    )

    # Alpha Clip
    def get_use_alpha_clip(self):
        return (self.alpha_flags & 0x0200) != 0

    def set_use_alpha_clip(self, value):
        if value:
            self.alpha_flags |= 0x0200
        else:
            self.alpha_flags &= ~0x0200
        self.sync_alpha_state()

    use_alpha_clip: bpy.props.BoolProperty(
        name="Enable Testing",
        description="Enable alpha testing",
        get=get_use_alpha_clip,
        set=set_use_alpha_clip,
        options=set(),
    )

    # Alpha Test Function
    def get_test_mode(self):
        return get_bit_field(self.alpha_flags, 0x1C00, 10)

    def set_test_mode(self, value):
        self.alpha_flags = set_bit_field(self.alpha_flags, value, 0x1C00, 10)
        self.sync_alpha_state()

    test_mode: bpy.props.EnumProperty(
        name="Alpha Test Function",
        items=ALPHA_TEST_ITEMS,
        get=get_test_mode,
        set=set_test_mode,
        options=set(),
    )

    # No Sort
    def get_no_sort(self):
        return (self.alpha_flags & 0x2000) != 0

    def set_no_sort(self, value):
        if value:
            self.alpha_flags |= 0x2000
        else:
            self.alpha_flags &= ~0x2000

    no_sort: bpy.props.BoolProperty(
        name="No Sort",
        description="Disable sorting for this alpha property",
        get=get_no_sort,
        set=set_no_sort,
        options=set(),
    )
