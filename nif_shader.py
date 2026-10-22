import bpy


def execute(mesh):
    name = getattr(mesh.active_material, "name", "")
    image = get_base_texture_image(mesh.active_material)

    material = create_material(name)

    try:
        slot = mesh.material_slots[mesh.active_material_index]
    except IndexError:
        mesh.data.materials.append(None)
        slot = mesh.material_slots[-1]

    slot.link = 'OBJECT'
    slot.material = material

    try:
        material.mw.base_texture.image = image
        material.mw.base_texture.layer = mesh.data.uv_layers.active.name
    except AttributeError:
        pass

    return material.mw


def create_material(name=""):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    if bpy.app.version < (4, 3, 0):
        material.shadow_method = "NONE"
    material.use_backface_culling = True
    material.show_transparent_back = False
    material.mw.material = material
    material.mw.reset_nodes()
    material.mw.apply_render_method()
    return material


def get_base_texture_image(mat):
    try:
        node = mat.node_tree.get_output_node('EEVEE')
        while node.bl_idname != "ShaderNodeTexImage":
            node = node.inputs[0].links[0].from_node
        return node.image
    except (AttributeError, LookupError):
        return None


def new_socket(node_tree, in_out, socket_type, name, default=None):
    if bpy.app.version >= (4, 0, 0):
        base_type, subtype = socket_type, None
        if socket_type == 'NodeSocketFloatFactor':
            base_type, subtype = 'NodeSocketFloat', 'FACTOR'
        socket = node_tree.interface.new_socket(name, in_out=in_out, socket_type=base_type)
        if subtype is not None:
            socket.subtype = subtype
    else:
        sockets = node_tree.inputs if in_out == 'INPUT' else node_tree.outputs
        socket = sockets.new(socket_type, name)
    if default is not None:
        socket.default_value = default
    return socket


def build_alpha_blend_tree():
    """Mixes Color1/Color2 by (Vector1 * Vector2), used for the 4 decal layers."""
    tree = bpy.data.node_groups.new("MW Alpha Blend", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links

    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")
    new_socket(tree, 'INPUT', 'NodeSocketVector', "Vector")
    new_socket(tree, 'INPUT', 'NodeSocketVector', "Vector")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color1", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color2", default=(1, 1, 1, 1))

    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')

    factor = nodes.new('ShaderNodeVectorMath')
    factor.operation = 'MULTIPLY'
    factor.location = 200, 0

    mix = nodes.new('ShaderNodeMix')
    mix.data_type, mix.blend_type = 'RGBA', 'MIX'
    mix.location = 400, 0

    links.new(group_input.outputs[0], factor.inputs[0])
    links.new(group_input.outputs[1], factor.inputs[1])
    links.new(factor.outputs[0], mix.inputs[0])  # Vector -> Factor
    links.new(group_input.outputs[2], mix.inputs[6])  # Color1 -> A
    links.new(group_input.outputs[3], mix.inputs[7])  # Color2 -> B
    links.new(mix.outputs[2], group_output.inputs[0])  # Result(Color) -> Color

    return tree


def tree_has_input(tree, name):
    if bpy.app.version >= (4, 0, 0):
        return any(it.name == name and it.in_out == 'INPUT' for it in tree.interface.items_tree)
    return name in tree.inputs


def get_current_mw_shader_tree():
    """Returns the shared 'MW Shader' tree, transparently rebuilding it if a stale
    (pre-alpha-clip-rework) copy is cached under that name -- e.g. from an older
    saved file. The stale tree is kept around (renamed aside) so materials still
    pointing at it can be migrated lazily via MaterialProperties.upgrade()."""
    tree = bpy.data.node_groups.get("MW Shader")
    if tree and not tree_has_input(tree, "Alpha Clip Threshold"):
        tree.name = "MW Shader.legacy"
        tree = None
    return tree or build_mw_shader_tree()


def build_mw_shader_tree():
    """Procedural equivalent of the old assets/shader.blend 'MW Shader' node group."""
    alpha_blend_tree = bpy.data.node_groups.get("MW Alpha Blend") or build_alpha_blend_tree()

    tree = bpy.data.node_groups.new("MW Shader", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links

    new_socket(tree, 'INPUT', 'NodeSocketColor', "Base Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Detail Texture", default=(0.5, 0.5, 0.5, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Dark Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 0 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 0 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 1 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 1 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 2 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 2 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 3 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 3 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Diffuse Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Ambient Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Diffuse Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Specular Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Emissive Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Glow Texture", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Factor")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Base Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Dark Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Clip Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Clip Threshold", default=133 / 255)

    group_input = nodes.new('NodeGroupInput')

    def gi(name):
        return group_input.outputs[name]

    # Base, Detail, Dark: (Base * Detail * 2) * Dark
    base_detail = nodes.new('ShaderNodeVectorMath')
    base_detail.operation = 'MULTIPLY'
    base_detail.location = 200, 200
    links.new(gi("Base Texture"), base_detail.inputs[0])
    links.new(gi("Detail Texture"), base_detail.inputs[1])

    base_detail_x2 = nodes.new('ShaderNodeVectorMath')
    base_detail_x2.operation = 'ADD'
    base_detail_x2.location = 400, 200
    links.new(base_detail.outputs[0], base_detail_x2.inputs[0])
    links.new(base_detail.outputs[0], base_detail_x2.inputs[1])

    base_combined = nodes.new('ShaderNodeVectorMath')
    base_combined.operation = 'MULTIPLY'
    base_combined.location = 600, 200
    links.new(base_detail_x2.outputs[0], base_combined.inputs[0])
    links.new(gi("Dark Texture"), base_combined.inputs[1])

    # Decal: 4x shared Alpha Blend subgroup, chained
    color = base_combined.outputs[0]
    for i, decal in enumerate(["Decal 0", "Decal 1", "Decal 2", "Decal 3"]):
        blend = nodes.new('ShaderNodeGroup')
        blend.name = blend.label = "Alpha Blend" if i == 0 else f"Alpha Blend.{i:03d}"
        blend.node_tree = alpha_blend_tree
        blend.location = 800, 200 - i * 200
        links.new(gi(f"{decal} Alpha"), blend.inputs[0])
        links.new(gi("Diffuse Alpha"), blend.inputs[1])
        links.new(color, blend.inputs[2])
        links.new(gi(f"{decal} Texture"), blend.inputs[3])
        color = blend.outputs[0]

    # Diffuse
    diffuse_color = nodes.new('ShaderNodeVectorMath')
    diffuse_color.operation = 'MULTIPLY'
    diffuse_color.location = 1000, 400
    links.new(color, diffuse_color.inputs[0])
    links.new(gi("Diffuse Color"), diffuse_color.inputs[1])

    diffuse_bsdf = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse_bsdf.location = 1200, 400
    links.new(diffuse_color.outputs[0], diffuse_bsdf.inputs["Color"])

    # Emissive
    emissive_bw = nodes.new('ShaderNodeRGBToBW')
    emissive_bw.location = 1000, 100
    links.new(gi("Emissive Color"), emissive_bw.inputs[0])

    emissive_mix = nodes.new('ShaderNodeMix')
    emissive_mix.data_type, emissive_mix.blend_type = 'RGBA', 'ADD'
    emissive_mix.location = 1200, 100
    links.new(emissive_bw.outputs[0], emissive_mix.inputs[0])  # Factor
    links.new(gi("Glow Texture"), emissive_mix.inputs[6])  # A

    emission = nodes.new('ShaderNodeEmission')
    emission.location = 1400, 100
    links.new(emissive_mix.outputs[2], emission.inputs["Color"])  # Result(Color)

    surface = nodes.new('ShaderNodeAddShader')
    surface.location = 1600, 300
    links.new(emission.outputs[0], surface.inputs[0])
    links.new(diffuse_bsdf.outputs[0], surface.inputs[1])

    # Alpha: opacity = Base Alpha * Dark Alpha * Diffuse Alpha
    base_dark_alpha = nodes.new('ShaderNodeMix')
    base_dark_alpha.data_type, base_dark_alpha.blend_type = 'RGBA', 'MULTIPLY'
    base_dark_alpha.location = 1000, -200
    base_dark_alpha.inputs[0].default_value = 1.0  # Factor
    links.new(gi("Base Alpha"), base_dark_alpha.inputs[6])
    links.new(gi("Dark Alpha"), base_dark_alpha.inputs[7])

    opacity = nodes.new('ShaderNodeMix')
    opacity.data_type, opacity.blend_type = 'RGBA', 'MULTIPLY'
    opacity.location = 1200, -200
    opacity.inputs[0].default_value = 1.0  # Factor
    links.new(base_dark_alpha.outputs[2], opacity.inputs[6])
    links.new(gi("Diffuse Alpha"), opacity.inputs[7])

    # Alpha Clip: opacity_out = opacity * mix(1, opacity > threshold, clip_enabled)
    clip_test = nodes.new('ShaderNodeMath')
    clip_test.operation = 'GREATER_THAN'
    clip_test.location = 1400, -300
    links.new(opacity.outputs[2], clip_test.inputs[0])
    links.new(gi("Alpha Clip Threshold"), clip_test.inputs[1])

    clip_gate = nodes.new('ShaderNodeMix')
    clip_gate.data_type = 'FLOAT'
    clip_gate.clamp_factor = True
    clip_gate.location = 1600, -300
    links.new(gi("Alpha Clip Enabled"), clip_gate.inputs[0])  # Factor
    clip_gate.inputs[2].default_value = 1.0  # A
    links.new(clip_test.outputs[0], clip_gate.inputs[3])  # B

    opacity_clipped = nodes.new('ShaderNodeMath')
    opacity_clipped.operation = 'MULTIPLY'
    opacity_clipped.location = 1800, -200
    links.new(opacity.outputs[2], opacity_clipped.inputs[0])
    links.new(clip_gate.outputs[0], opacity_clipped.inputs[1])

    coverage = nodes.new('ShaderNodeMath')
    coverage.operation = 'SUBTRACT'
    coverage.use_clamp = True
    coverage.location = 2000, -100
    links.new(gi("Alpha Factor"), coverage.inputs[0])
    links.new(opacity_clipped.outputs[0], coverage.inputs[1])

    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = 1600, -500

    mix_shader = nodes.new('ShaderNodeMixShader')
    mix_shader.location = 2200, 100
    links.new(coverage.outputs[0], mix_shader.inputs[0])  # Factor
    links.new(surface.outputs[0], mix_shader.inputs[1])  # opaque
    links.new(transparent.outputs[0], mix_shader.inputs[2])  # transparent

    material_output = nodes.new('ShaderNodeOutputMaterial')
    material_output.location = 2400, 100
    links.new(mix_shader.outputs[0], material_output.inputs[0])

    return tree


class NodeTreeWrapper:

    node_tree = property(lambda self: self.material.node_tree)

    nodes = property(lambda self: self.material.node_tree.nodes)
    links = property(lambda self: self.material.node_tree.links)

    shader = property(lambda self: self.nodes["MW Shader"])
    inputs = property(lambda self: self.shader.inputs)

    # NODES
    vertex_color = property(lambda self: self.nodes["Vertex Color"])

    # INPUTS
    ambient_input = property(lambda self: self.inputs["Ambient Color"])
    diffuse_input = property(lambda self: self.inputs["Diffuse Color"])
    specular_input = property(lambda self: self.inputs["Specular Color"])
    emissive_input = property(lambda self: self.inputs["Emissive Color"])

    alpha_input = property(lambda self: self.inputs["Base Alpha"])
    opacity_input = property(lambda self: self.inputs["Diffuse Alpha"])
    alpha_factor_input = property(lambda self: self.inputs["Alpha Factor"])
    alpha_clip_enabled_input = property(lambda self: self.inputs["Alpha Clip Enabled"])
    alpha_threshold_input = property(lambda self: self.inputs["Alpha Clip Threshold"])

    # COLORS
    ambient_color = property(lambda self: self.ambient_input.default_value)
    diffuse_color = property(lambda self: self.diffuse_input.default_value)
    specular_color = property(lambda self: self.specular_input.default_value)
    emissive_color = property(lambda self: self.emissive_input.default_value)

    # TEXTURE NODES
    texture_group = property(lambda self: self.nodes["MW Inputs"])
    texture_outputs = property(lambda self: self.texture_group.outputs)

    # TEXTURE SLOTS
    base_texture = property(lambda self: self.texture_slots["Base Texture"])
    dark_texture = property(lambda self: self.texture_slots["Dark Texture"])
    detail_texture = property(lambda self: self.texture_slots["Detail Texture"])
    glow_texture = property(lambda self: self.texture_slots["Glow Texture"])
    decal_0_texture = property(lambda self: self.texture_slots["Decal 0 Texture"])
    decal_1_texture = property(lambda self: self.texture_slots["Decal 1 Texture"])
    decal_2_texture = property(lambda self: self.texture_slots["Decal 2 Texture"])
    decal_3_texture = property(lambda self: self.texture_slots["Decal 3 Texture"])

    @property
    def alpha(self):
        return self.opacity_input.default_value

    @alpha.setter
    def alpha(self, value):
        self.opacity_input.default_value = value

    @property
    def alpha_factor(self):
        return self.alpha_factor_input.default_value

    @alpha_factor.setter
    def alpha_factor(self, value):
        self.alpha_factor_input.default_value = value

    @property
    def alpha_threshold(self):
        return self.alpha_threshold_input.default_value

    @alpha_threshold.setter
    def alpha_threshold(self, value):
        self.alpha_threshold_input.default_value = value

    def reset_nodes(self):
        self.nodes.clear()
        self.links.clear()

        # Shader Group
        self.create_shader_group("MW Shader", 0, 0)

        # Texture Groups
        self.create_texture_group("MW Inputs", -200, 0)

        # Vertex Color
        self.create_node("Vertex Color", "ShaderNodeVertexColor", -200, -400)

    def create_shader_group(self, name, x, y):
        group = self.create_node(name, "ShaderNodeGroup", x, y)
        group.node_tree = get_current_mw_shader_tree()

        return group

    def create_texture_group(self, name, x, y):
        group = self.create_node(name, 'ShaderNodeGroup', x, y)
        group.node_tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')

        nodes = group.node_tree.nodes
        links = group.node_tree.links

        # make nodes
        base_image = nodes.new('ShaderNodeTexImage')
        dark_image = nodes.new('ShaderNodeTexImage')
        deta_image = nodes.new('ShaderNodeTexImage')
        glow_image = nodes.new('ShaderNodeTexImage')
        dec0_image = nodes.new('ShaderNodeTexImage')
        dec1_image = nodes.new('ShaderNodeTexImage')
        dec2_image = nodes.new('ShaderNodeTexImage')
        dec3_image = nodes.new('ShaderNodeTexImage')

        # name nodes
        base_image.name = base_image.label = "Base Texture"
        dark_image.name = dark_image.label = "Dark Texture"
        deta_image.name = deta_image.label = "Detail Texture"
        glow_image.name = glow_image.label = "Glow Texture"
        dec0_image.name = dec0_image.label = "Decal 0 Texture"
        dec1_image.name = dec1_image.label = "Decal 1 Texture"
        dec2_image.name = dec2_image.label = "Decal 2 Texture"
        dec3_image.name = dec3_image.label = "Decal 3 Texture"

        # texture slots
        for node in nodes:
            slot = self.texture_slots.add()
            slot.node_tree = group.node_tree
            slot.name = node.name

        # outputs
        self.new_output_socket(group, 'NodeSocketColor', "Base Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Dark Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Detail Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Glow Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 0 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 1 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 2 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 3 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Base Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Dark Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 0 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 1 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 2 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 3 Alpha")

        # links
        output = nodes.new('NodeGroupOutput')
        links.new(base_image.outputs["Color"], output.inputs["Base Texture"])
        links.new(dark_image.outputs["Color"], output.inputs["Dark Texture"])
        links.new(deta_image.outputs["Color"], output.inputs["Detail Texture"])
        links.new(glow_image.outputs["Color"], output.inputs["Glow Texture"])
        links.new(dec0_image.outputs["Color"], output.inputs["Decal 0 Texture"])
        links.new(dec1_image.outputs["Color"], output.inputs["Decal 1 Texture"])
        links.new(dec2_image.outputs["Color"], output.inputs["Decal 2 Texture"])
        links.new(dec3_image.outputs["Color"], output.inputs["Decal 3 Texture"])
        links.new(base_image.outputs["Alpha"], output.inputs["Base Alpha"])
        links.new(dark_image.outputs["Alpha"], output.inputs["Dark Alpha"])
        links.new(dec0_image.outputs["Alpha"], output.inputs["Decal 0 Alpha"])
        links.new(dec1_image.outputs["Alpha"], output.inputs["Decal 1 Alpha"])
        links.new(dec2_image.outputs["Alpha"], output.inputs["Decal 2 Alpha"])
        links.new(dec3_image.outputs["Alpha"], output.inputs["Decal 3 Alpha"])

    def create_node(self, name, node_type, x, y):
        node = self.nodes.new(node_type)
        node.name = node.label = name
        node.location = x, y
        return node

    def create_link(self, src_node, dst_node, src_key=0, dst_key=0):
        return self.links.new(src_node.outputs[src_key], dst_node.inputs[dst_key])

    def remove_link(self, socket):
        for link in socket.links:
            self.links.remove(link)

    def validate(self):
        m = self.material
        if not (m and m.use_nodes):
            raise TypeError("Invalid Material: use_nodes not enabled")
        if m.node_tree.get_output_node('EEVEE'):
            raise TypeError("Invalid Material: eevee output override")
        if "MW Shader" not in m.node_tree.nodes:
            raise TypeError("Invalid Material: mw_shader not present")
        return self

    @staticmethod
    def new_output_socket(group, socket_type, name):
        if bpy.app.version >= (4, 0, 0):
            group.node_tree.interface.new_socket(name, in_out='OUTPUT', socket_type=socket_type)
        else:
            group.node_tree.outputs.new(socket_type, name)
