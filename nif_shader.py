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
    material.mw.sync_alpha_state()
    return material


def get_base_texture_image(mat):
    try:
        node = mat.node_tree.get_output_node('EEVEE')
        while node.bl_idname != "ShaderNodeTexImage":
            node = node.inputs[0].links[0].from_node
        return node.image
    except (AttributeError, LookupError):
        return None


def new_socket(node_tree, in_out, socket_type, name, default=None, min_value=None, max_value=None):
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
    if min_value is not None:
        socket.min_value = min_value
    if max_value is not None:
        socket.max_value = max_value
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


def clamp_factor_sockets(tree):
    """Ensure slider-exposed alpha factor inputs are clamped to 0-1 even on pre-fix MW Shader trees."""
    if tree is None or tree.library is not None:
        return

    names = ("Diffuse Alpha", "Alpha Clip Threshold")

    if bpy.app.version >= (4, 0, 0):
        sockets = [it for it in tree.interface.items_tree if it.in_out == 'INPUT' and it.name in names]
        for socket in sockets:
            socket.subtype = 'FACTOR'
    else:
        sockets = [socket for name in names if (socket := tree.inputs.get(name)) is not None]

    for socket in sockets:
        socket.min_value = 0.0
        socket.max_value = 1.0


def get_current_mw_shader_tree():
    """Keep stale groups aside until MaterialProperties.upgrade() migrates their users."""
    tree = bpy.data.node_groups.get("MW Shader")
    if tree and (not tree_has_input(tree, "Alpha Test Function") or not tree_has_input(tree, "Use Vertex Colors")):
        tree.name = "MW Shader.legacy"
        tree = None
    if tree is not None:
        clamp_factor_sockets(tree)
        return tree
    return build_mw_shader_tree()


def build_srgb_encode_tree():
    """The same piecewise input encoding as tes3_blend's native material recipe."""
    tree = bpy.data.node_groups.new("MW sRGB Encode", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color")
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')

    def vector_math(name, operation, column):
        node = nodes.new('ShaderNodeVectorMath')
        node.name = node.label = name
        node.operation = operation
        node.location = column * 200, 0
        return node

    threshold = vector_math("Raw Threshold", 'MAXIMUM', 1)
    threshold.inputs[1].default_value = (0.0031308,) * 3
    # Gamma is component-wise power too, and is available on the Blender 3.6 floor.
    power = nodes.new('ShaderNodeGamma')
    power.name = power.label = "Raw Power"
    power.location = 400, 0
    power.inputs[1].default_value = 1 / 2.4
    curve = vector_math("Raw Curve", 'MULTIPLY_ADD', 3)
    curve.inputs[1].default_value = (1.055,) * 3
    curve.inputs[2].default_value = (-0.055,) * 3
    below = vector_math("Raw Below Threshold", 'SUBTRACT', 2)
    below.location.y = -200
    encoded = vector_math("Raw Color", 'MULTIPLY_ADD', 4)
    encoded.inputs[1].default_value = (-12.92,) * 3
    # p(max(c,t)) - 12.92 * (max(c,t) - c), with no final encoding after combines.
    links.new(group_input.outputs["Color"], threshold.inputs[0])
    links.new(threshold.outputs[0], power.inputs[0])
    links.new(power.outputs[0], curve.inputs[0])
    links.new(threshold.outputs[0], below.inputs[0])
    links.new(group_input.outputs["Color"], below.inputs[1])
    links.new(below.outputs[0], encoded.inputs[0])
    links.new(curve.outputs[0], encoded.inputs[2])
    links.new(encoded.outputs[0], group_output.inputs["Color"])
    group_output.location = 1000, 0
    return tree


def build_srgb_decode_tree():
    """Decode raw NIF constants for display without changing stored/exported colors."""
    tree = bpy.data.node_groups.new("MW sRGB Decode", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color")
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')

    threshold = nodes.new('ShaderNodeVectorMath')
    threshold.operation = 'MAXIMUM'
    threshold.inputs[1].default_value = (0.04045,) * 3
    shifted = nodes.new('ShaderNodeVectorMath')
    shifted.operation = 'MULTIPLY_ADD'
    shifted.inputs[1].default_value = (1 / 1.055,) * 3
    shifted.inputs[2].default_value = (0.055 / 1.055,) * 3
    power = nodes.new('ShaderNodeGamma')
    power.inputs[1].default_value = 2.4
    below = nodes.new('ShaderNodeVectorMath')
    below.operation = 'SUBTRACT'
    decoded = nodes.new('ShaderNodeVectorMath')
    decoded.operation = 'MULTIPLY_ADD'
    decoded.inputs[1].default_value = (-1 / 12.92,) * 3
    # high(max(c,t)) - (max(c,t) - c) / 12.92, including the linear dark segment.
    links.new(group_input.outputs["Color"], threshold.inputs[0])
    links.new(threshold.outputs[0], shifted.inputs[0])
    links.new(shifted.outputs[0], power.inputs[0])
    links.new(threshold.outputs[0], below.inputs[0])
    links.new(group_input.outputs["Color"], below.inputs[1])
    links.new(below.outputs[0], decoded.inputs[0])
    links.new(power.outputs[0], decoded.inputs[2])
    links.new(decoded.outputs[0], group_output.inputs["Color"])
    for index, node in enumerate(nodes):
        node.location = index * 200, 0
    group_output.location = 1400, 0
    return tree


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
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Diffuse Alpha", default=1.0, min_value=0.0, max_value=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Ambient Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Diffuse Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Specular Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Emissive Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Glow Texture", default=(0, 0, 0, 1))
    # Keep this socket at the old Alpha Factor index. Blender preserves group-node
    # links/defaults by interface index when a material migrates to a new node tree.
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Blend Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Base Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Dark Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Clip Enabled")
    new_socket(
        tree,
        'INPUT',
        'NodeSocketFloatFactor',
        "Alpha Clip Threshold",
        default=133 / 255,
        min_value=0.0,
        max_value=1.0,
    )
    # New sockets must remain after the legacy interface prefix; see the index note above.
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Additive Blend Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Test Function", default=4.0)
    # Detail's unlinked linear 0.5 is a synthetic neutral, not a decoded texture sample.
    # A separate raw-mode source keeps it neutral without changing old defaults or indices.
    neutral_detail = ((0.5 + 0.055) / 1.055) ** 2.4
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Raw Detail Texture", default=(neutral_detail,) * 3 + (1,))
    # Material constants remain raw NIF values on their original sockets (including
    # animation targets). Vertex colors are already linear and have their own source.
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Vertex Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Use Vertex Colors")

    group_input = nodes.new('NodeGroupInput')
    attribute = nodes.new('ShaderNodeAttribute')
    attribute.name = "PreviewRawColor"
    attribute.attribute_type = 'OBJECT'
    attribute.attribute_name = "tes3_preview_raw"
    attribute.location = -600, 200
    encode_tree = bpy.data.node_groups.get("MW sRGB Encode") or build_srgb_encode_tree()
    colors = {}
    color_names = ["Base Texture", "Detail Texture", "Dark Texture"]
    color_names += [f"Decal {i} Texture" for i in range(4)]
    color_names += ["Vertex Color", "Glow Texture"]
    for index, name in enumerate(color_names):
        source = group_input.outputs[name]
        raw_source = group_input.outputs["Raw Detail Texture"] if name == "Detail Texture" else source
        encoded = nodes.new('ShaderNodeGroup')
        encoded.name = encoded.label = f"{name} Encode"
        encoded.node_tree = encode_tree
        encoded.location = -400, -index * 200
        mix = nodes.new('ShaderNodeMixRGB')
        mix.name = mix.label = f"{name} Preview"
        mix.blend_type = 'MIX'
        mix.location = -200, -index * 200
        links.new(raw_source, encoded.inputs["Color"])
        links.new(attribute.outputs["Fac"], mix.inputs[0])
        links.new(source, mix.inputs[1])
        links.new(encoded.outputs["Color"], mix.inputs[2])
        colors[name] = mix.outputs[0]

    decode_tree = bpy.data.node_groups.get("MW sRGB Decode") or build_srgb_decode_tree()
    for index, name in enumerate(["Diffuse Color", "Emissive Color"]):
        source = group_input.outputs[name]
        decoded = nodes.new('ShaderNodeGroup')
        decoded.name = decoded.label = f"{name} Decode"
        decoded.node_tree = decode_tree
        decoded.location = -400, -2000 - index * 200
        mix = nodes.new('ShaderNodeMixRGB')
        mix.name = mix.label = f"{name} Preview"
        mix.location = -200, -2000 - index * 200
        links.new(source, decoded.inputs["Color"])
        links.new(attribute.outputs["Fac"], mix.inputs[0])
        links.new(decoded.outputs["Color"], mix.inputs[1])
        links.new(source, mix.inputs[2])
        colors[name] = mix.outputs[0]

    def gi(name):
        return colors[name] if name in colors else group_input.outputs[name]

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

    # V is either decoded material diffuse or the separately converted vertex color.
    vertex_or_material = nodes.new('ShaderNodeMixRGB')
    vertex_or_material.name = vertex_or_material.label = "Diffuse Source"
    vertex_or_material.location = 800, 400
    links.new(gi("Use Vertex Colors"), vertex_or_material.inputs[0])
    links.new(gi("Diffuse Color"), vertex_or_material.inputs[1])
    links.new(gi("Vertex Color"), vertex_or_material.inputs[2])

    # Diffuse = T × V, under Blender lighting.
    diffuse_color = nodes.new('ShaderNodeVectorMath')
    diffuse_color.operation = 'MULTIPLY'
    diffuse_color.location = 1000, 400
    links.new(color, diffuse_color.inputs[0])
    links.new(vertex_or_material.outputs[0], diffuse_color.inputs[1])

    diffuse_bsdf = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse_bsdf.location = 1200, 400
    links.new(diffuse_color.outputs[0], diffuse_bsdf.inputs["Color"])

    # Self-illumination = T × E + G, keeping emissive RGB and texture modulation.
    emissive_color = nodes.new('ShaderNodeVectorMath')
    emissive_color.operation = 'MULTIPLY'
    emissive_color.location = 1000, 100
    links.new(color, emissive_color.inputs[0])
    links.new(gi("Emissive Color"), emissive_color.inputs[1])
    glow = nodes.new('ShaderNodeVectorMath')
    glow.operation = 'ADD'
    glow.location = 1200, 100
    links.new(emissive_color.outputs[0], glow.inputs[0])
    links.new(gi("Glow Texture"), glow.inputs[1])

    emission = nodes.new('ShaderNodeEmission')
    emission.location = 1400, 100
    links.new(glow.outputs[0], emission.inputs["Color"])
    # Game emission is visible, but does not become a light source for other objects.
    gate = nodes.new('ShaderNodeLightPath')
    gate.name = gate.label = "Emission Gate"
    gate.location = 1200, -100
    links.new(gate.outputs["Is Camera Ray"], emission.inputs["Strength"])

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

    # Alpha test predicates, in AlphaTestFunction enum order.
    always = nodes.new('ShaderNodeValue')
    always.outputs[0].default_value = 1.0
    never = nodes.new('ShaderNodeValue')
    never.outputs[0].default_value = 0.0

    less = nodes.new('ShaderNodeMath')
    less.operation = 'LESS_THAN'
    links.new(opacity.outputs[2], less.inputs[0])
    links.new(gi("Alpha Clip Threshold"), less.inputs[1])

    equal = nodes.new('ShaderNodeMath')
    equal.operation = 'COMPARE'
    equal.inputs[2].default_value = 0.5 / 255
    links.new(opacity.outputs[2], equal.inputs[0])
    links.new(gi("Alpha Clip Threshold"), equal.inputs[1])

    greater = nodes.new('ShaderNodeMath')
    greater.operation = 'GREATER_THAN'
    links.new(opacity.outputs[2], greater.inputs[0])
    links.new(gi("Alpha Clip Threshold"), greater.inputs[1])

    def invert(value):
        node = nodes.new('ShaderNodeMath')
        node.operation = 'SUBTRACT'
        node.inputs[0].default_value = 1.0
        links.new(value, node.inputs[1])
        return node.outputs[0]

    predicates = (
        always.outputs[0],
        less.outputs[0],
        equal.outputs[0],
        invert(greater.outputs[0]),
        greater.outputs[0],
        invert(equal.outputs[0]),
        invert(less.outputs[0]),
        never.outputs[0],
    )

    # Select predicate N by successively replacing the result when mode >= N.
    selected_test = predicates[0]
    for mode, predicate in enumerate(predicates[1:], 1):
        select_mode = nodes.new('ShaderNodeMath')
        select_mode.operation = 'GREATER_THAN'
        select_mode.inputs[1].default_value = mode - 0.5
        links.new(gi("Alpha Test Function"), select_mode.inputs[0])

        select_predicate = nodes.new('ShaderNodeMix')
        select_predicate.data_type = 'FLOAT'
        select_predicate.clamp_factor = True
        links.new(select_mode.outputs[0], select_predicate.inputs[0])
        links.new(selected_test, select_predicate.inputs[2])
        links.new(predicate, select_predicate.inputs[3])
        selected_test = select_predicate.outputs[0]

    test_pass = nodes.new('ShaderNodeMix')
    test_pass.data_type = 'FLOAT'
    test_pass.clamp_factor = True
    test_pass.inputs[2].default_value = 1.0
    links.new(gi("Alpha Clip Enabled"), test_pass.inputs[0])
    links.new(selected_test, test_pass.inputs[3])

    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.inputs["Color"].default_value = (1, 1, 1, 1)
    transparent.location = 1800, -500

    zero = nodes.new('ShaderNodeBsdfTransparent')
    zero.inputs["Color"].default_value = (0, 0, 0, 1)
    zero.location = 1800, -700

    transparency_weight = nodes.new('ShaderNodeMath')
    transparency_weight.operation = 'SUBTRACT'
    transparency_weight.use_clamp = True
    transparency_weight.inputs[0].default_value = 1.0
    links.new(opacity.outputs[2], transparency_weight.inputs[1])

    standard = nodes.new('ShaderNodeMixShader')
    links.new(transparency_weight.outputs[0], standard.inputs[0])
    links.new(surface.outputs[0], standard.inputs[1])
    links.new(transparent.outputs[0], standard.inputs[2])

    scaled_source = nodes.new('ShaderNodeMixShader')
    links.new(opacity.outputs[2], scaled_source.inputs[0])
    links.new(zero.outputs[0], scaled_source.inputs[1])
    links.new(surface.outputs[0], scaled_source.inputs[2])

    additive = nodes.new('ShaderNodeAddShader')
    links.new(transparent.outputs[0], additive.inputs[0])
    links.new(scaled_source.outputs[0], additive.inputs[1])

    blend_mode = nodes.new('ShaderNodeMixShader')
    links.new(gi("Additive Blend Enabled"), blend_mode.inputs[0])
    links.new(standard.outputs[0], blend_mode.inputs[1])
    links.new(additive.outputs[0], blend_mode.inputs[2])

    blend_or_opaque = nodes.new('ShaderNodeMixShader')
    links.new(gi("Alpha Blend Enabled"), blend_or_opaque.inputs[0])
    links.new(surface.outputs[0], blend_or_opaque.inputs[1])
    links.new(blend_mode.outputs[0], blend_or_opaque.inputs[2])

    mix_shader = nodes.new('ShaderNodeMixShader')
    mix_shader.location = 2200, 100
    links.new(test_pass.outputs[0], mix_shader.inputs[0])
    links.new(transparent.outputs[0], mix_shader.inputs[1])
    links.new(blend_or_opaque.outputs[0], mix_shader.inputs[2])

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
    vertex_input = property(lambda self: self.inputs["Vertex Color"])
    specular_input = property(lambda self: self.inputs["Specular Color"])
    emissive_input = property(lambda self: self.inputs["Emissive Color"])

    alpha_input = property(lambda self: self.inputs["Base Alpha"])
    opacity_input = property(lambda self: self.inputs["Diffuse Alpha"])
    alpha_blend_enabled_input = property(lambda self: self.inputs["Alpha Blend Enabled"])
    additive_blend_enabled_input = property(lambda self: self.inputs["Additive Blend Enabled"])
    alpha_clip_enabled_input = property(lambda self: self.inputs["Alpha Clip Enabled"])
    alpha_threshold_input = property(lambda self: self.inputs["Alpha Clip Threshold"])
    alpha_test_function_input = property(lambda self: self.inputs["Alpha Test Function"])

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
    def alpha_blend_enabled(self):
        return self.alpha_blend_enabled_input.default_value

    @alpha_blend_enabled.setter
    def alpha_blend_enabled(self, value):
        self.alpha_blend_enabled_input.default_value = value

    @property
    def additive_blend_enabled(self):
        return self.additive_blend_enabled_input.default_value

    @additive_blend_enabled.setter
    def additive_blend_enabled(self, value):
        self.additive_blend_enabled_input.default_value = value

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
