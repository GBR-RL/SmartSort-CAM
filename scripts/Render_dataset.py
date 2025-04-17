import bpy
import os
import random
from math import radians
from mathutils import Vector
import csv
# === CONFIG ===
CLASSES = ['bolt', 'gear', 'nut', 'washer', 'bearing', 'connector']
MODELS_DIR = "assests/STL"     
OUTPUT_DIR = "data/dataset_samples"      
RENDERS_PER_STL = 200                       
IMAGE_RESOLUTION = 512
SAMPLES = 512     # Try 128 for speed; increase if needed

# === ENABLE GPU RENDERING ===
def enable_gpu_rendering():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'  # or 'OPTIX' if supported
    for device in prefs.devices:
        if device.type in {'CUDA', 'OPTIX'}:
            device.use = True
            print(f"✅ Enabled GPU device: {device.name}")
    bpy.context.scene.cycles.device = 'GPU'
    for scene in bpy.data.scenes:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'

# === SETUP HDRI WITH WHITE CAMERA BACKGROUND (VARIABLE) ===
def setup_hdri_white_background(hdri_path, strength):
    """
    Sets up the world with an HDRI for lighting while forcing the camera‐visible background to be white.
    The 'strength' parameter is randomized per render (or per class) to vary the scene illumination.
    """
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # HDRI Environment Texture Node
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    try:
        env_tex.image = bpy.data.images.load(hdri_path)
    except Exception as e:
        print("Error loading HDRI:", e)
    env_tex.location = (-800, 300)

    # Background node for the HDRI
    bg_hdri = nodes.new('ShaderNodeBackground')
    bg_hdri.inputs['Strength'].default_value = strength  # strength randomized
    bg_hdri.location = (-400, 300)
    links.new(env_tex.outputs['Color'], bg_hdri.inputs['Color'])

    # White Background node for camera rays
    bg_white = nodes.new('ShaderNodeBackground')
    bg_white.inputs['Color'].default_value = (1, 1, 1, 1)
    bg_white.inputs['Strength'].default_value = 1.0
    bg_white.location = (-400, 0)

    # Light Path node: distinguishes camera rays from others
    lp = nodes.new('ShaderNodeLightPath')
    lp.location = (-800, 0)

    # Mix Shader to blend white background only (ignoring HDRI for visible output)
    mix_node = nodes.new('ShaderNodeMixShader')
    mix_node.location = (-200, 200)
    links.new(lp.outputs['Is Camera Ray'], mix_node.inputs['Fac'])
    links.new(bg_white.outputs['Background'], mix_node.inputs[1])
    links.new(bg_white.outputs['Background'], mix_node.inputs[2])

    # World Output node
    out_node = nodes.new('ShaderNodeOutputWorld')
    out_node.location = (0, 200)
    links.new(mix_node.outputs['Shader'], out_node.inputs['Surface'])

    bpy.context.scene.render.film_transparent = False

# === MINIMAL OVERHEAD LIGHT WITH RANDOMIZATION ===
def setup_minimal_overhead_light():
    """Adds one overhead area light with randomized energy to create subtle, varied highlights."""
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.object
    # Randomize light energy within a range
    light.data.energy = random.uniform(5, 10)
    # Randomize light size for variation in softness
    light.data.size = random.uniform(2.5, 3.5)

# === CLEANUP ===
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# === CAMERA + AUTO AIM ===
def setup_camera_aimed_at(target_obj):
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    bbox_size = target_obj.dimensions
    max_dim = max(bbox_size.x, bbox_size.y, bbox_size.z)
    radius = max_dim * 2.5  # Heuristic multiplier
    cam.data.lens = random.uniform(35, 70)  # 35mm = wide, 70mm = zoomed in

    import math
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(10), math.radians(40))
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    cam.location = (x, y, z)

    # Aim the camera at the object's origin
    direction = target_obj.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.clip_start = 0.001
    cam.data.clip_end = 1000
    bpy.context.scene.camera = cam

def apply_positional_jitter(obj, jitter_range=0.5):
    """
    Adds a small random offset to the object's position.
    jitter_range: Maximum offset (in scene units) along each axis.
    """
    offset = Vector((
        random.uniform(-jitter_range, jitter_range),
        random.uniform(-jitter_range, jitter_range),
        random.uniform(-jitter_range, jitter_range)
    ))
    obj.location += offset

# === IMPORT AND TWEAK OBJECT MATERIAL (STEEL TEXTURE) ===
def import_stl(filepath):
    # 1. Import STL
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = Vector((0, 0, 0))
    obj.rotation_euler = (
        radians(random.uniform(-10, 10)),
        radians(random.uniform(-10, 10)),
        radians(random.uniform(0, 360))
    )
    bpy.ops.object.shade_smooth()

    scale_factor = random.uniform(0.95, 1.1)
    obj.scale = (scale_factor, scale_factor, scale_factor)

    # 2. Apply positional jitter (if you still want this randomness)
    apply_positional_jitter(obj, jitter_range=0.5)

    # 3. Create a new material for the steel texture
    mat = bpy.data.materials.new(name="SteelMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 4. Get the Principled BSDF node
    bsdf = nodes.get("Principled BSDF")

    # 5. Add an Image Texture node
    steel_tex = nodes.new("ShaderNodeTexImage")
    # Update the path to steel texture file
    steel_tex.image = bpy.data.images.load("assets/texture/steel.png")
    steel_tex.location = (-300, 300)

    # 6. Connect the Image Texture output to BSDF Base Color
    links.new(steel_tex.outputs['Color'], bsdf.inputs['Base Color'])

    # 7. Principled BSDF for a metallic look
    bsdf.inputs['Metallic'].default_value = 1.0  
    bsdf.inputs['Roughness'].default_value = 0.9
    bsdf.inputs['Specular'].default_value = 0.2                

    # 8. Assign the material to your object
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return obj

# === RENDER SETTINGS ===
def configure_renderer(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_ambient_occlusion = True
    scene.render.resolution_x = IMAGE_RESOLUTION
    scene.render.resolution_y = IMAGE_RESOLUTION
    scene.render.resolution_percentage = 100

    # exposure and view transform settings
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.exposure = random.uniform(-0.1, -0.5)  # Slightly vary exposure

def render_image(part_class, model_name, index, tag):
    """
    Renders an image and saves it to the correct class/tag folder.
    Args:
        part_class: 'connector'
        model_name: 'bolt_01'
        index: integer count
        tag: 'good'
    """
    scene = bpy.context.scene
    configure_renderer(scene)

    filename = f"{part_class}_{model_name}_{tag}_{index:03d}.png"
    out_path = os.path.join(OUTPUT_DIR, part_class, tag, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


# === MAIN LOOP ===
def generate_samples_for_class(part_class):
    model_dir = os.path.join(MODELS_DIR, part_class)
    model_files = [f for f in os.listdir(model_dir) if f.lower().endswith(".stl")]
    if not model_files:
        print(f" No .stl files for '{part_class}'")
        return

    print(f"Generating for: {part_class} ({len(model_files)} models)")

    for model_name in model_files:
        model_path = os.path.join(model_dir, model_name)
        base_name = os.path.splitext(model_name)[0] 

        for i in range(RENDERS_PER_STL):
            clear_scene()
            setup_minimal_overhead_light()

            # Set HDRI
            random_hdri_strength = 0.8
            hdri_path = "assests/hdri/my_environment.hdr"
            setup_hdri_white_background(hdri_path, strength=random_hdri_strength)

            # Import and render
            obj = import_stl(model_path)
            setup_camera_aimed_at(obj)

            tag = "good"
            render_image(part_class, base_name, i, tag)



# === GENERATE CSV FILE ===

def generate_image_label_csv(output_dir, csv_filename="part_labels.csv"):
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "label"])
        for class_name in CLASSES:
            class_dir = os.path.join(output_dir, class_name, "good")
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    relative_path = os.path.join(class_name, "good", fname)
                    writer.writerow([relative_path.replace("\\", "/"), class_name])
    print(f"Label CSV written to: {csv_path}")


# === ENTRY POINT ===
enable_gpu_rendering()

for cls in CLASSES:
    generate_samples_for_class(cls)

generate_image_label_csv(OUTPUT_DIR)