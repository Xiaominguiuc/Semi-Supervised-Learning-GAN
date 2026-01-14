bl_info = {
    "name": "UAV Tile Capture (Blender 4.5)",
    "author": "ChatGPT & Improved",
    "version": (1, 3, 0),
    "blender": (4, 5, 0),
    "location": "3D View > Sidebar > UAV Capture",
    "description": "Scan GLB tiles, generate waypoints, dynamic load/unload tiles, capture images + camera metadata. Compatible with Blender 4.5.",
    "category": "Import-Export"
}

import bpy, os, json, math, csv, sys, argparse, traceback
from mathutils import Vector
from bpy.props import (
    StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
)
from bpy.types import Operator, Panel, PropertyGroup

# ---------------------------
# Global state for tile tracking (runtime only)
# ---------------------------
_tile_runtime_state = {
    "records": [],       # list of tile record dicts
    "index_loaded": False,
}

def get_tile_records():
    return _tile_runtime_state["records"]

def set_tile_records(records):
    _tile_runtime_state["records"] = records
    _tile_runtime_state["index_loaded"] = True

def clear_tile_records():
    _tile_runtime_state["records"] = []
    _tile_runtime_state["index_loaded"] = False

# ---------------------------
# Compatibility helpers for Blender 4.5
# ---------------------------
def available_engine_enums():
    """Return list of available engine enum keys in this Blender build."""
    try:
        return bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys()
    except Exception:
        return ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"]

def get_eevee_engine():
    """Return the correct Eevee engine string for this Blender build."""
    enums = available_engine_enums()
    if "BLENDER_EEVEE_NEXT" in enums:
        return "BLENDER_EEVEE_NEXT"
    if "BLENDER_EEVEE" in enums:
        return "BLENDER_EEVEE"
    return "BLENDER_WORKBENCH"

def set_render_engine_by_choice(choice):
    """Set scene.render.engine given choice in ('AUTO','EEVEE','CYCLES','WORKBENCH'). Returns final engine string."""
    enums = available_engine_enums()
    if choice == 'AUTO' or choice == 'EEVEE':
        if "BLENDER_EEVEE_NEXT" in enums:
            eng = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in enums:
            eng = "BLENDER_EEVEE"
        else:
            eng = "BLENDER_WORKBENCH"
    elif choice == 'CYCLES':
        eng = "CYCLES" if "CYCLES" in enums else get_eevee_engine()
    elif choice == 'WORKBENCH':
        eng = "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in enums else get_eevee_engine()
    else:
        eng = get_eevee_engine()
    try:
        bpy.context.scene.render.engine = eng
    except Exception:
        bpy.context.scene.render.engine = get_eevee_engine()
        eng = bpy.context.scene.render.engine
    return eng

# ---------------------------
# Light memory helpers
# ---------------------------
def purge_unused():
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass
    for img in list(bpy.data.images):
        if img.users == 0:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            try:
                bpy.data.materials.remove(mat)
            except Exception:
                pass

def import_glb(filepath):
    """Import .glb/.gltf and return list of newly created object names."""
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = set(bpy.context.scene.objects)
    new_objs = list(after - before)
    return [o.name for o in new_objs]

def remove_objects_by_names(names):
    for n in names:
        o = bpy.data.objects.get(n)
        if o:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
    purge_unused()

def bbox_center_of_names(names):
    """Calculate bounding box center for given object names."""
    bbox_min = Vector((1e9, 1e9, 1e9))
    bbox_max = Vector((-1e9, -1e9, -1e9))
    found = False
    for n in names:
        o = bpy.data.objects.get(n)
        if not o or o.type != 'MESH':
            continue
        found = True
        for v in o.bound_box:
            co = o.matrix_world @ Vector(v)
            # FIX: use list comprehension instead of generator
            bbox_min = Vector([min(bbox_min[i], co[i]) for i in range(3)])
            bbox_max = Vector([max(bbox_max[i], co[i]) for i in range(3)])
    if not found:
        return None
    return (bbox_min + bbox_max) / 2.0

# ---------------------------
# Tile scanning / indexing
# ---------------------------
def find_all_glb(root):
    glbs = []
    for base, dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".glb", ".gltf")):
                glbs.append(os.path.join(base, f))
    glbs.sort()
    return glbs

def compute_tile_centers(root, limit=None, progress_cb=None):
    files = find_all_glb(root)
    if limit and limit > 0:
        files = files[:limit]
    tiles = {}
    total = len(files)
    for i, f in enumerate(files):
        rel = os.path.relpath(f, root)
        try:
            names = import_glb(f)
            center = bbox_center_of_names(names)
            remove_objects_by_names(names)
            if center is None:
                center = Vector((0.0, 0.0, 0.0))
            tiles[rel] = [float(center.x), float(center.y), float(center.z)]
        except Exception as e:
            tiles[rel] = [0.0, 0.0, 0.0]
            print(f"Error scanning {f}: {e}")
        if progress_cb:
            try:
                progress_cb(i + 1, total)
            except Exception:
                pass
    return tiles

def save_tiles_index(root, tiles):
    path = os.path.join(root, "tiles_index.json")
    with open(path, "w") as f:
        json.dump(tiles, f, indent=2)
    return path

def load_tiles_index(root):
    path = os.path.join(root, "tiles_index.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ---------------------------
# Waypoint generation
# ---------------------------
def generate_waypoints(mode, n, radius, height, csv_path=None):
    wps = []
    if mode == "circle":
        for i in range(n):
            theta = 2 * math.pi * i / n
            wps.append(Vector((radius * math.cos(theta), radius * math.sin(theta), height)))
    elif mode == "line":
        for i in range(n):
            t = i / (max(1, n - 1))
            wps.append(Vector((-radius + 2 * radius * t, 0.0, height)))
    elif mode == "grid":
        side = int(math.sqrt(n))
        if side < 1:
            side = 1
        step = (2 * radius) / max(1, side - 1)
        for i in range(side):
            for j in range(side):
                wps.append(Vector((-radius + i * step, -radius + j * step, height)))
    elif mode == "snake":
        side = int(math.sqrt(n))
        if side < 1:
            side = 1
        step = (2 * radius) / max(1, side - 1)
        idx = 0
        for i in range(side):
            row = range(side) if i % 2 == 0 else range(side - 1, -1, -1)
            for j in row:
                if idx >= n:
                    break
                wps.append(Vector((-radius + j * step, -radius + i * step, height)))
                idx += 1
    elif mode == "csv" and csv_path:
        try:
            with open(csv_path, 'r') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = [p.strip() for p in line.split(',') if p.strip() != '']
                    if len(parts) < 3:
                        print(f"CSV line {line_num}: insufficient columns, skipping")
                        continue
                    try:
                        wps.append(Vector((float(parts[0]), float(parts[1]), float(parts[2]))))
                    except ValueError as e:
                        print(f"CSV line {line_num}: parse error {e}, skipping")
        except Exception as e:
            print(f"Error reading CSV waypoints: {e}")
    elif mode == "manual":
        # Collect waypoints from scene objects marked as uav_waypoint
        wps = collect_manual_waypoints()
    else:
        raise ValueError(f"Unsupported waypoint mode: {mode}")
    return wps

def collect_manual_waypoints():
    """Collect waypoints from scene objects with uav_waypoint custom property."""
    wps = []
    waypoint_objs = []
    for o in bpy.context.scene.objects:
        if o.get("uav_waypoint") == True:
            waypoint_objs.append(o)
    # Sort by name to maintain order
    waypoint_objs.sort(key=lambda x: x.name)
    for o in waypoint_objs:
        wps.append(o.location.copy())
    return wps

# ---------------------------
# Camera intrinsics helper
# ---------------------------
def camera_intrinsics(cam_obj, res_x, res_y):
    cam = cam_obj.data
    f_mm = cam.lens if cam.lens else 35.0
    sensor_w = cam.sensor_width if cam.sensor_width else 36.0
    sensor_h = cam.sensor_height if cam.sensor_height else 24.0
    if cam.sensor_fit == 'VERTICAL':
        s_u = res_y / sensor_h
        s_v = res_y / sensor_h
    else:
        s_u = res_x / sensor_w
        s_v = res_x / sensor_w
    fx = f_mm * s_u
    fy = f_mm * s_v
    cx = res_x * (0.5 - cam.shift_x)
    cy = res_y * (0.5 + cam.shift_y)
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return {"K": K, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

# ---------------------------
# Tile visualization helpers
# ---------------------------
def create_tile_visual_material(name, color):
    """Create or get a material with given color for tile visualization."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = False
        mat.diffuse_color = color
    else:
        mat.diffuse_color = color
    return mat

def get_or_create_uav_collection(name="UAV_Visualization"):
    """Get or create a collection for UAV visualization objects."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def update_tile_visuals():
    """Update tile center visualizations based on current tile records."""
    records = get_tile_records()
    col = get_or_create_uav_collection("UAV_TileCenters")

    # Colors for loaded/unloaded states
    color_loaded = (0.0, 1.0, 0.0, 1.0)    # Green
    color_unloaded = (1.0, 0.5, 0.0, 1.0)  # Orange
    color_error = (1.0, 0.0, 0.0, 1.0)     # Red

    mat_loaded = create_tile_visual_material("UAV_TileLoaded", color_loaded)
    mat_unloaded = create_tile_visual_material("UAV_TileUnloaded", color_unloaded)

    for rec in records:
        vis_name = f"tile_vis_{rec['rel'].replace('/', '_').replace('.', '_')}"
        vis_obj = bpy.data.objects.get(vis_name)

        if vis_obj is None:
            # Create a small sphere mesh for visualization
            mesh = bpy.data.meshes.new(vis_name + "_mesh")
            bm_verts = []
            # Create icosphere manually (simplified - just use empty with sphere display)
            vis_obj = bpy.data.objects.new(vis_name, None)
            vis_obj.empty_display_type = 'SPHERE'
            vis_obj.empty_display_size = 2.0
            vis_obj["uav_tile_visual"] = True
            vis_obj["tile_rel"] = rec['rel']
            col.objects.link(vis_obj)

        # Update position
        vis_obj.location = rec['center']

        # Update color based on loaded state
        if rec.get('loaded', False):
            vis_obj.color = color_loaded
        else:
            vis_obj.color = color_unloaded

def clear_tile_visuals():
    """Remove all tile visualization objects."""
    col = bpy.data.collections.get("UAV_TileCenters")
    if col:
        for obj in list(col.objects):
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        try:
            bpy.data.collections.remove(col)
        except Exception:
            pass

def clear_waypoint_visuals():
    """Remove all waypoint visualization objects."""
    for o in list(bpy.data.objects):
        if o.get("uav_waypoint") == True:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

# ---------------------------
# Capture routine (UI + headless)
# ---------------------------
def run_capture(tiles_root, out_dir, waypoint_mode="circle", n_waypoints=36, radius=20.0, height=10.0,
                load_radius=50.0, unload_margin=10.0, max_loaded=8, decimate_ratio=0.0,
                image_width=1920, image_height=1080, image_format='PNG', engine_choice='AUTO',
                csv_waypoint=None, start_index=0, tiles_limit_scan=0, headless=False,
                skip_tile_loading=False, use_manual_waypoints=False, look_at_center=None,
                progress_cb=None):
    """
    Main capture routine.

    Args:
        skip_tile_loading: If True, skip all tile loading and use existing scene objects
        use_manual_waypoints: If True, use manual waypoints from scene instead of generating
        look_at_center: Optional Vector for camera look-at target (default: origin)
        progress_cb: Optional callback(current, total, message) for progress updates
    """
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    eng = set_render_engine_by_choice(engine_choice)
    print(f"Render engine: {eng}")

    # Initialize tile records (only if not skipping tile loading)
    tile_records = []
    if not skip_tile_loading:
        if tiles_root and os.path.isdir(tiles_root):
            index = load_tiles_index(tiles_root)
            if index is None:
                print("tiles_index.json not found, computing centers (this may take time)...")
                index = compute_tile_centers(tiles_root, limit=(tiles_limit_scan or None),
                                            progress_cb=lambda i, t: progress_cb(i, t, "Scanning tiles") if progress_cb else None)
                save_tiles_index(tiles_root, index)

            for rel, center in index.items():
                tile_records.append({
                    "rel": rel,
                    "path": os.path.join(tiles_root, rel),
                    "center": Vector(center),
                    "loaded": False,
                    "objs": []
                })
            set_tile_records(tile_records)
            print(f"Loaded index with {len(tile_records)} tiles")
        else:
            print("No valid tiles folder, skipping tile loading")
            skip_tile_loading = True
    else:
        print("Tile loading skipped - using existing scene objects")

    scene = bpy.context.scene
    scene.render.engine = eng
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height
    scene.render.image_settings.file_format = image_format

    cam = scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new("UAV_CAMERA_DATA")
        cam = bpy.data.objects.new("UAV_CAMERA", cam_data)
        bpy.context.collection.objects.link(cam)
        scene.camera = cam
    # ensure lens & sensor
    cam.data.lens = cam.data.lens or 35.0
    cam.data.sensor_width = cam.data.sensor_width or 36.0
    cam.data.sensor_height = cam.data.sensor_height or 24.0

    # Generate or collect waypoints
    if use_manual_waypoints or waypoint_mode == "manual":
        wps = collect_manual_waypoints()
        if not wps:
            raise ValueError("No manual waypoints found in scene. Add objects with 'uav_waypoint' property.")
        print(f"Using {len(wps)} manual waypoints from scene")
    else:
        wps = generate_waypoints(waypoint_mode, n_waypoints, radius, height, csv_waypoint)
        print(f"Generated {len(wps)} waypoints in '{waypoint_mode}' mode")

    if not wps:
        raise ValueError("No waypoints available for capture")

    # Look-at target
    target = look_at_center if look_at_center else Vector((0.0, 0.0, 0.0))

    def load_tiles_near(pos):
        if skip_tile_loading:
            return
        loaded = sum(1 for t in tile_records if t["loaded"])
        candidates = []
        for t in tile_records:
            if t["loaded"]:
                continue
            d = (t["center"] - pos).length
            if d <= load_radius:
                candidates.append((d, t))
        candidates.sort(key=lambda x: x[0])
        for d, t in candidates:
            if loaded >= max_loaded:
                break
            if os.path.exists(t["path"]):
                try:
                    names = import_glb(t["path"])
                    # decimate if requested
                    if decimate_ratio and decimate_ratio > 0.0:
                        for nm in names:
                            o = bpy.data.objects.get(nm)
                            if o and o.type == 'MESH':
                                try:
                                    mod = o.modifiers.new("DECIMATE_UAV", type='DECIMATE')
                                    mod.ratio = decimate_ratio
                                    bpy.context.view_layer.objects.active = o
                                    o.select_set(True)
                                    bpy.ops.object.modifier_apply(modifier=mod.name)
                                    o.select_set(False)
                                except Exception as e:
                                    print(f"Decimate fail for {nm}: {e}")
                    t["loaded"] = True
                    t["objs"] = names
                    loaded += 1
                    print(f"Loaded tile: {t['rel']} ({loaded}/{max_loaded})")
                except Exception as e:
                    print(f"Failed to load tile {t['rel']}: {e}")
        # Update visualization
        if bpy.data.collections.get("UAV_TileCenters"):
            update_tile_visuals()

    def unload_far(pos):
        if skip_tile_loading:
            return
        for t in tile_records:
            if not t["loaded"]:
                continue
            d = (t["center"] - pos).length
            if d > (load_radius + unload_margin) or sum(1 for tt in tile_records if tt["loaded"]) > max_loaded:
                remove_objects_by_names(t["objs"])
                t["objs"] = []
                t["loaded"] = False
                print(f"Unloaded tile: {t['rel']}")
        purge_unused()
        # Update visualization
        if bpy.data.collections.get("UAV_TileCenters"):
            update_tile_visuals()

    metadata = []
    idx = start_index
    total_wps = len(wps)

    for wp_idx, pos in enumerate(wps):
        cam.location = pos
        dir_vec = (target - pos)
        if dir_vec.length > 0.0:
            cam.rotation_quaternion = dir_vec.to_track_quat('-Z', 'Y')

        load_tiles_near(pos)
        unload_far(pos)

        img_name = f"img_{idx:06d}.png"
        img_path = os.path.join(images_dir, img_name)
        scene.render.filepath = img_path

        if progress_cb:
            progress_cb(wp_idx + 1, total_wps, f"Rendering {img_name}")

        bpy.ops.render.render(write_still=True)

        intr = camera_intrinsics(cam, image_width, image_height)
        cam2world = [list(row) for row in cam.matrix_world]
        world2cam = [list(row) for row in cam.matrix_world.inverted()]
        item = {
            "image": img_name,
            "index": idx,
            "cam_location": [float(x) for x in cam.location],
            "cam_rotation_euler": [float(x) for x in cam.rotation_euler],
            "K": intr["K"],
            "fx": intr["fx"],
            "fy": intr["fy"],
            "cx": intr["cx"],
            "cy": intr["cy"],
            "cam2world": cam2world,
            "world2cam": world2cam
        }
        metadata.append(item)

        # Save metadata periodically
        if idx % 10 == 0:
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        idx += 1
        print(f"Captured {wp_idx + 1}/{total_wps}: {img_name}")

    # Final metadata save
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if metadata:
        keys = list(metadata[0].keys())
        with open(os.path.join(out_dir, "metadata.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for it in metadata:
                row = [json.dumps(it[k]) if isinstance(it[k], (list, dict)) else it[k] for k in keys]
                writer.writerow(row)

    print(f"Capture finished. {len(metadata)} images saved to: {images_dir}")
    return True

# ---------------------------
# Blender UI: properties & operators
# ---------------------------
class UAVProps(PropertyGroup):
    tiles_folder: StringProperty(name="Tiles Folder", subtype='DIR_PATH', default="")
    out_dir: StringProperty(name="Output Folder", subtype='DIR_PATH', default="")
    engine_choice: EnumProperty(name="Engine", items=[
        ('AUTO', 'Auto', 'Automatically select best available engine'),
        ('EEVEE', 'Eevee', 'Use Eevee renderer'),
        ('CYCLES', 'Cycles', 'Use Cycles renderer'),
        ('WORKBENCH', 'Workbench', 'Use Workbench renderer')
    ], default='AUTO')
    width: IntProperty(name="Width", default=1920, min=1, max=8192)
    height: IntProperty(name="Height", default=1080, min=1, max=8192)
    image_format: EnumProperty(name="Image Format", items=[
        ('PNG', 'PNG', 'PNG format'),
        ('JPEG', 'JPEG', 'JPEG format')
    ], default='PNG')
    waypoint_mode: EnumProperty(name="Waypoint Mode", items=[
        ('circle', 'Circle', 'Generate waypoints in a circle'),
        ('line', 'Line', 'Generate waypoints in a line'),
        ('grid', 'Grid', 'Generate waypoints in a grid'),
        ('snake', 'Snake', 'Generate waypoints in a snake pattern'),
        ('csv', 'From CSV', 'Load waypoints from CSV file'),
        ('manual', 'Manual', 'Use manually placed waypoints in scene')
    ], default='circle')
    n_waypoints: IntProperty(name="Waypoints", default=36, min=1, max=10000)
    radius: FloatProperty(name="Radius", default=20.0, min=0.1)
    height_m: FloatProperty(name="Height", default=10.0)
    csv_waypoint_file: StringProperty(name="Waypoint CSV", subtype='FILE_PATH', default="")
    load_radius: FloatProperty(name="Load radius", default=50.0, min=1.0)
    unload_margin: FloatProperty(name="Unload margin", default=10.0, min=0.0)
    max_loaded: IntProperty(name="Max loaded tiles", default=8, min=1, max=100)
    decimate_ratio: FloatProperty(name="Decimate ratio (0=skip)", default=0.0, min=0.0, max=1.0)
    compute_limit: IntProperty(name="Pre-scan limit (0=all)", default=0, min=0)
    auto_compute: BoolProperty(name="Auto compute centers if missing", default=True)
    headless_mode: BoolProperty(name="Headless mode", default=False)

    # New properties for improved workflow
    skip_tile_loading: BoolProperty(
        name="Skip Tile Loading",
        description="Use existing scene objects instead of loading tiles dynamically",
        default=False
    )
    use_manual_waypoints: BoolProperty(
        name="Use Manual Waypoints",
        description="Use waypoints placed manually in the scene (objects with 'uav_waypoint' property)",
        default=False
    )
    show_tile_visuals: BoolProperty(
        name="Show Tile Centers",
        description="Display tile center points in the viewport",
        default=True
    )
    look_at_x: FloatProperty(name="X", default=0.0, description="Camera look-at target X")
    look_at_y: FloatProperty(name="Y", default=0.0, description="Camera look-at target Y")
    look_at_z: FloatProperty(name="Z", default=0.0, description="Camera look-at target Z")

# ---------------------------
# Operators
# ---------------------------
class UAV_OT_ScanTiles(Operator):
    bl_idname = "uav.scan_tiles"
    bl_label = "Scan Tiles"
    bl_description = "Scan tiles folder and compute center positions"

    def execute(self, context):
        props = context.scene.uav_props
        folder = bpy.path.abspath(props.tiles_folder)
        if not folder or not os.path.isdir(folder):
            self.report({'ERROR'}, "Invalid tiles folder")
            return {'CANCELLED'}
        limit = props.compute_limit if props.compute_limit > 0 else None

        def progress(i, t):
            print(f"[Scan] {i}/{t}")

        try:
            tiles = compute_tile_centers(folder, limit=limit, progress_cb=progress)
            save_tiles_index(folder, tiles)

            # Update runtime state
            records = []
            for rel, center in tiles.items():
                records.append({
                    "rel": rel,
                    "path": os.path.join(folder, rel),
                    "center": Vector(center),
                    "loaded": False,
                    "objs": []
                })
            set_tile_records(records)

            # Update visuals if enabled
            if props.show_tile_visuals:
                update_tile_visuals()

            self.report({'INFO'}, f"Scanned {len(tiles)} tiles.")
            return {'FINISHED'}
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

class UAV_OT_LoadTilesIndex(Operator):
    bl_idname = "uav.load_tiles_index"
    bl_label = "Load Tiles Index"
    bl_description = "Load existing tiles_index.json without re-scanning"

    def execute(self, context):
        props = context.scene.uav_props
        folder = bpy.path.abspath(props.tiles_folder)
        if not folder or not os.path.isdir(folder):
            self.report({'ERROR'}, "Invalid tiles folder")
            return {'CANCELLED'}

        index = load_tiles_index(folder)
        if index is None:
            self.report({'ERROR'}, "No tiles_index.json found. Please scan tiles first.")
            return {'CANCELLED'}

        records = []
        for rel, center in index.items():
            records.append({
                "rel": rel,
                "path": os.path.join(folder, rel),
                "center": Vector(center),
                "loaded": False,
                "objs": []
            })
        set_tile_records(records)

        if props.show_tile_visuals:
            update_tile_visuals()

        self.report({'INFO'}, f"Loaded index with {len(records)} tiles")
        return {'FINISHED'}

class UAV_OT_ShowTileVisuals(Operator):
    bl_idname = "uav.show_tile_visuals"
    bl_label = "Show Tile Centers"
    bl_description = "Display tile center points in the viewport"

    def execute(self, context):
        records = get_tile_records()
        if not records:
            self.report({'WARNING'}, "No tile records loaded. Load tiles index first.")
            return {'CANCELLED'}

        update_tile_visuals()
        self.report({'INFO'}, f"Showing {len(records)} tile centers")
        return {'FINISHED'}

class UAV_OT_HideTileVisuals(Operator):
    bl_idname = "uav.hide_tile_visuals"
    bl_label = "Hide Tile Centers"
    bl_description = "Remove tile center visualizations"

    def execute(self, context):
        clear_tile_visuals()
        self.report({'INFO'}, "Tile visuals cleared")
        return {'FINISHED'}

class UAV_OT_GenWaypoints(Operator):
    bl_idname = "uav.gen_waypoints"
    bl_label = "Generate Waypoints"
    bl_description = "Generate waypoint empties in the scene"

    def execute(self, context):
        props = context.scene.uav_props
        # Remove old waypoint empties
        clear_waypoint_visuals()

        try:
            csvp = bpy.path.abspath(props.csv_waypoint_file) if props.waypoint_mode == 'csv' else None

            if props.waypoint_mode == 'manual':
                self.report({'WARNING'}, "Manual mode - add waypoints manually using 'Add Waypoint' button")
                return {'CANCELLED'}

            wps = generate_waypoints(props.waypoint_mode, props.n_waypoints, props.radius, props.height_m, csvp)
            col = get_or_create_uav_collection("UAV_Waypoints")

            for i, v in enumerate(wps):
                e = bpy.data.objects.new(f"uav_wp_{i:04d}", None)
                e.empty_display_type = 'ARROWS'
                e.empty_display_size = 1.0
                e.location = v
                e["uav_waypoint"] = True
                e["uav_waypoint_index"] = i
                col.objects.link(e)

            self.report({'INFO'}, f"Created {len(wps)} waypoint empties.")
            return {'FINISHED'}
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

class UAV_OT_AddWaypoint(Operator):
    bl_idname = "uav.add_waypoint"
    bl_label = "Add Waypoint"
    bl_description = "Add a waypoint at the 3D cursor position"

    def execute(self, context):
        cursor_loc = context.scene.cursor.location.copy()
        col = get_or_create_uav_collection("UAV_Waypoints")

        # Find next index
        existing = [o for o in bpy.context.scene.objects if o.get("uav_waypoint") == True]
        next_idx = len(existing)

        e = bpy.data.objects.new(f"uav_wp_{next_idx:04d}", None)
        e.empty_display_type = 'ARROWS'
        e.empty_display_size = 1.0
        e.location = cursor_loc
        e["uav_waypoint"] = True
        e["uav_waypoint_index"] = next_idx
        col.objects.link(e)

        self.report({'INFO'}, f"Added waypoint {next_idx} at {cursor_loc}")
        return {'FINISHED'}

class UAV_OT_UseSelectedAsWaypoints(Operator):
    bl_idname = "uav.use_selected_as_waypoints"
    bl_label = "Selection → Waypoints"
    bl_description = "Convert selected objects to waypoints"

    def execute(self, context):
        selected = [o for o in context.selected_objects]
        if not selected:
            self.report({'WARNING'}, "No objects selected")
            return {'CANCELLED'}

        col = get_or_create_uav_collection("UAV_Waypoints")
        count = 0

        for o in selected:
            if o.get("uav_waypoint") != True:
                o["uav_waypoint"] = True
                o["uav_waypoint_index"] = count
                # Ensure it's in the waypoints collection
                if o.name not in col.objects:
                    try:
                        col.objects.link(o)
                    except Exception:
                        pass
                count += 1

        self.report({'INFO'}, f"Marked {count} objects as waypoints")
        return {'FINISHED'}

class UAV_OT_ClearWaypoints(Operator):
    bl_idname = "uav.clear_waypoints"
    bl_label = "Clear Waypoints"
    bl_description = "Remove all waypoint empties from the scene"

    def execute(self, context):
        clear_waypoint_visuals()
        # Also remove the collection if empty
        col = bpy.data.collections.get("UAV_Waypoints")
        if col and len(col.objects) == 0:
            try:
                bpy.data.collections.remove(col)
            except Exception:
                pass
        self.report({'INFO'}, "Waypoints cleared")
        return {'FINISHED'}

class UAV_OT_PreviewCamera(Operator):
    bl_idname = "uav.preview_camera"
    bl_label = "Preview Camera Path"
    bl_description = "Move camera through waypoints to preview the path"

    _timer = None
    _waypoints = []
    _current_idx = 0

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._current_idx >= len(self._waypoints):
                self.cancel(context)
                self.report({'INFO'}, "Preview complete")
                return {'FINISHED'}

            cam = context.scene.camera
            if cam:
                props = context.scene.uav_props
                pos = self._waypoints[self._current_idx]
                cam.location = pos
                target = Vector((props.look_at_x, props.look_at_y, props.look_at_z))
                dir_vec = target - pos
                if dir_vec.length > 0:
                    cam.rotation_quaternion = dir_vec.to_track_quat('-Z', 'Y')

            self._current_idx += 1

        return {'PASS_THROUGH'}

    def execute(self, context):
        props = context.scene.uav_props

        if props.use_manual_waypoints or props.waypoint_mode == 'manual':
            self._waypoints = collect_manual_waypoints()
        else:
            csvp = bpy.path.abspath(props.csv_waypoint_file) if props.waypoint_mode == 'csv' else None
            self._waypoints = generate_waypoints(props.waypoint_mode, props.n_waypoints, props.radius, props.height_m, csvp)

        if not self._waypoints:
            self.report({'ERROR'}, "No waypoints to preview")
            return {'CANCELLED'}

        self._current_idx = 0
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        self.report({'INFO'}, f"Previewing {len(self._waypoints)} waypoints (ESC to stop)")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)

class UAV_OT_LoadSingleTile(Operator):
    bl_idname = "uav.load_single_tile"
    bl_label = "Load Selected Tile"
    bl_description = "Load a single tile for preview (select tile visual first)"

    def execute(self, context):
        # Check if a tile visual is selected
        selected = context.active_object
        if not selected or not selected.get("uav_tile_visual"):
            self.report({'WARNING'}, "Select a tile center visual first")
            return {'CANCELLED'}

        tile_rel = selected.get("tile_rel")
        if not tile_rel:
            self.report({'ERROR'}, "Invalid tile visual")
            return {'CANCELLED'}

        props = context.scene.uav_props
        folder = bpy.path.abspath(props.tiles_folder)
        tile_path = os.path.join(folder, tile_rel)

        if not os.path.exists(tile_path):
            self.report({'ERROR'}, f"Tile file not found: {tile_path}")
            return {'CANCELLED'}

        # Find and update the record
        records = get_tile_records()
        for rec in records:
            if rec['rel'] == tile_rel:
                if rec['loaded']:
                    self.report({'INFO'}, f"Tile already loaded: {tile_rel}")
                    return {'CANCELLED'}

                try:
                    names = import_glb(tile_path)
                    rec['loaded'] = True
                    rec['objs'] = names
                    update_tile_visuals()
                    self.report({'INFO'}, f"Loaded tile: {tile_rel}")
                    return {'FINISHED'}
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to load tile: {e}")
                    return {'CANCELLED'}

        self.report({'ERROR'}, "Tile not found in index")
        return {'CANCELLED'}

class UAV_OT_UnloadAllTiles(Operator):
    bl_idname = "uav.unload_all_tiles"
    bl_label = "Unload All Tiles"
    bl_description = "Unload all currently loaded tiles"

    def execute(self, context):
        records = get_tile_records()
        count = 0
        for rec in records:
            if rec['loaded']:
                remove_objects_by_names(rec['objs'])
                rec['objs'] = []
                rec['loaded'] = False
                count += 1

        purge_unused()
        update_tile_visuals()

        self.report({'INFO'}, f"Unloaded {count} tiles")
        return {'FINISHED'}

class UAV_OT_StartCapture(Operator):
    bl_idname = "uav.start_capture"
    bl_label = "Start Capture"
    bl_description = "Begin capturing images from waypoints"

    def execute(self, context):
        props = context.scene.uav_props
        tiles_folder = bpy.path.abspath(props.tiles_folder)
        out_dir = bpy.path.abspath(props.out_dir)

        if not props.skip_tile_loading:
            if not tiles_folder or not os.path.isdir(tiles_folder):
                self.report({'ERROR'}, "Invalid tiles folder (or enable 'Skip Tile Loading')")
                return {'CANCELLED'}

        if not out_dir:
            self.report({'ERROR'}, "Specify output folder")
            return {'CANCELLED'}

        try:
            look_at = Vector((props.look_at_x, props.look_at_y, props.look_at_z))

            run_capture(
                tiles_root=tiles_folder,
                out_dir=out_dir,
                waypoint_mode=props.waypoint_mode,
                n_waypoints=props.n_waypoints,
                radius=props.radius,
                height=props.height_m,
                load_radius=props.load_radius,
                unload_margin=props.unload_margin,
                max_loaded=props.max_loaded,
                decimate_ratio=props.decimate_ratio if props.decimate_ratio > 0 else 0.0,
                image_width=props.width,
                image_height=props.height,
                image_format=props.image_format,
                engine_choice=props.engine_choice,
                csv_waypoint=bpy.path.abspath(props.csv_waypoint_file) if props.waypoint_mode == 'csv' else None,
                start_index=0,
                tiles_limit_scan=props.compute_limit,
                headless=props.headless_mode,
                skip_tile_loading=props.skip_tile_loading,
                use_manual_waypoints=props.use_manual_waypoints or props.waypoint_mode == 'manual',
                look_at_center=look_at
            )
            self.report({'INFO'}, "Capture finished.")
            return {'FINISHED'}
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

# ---------------------------
# UI Panels
# ---------------------------
class UAV_PT_Panel(Panel):
    bl_label = "UAV Capture"
    bl_category = "UAV Capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        props = context.scene.uav_props

        # Folders
        box = layout.box()
        box.label(text="Folders", icon='FILE_FOLDER')
        box.prop(props, "tiles_folder")
        box.prop(props, "out_dir")

        # Mode selection
        box = layout.box()
        box.label(text="Capture Mode", icon='SETTINGS')
        box.prop(props, "skip_tile_loading")
        box.prop(props, "use_manual_waypoints")

class UAV_PT_Render(Panel):
    bl_label = "Render Settings"
    bl_category = "UAV Capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "UAV_PT_Panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.uav_props

        layout.prop(props, "engine_choice")
        row = layout.row(align=True)
        row.prop(props, "width")
        row.prop(props, "height")
        layout.prop(props, "image_format")

class UAV_PT_Waypoints(Panel):
    bl_label = "Waypoints"
    bl_category = "UAV Capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "UAV_PT_Panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.uav_props

        layout.prop(props, "waypoint_mode")

        if props.waypoint_mode not in ('csv', 'manual'):
            layout.prop(props, "n_waypoints")
            layout.prop(props, "radius")
            layout.prop(props, "height_m")
        elif props.waypoint_mode == 'csv':
            layout.prop(props, "csv_waypoint_file")

        # Look-at target
        box = layout.box()
        box.label(text="Camera Look-At Target")
        row = box.row(align=True)
        row.prop(props, "look_at_x")
        row.prop(props, "look_at_y")
        row.prop(props, "look_at_z")

        # Waypoint operations
        layout.separator()
        row = layout.row(align=True)
        row.operator("uav.gen_waypoints", icon='CURVE_PATH')
        row.operator("uav.clear_waypoints", icon='X')

        row = layout.row(align=True)
        row.operator("uav.add_waypoint", icon='ADD')
        row.operator("uav.use_selected_as_waypoints", icon='RESTRICT_SELECT_OFF')

        layout.operator("uav.preview_camera", icon='PLAY')

        # Show waypoint count
        wp_count = sum(1 for o in bpy.context.scene.objects if o.get("uav_waypoint") == True)
        layout.label(text=f"Waypoints in scene: {wp_count}")

class UAV_PT_Tiles(Panel):
    bl_label = "Tile Management"
    bl_category = "UAV Capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "UAV_PT_Panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.uav_props

        if props.skip_tile_loading:
            layout.label(text="Tile loading disabled", icon='INFO')
            return

        # Dynamic loading settings
        layout.prop(props, "load_radius")
        layout.prop(props, "unload_margin")
        layout.prop(props, "max_loaded")
        layout.prop(props, "decimate_ratio")
        layout.prop(props, "compute_limit")

        layout.separator()

        # Tile operations
        row = layout.row(align=True)
        row.operator("uav.scan_tiles", icon='FILE_REFRESH')
        row.operator("uav.load_tiles_index", icon='IMPORT')

        row = layout.row(align=True)
        row.operator("uav.show_tile_visuals", icon='HIDE_OFF')
        row.operator("uav.hide_tile_visuals", icon='HIDE_ON')

        row = layout.row(align=True)
        row.operator("uav.load_single_tile", icon='IMPORT')
        row.operator("uav.unload_all_tiles", icon='TRASH')

        # Status
        records = get_tile_records()
        loaded_count = sum(1 for r in records if r.get('loaded', False))
        layout.separator()
        box = layout.box()
        box.label(text="Tile Status", icon='INFO')
        box.label(text=f"Total tiles: {len(records)}")
        box.label(text=f"Loaded: {loaded_count}")
        box.label(text=f"Unloaded: {len(records) - loaded_count}")

class UAV_PT_Capture(Panel):
    bl_label = "Capture"
    bl_category = "UAV Capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "UAV_PT_Panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.uav_props

        layout.prop(props, "headless_mode")
        layout.separator()
        layout.operator("uav.start_capture", icon='RENDER_STILL', text="Start Capture")

# ---------------------------
# Registration
# ---------------------------
classes = (
    UAVProps,
    UAV_OT_ScanTiles,
    UAV_OT_LoadTilesIndex,
    UAV_OT_ShowTileVisuals,
    UAV_OT_HideTileVisuals,
    UAV_OT_GenWaypoints,
    UAV_OT_AddWaypoint,
    UAV_OT_UseSelectedAsWaypoints,
    UAV_OT_ClearWaypoints,
    UAV_OT_PreviewCamera,
    UAV_OT_LoadSingleTile,
    UAV_OT_UnloadAllTiles,
    UAV_OT_StartCapture,
    UAV_PT_Panel,
    UAV_PT_Render,
    UAV_PT_Waypoints,
    UAV_PT_Tiles,
    UAV_PT_Capture,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.uav_props = bpy.props.PointerProperty(type=UAVProps)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.uav_props
    clear_tile_records()

# ---------------------------
# Headless CLI support
# ---------------------------
def parse_and_run_headless():
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--") + 1
        argv = argv[idx:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--tiles_folder", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--waypoint_mode", type=str, default="circle")
    parser.add_argument("--n_waypoints", type=int, default=36)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--load_radius", type=float, default=50.0)
    parser.add_argument("--max_loaded", type=int, default=8)
    parser.add_argument("--decimate_ratio", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height_px", type=int, default=1080)
    parser.add_argument("--image_format", type=str, default="PNG")
    parser.add_argument("--engine_choice", type=str, default="AUTO")
    parser.add_argument("--csv_waypoint", type=str, default=None)
    parser.add_argument("--skip_tile_loading", action="store_true")
    parser.add_argument("--look_at", type=str, default="0,0,0", help="Look-at target as x,y,z")
    args = parser.parse_args(argv)

    if args.headless:
        if not args.skip_tile_loading and (not args.tiles_folder or not args.out_dir):
            print("headless requires --tiles_folder and --out_dir (or --skip_tile_loading)")
            return
        if not args.out_dir:
            print("headless requires --out_dir")
            return

        look_at = Vector([float(x.strip()) for x in args.look_at.split(',')])

        run_capture(
            tiles_root=args.tiles_folder or "",
            out_dir=args.out_dir,
            waypoint_mode=args.waypoint_mode,
            n_waypoints=args.n_waypoints,
            radius=args.radius,
            height=args.height,
            load_radius=args.load_radius,
            unload_margin=10.0,
            max_loaded=args.max_loaded,
            decimate_ratio=args.decimate_ratio,
            image_width=args.width,
            image_height=args.height_px,
            image_format=args.image_format,
            engine_choice=args.engine_choice,
            csv_waypoint=args.csv_waypoint,
            headless=True,
            skip_tile_loading=args.skip_tile_loading,
            look_at_center=look_at
        )

if __name__ == "__main__":
    register()
    try:
        parse_and_run_headless()
    except SystemExit:
        pass
