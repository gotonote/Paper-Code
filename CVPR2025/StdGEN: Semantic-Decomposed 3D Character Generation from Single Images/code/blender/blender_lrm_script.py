"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
import pdb
MAX_DEPTH = 5.0
import shutil
IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
    "vrm": bpy.ops.import_scene.vrm,
}

configs = {
    "custom2": {"camera_pose": "z-circular-elevated", 'elevation_range': [0,0], "rotate": 0.0},
    "custom_top": {"camera_pose": "z-circular-elevated", 'elevation_range': [90,90], "rotate": 0.0, "render_num": 1},
    "custom_bottom": {"camera_pose": "z-circular-elevated", 'elevation_range': [-90,-90], "rotate": 0.0, "render_num": 1},
    "custom_face": {"camera_pose": "z-circular-elevated", 'elevation_range': [0,0], "rotate": 0.0, "render_num": 8},
    "random": {"camera_pose": "random", 'elevation_range': [-90,90], "rotate": 0.0, "render_num": 20},
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


cached_cameras = []

def randomize_camera_with_cache(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
    idx: int = 0,
) -> bpy.types.Object:

    assert len(cached_cameras) >= idx

    if len(cached_cameras) == idx:
        x, y, z = _sample_spherical(
            radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
        )
        cached_cameras.append((x, y, z))
    else:
        x, y, z = cached_cameras[idx]

    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def set_camera(direction, camera_dist=2.0, camera_offset=0.0):
    camera = bpy.data.objects["Camera"]
    camera_pos = -camera_dist * direction
    if type(camera_offset) == float:
        camera_offset = Vector(np.array([0., 0., 0.]))
    camera_pos += camera_offset
    camera.location = camera_pos

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
        
    # delete all the collider collections
    for collider in bpy.data.collections:
        if collider.name != "Collection":
            bpy.data.collections.remove(collider, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }

def pan_camera(time, axis="Z", camera_dist=2.0, elevation=-0.1, camera_offset=0.0):
    angle = time * math.pi * 2 - math.pi / 2 # start from -90 degree
    direction = [-math.cos(angle), -math.sin(angle), -elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], -elevation, direction[1]]
    direction = Vector(direction).normalized()
    camera = set_camera(direction, camera_dist=camera_dist, camera_offset=camera_offset)
    return camera


def pan_camera_along(time, pose="alone-x-rotate", camera_dist=2.0, rotate=0.0):
    angle = time * math.pi * 2
    # direction_plane = [-math.cos(angle), -math.sin(angle), 0]
    x_new = math.cos(angle)
    y_new = math.cos(rotate) * math.sin(angle)
    z_new = math.sin(rotate) * math.sin(angle)
    direction = [-x_new, -y_new, -z_new]
    assert pose in ["alone-x-rotate"]
    direction = Vector(direction).normalized()
    camera = set_camera(direction, camera_dist=camera_dist)
    return camera

def pan_camera_by_angle(angle, axis="Z", camera_dist=2.0, elevation=-0.1 ):
    direction = [-math.cos(angle), -math.sin(angle), -elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], -elevation, direction[1]]
    direction = Vector(direction).normalized()
    camera = set_camera(direction, camera_dist=camera_dist)
    return camera

def z_circular_custom_track(time,
                            camera_dist,
                            azimuth_shift = [-9, 9],
                            init_elevation = 0.0,
                            elevation_shift = [-5, 5]):

    adjusted_azimuth = (-math.degrees(math.pi / 2) +
                        time * 360 +
                        np.random.uniform(low=azimuth_shift[0], high=azimuth_shift[1]))

    # Add random noise to the elevation
    adjusted_elevation = init_elevation + np.random.uniform(low=elevation_shift[0], high=elevation_shift[1])
    return math.radians(adjusted_azimuth), math.radians(adjusted_elevation), camera_dist


def place_camera(time, camera_pose_mode="random", camera_dist=2.0, rotate=0.0, elevation=0.0, camera_offset=0.0, idx=0):
    if camera_pose_mode == "z-circular-elevated":
        cam = pan_camera(time, axis="Z", camera_dist=camera_dist, elevation=elevation, camera_offset=camera_offset)
    elif camera_pose_mode == 'alone-x-rotate':
        cam = pan_camera_along(time, pose=camera_pose_mode, camera_dist=camera_dist, rotate=rotate)
    elif camera_pose_mode == 'z-circular-elevated-noise':
        angle, elevation, camera_dist = z_circular_custom_track(time, camera_dist=camera_dist, init_elevation=elevation)
        cam = pan_camera_by_angle(angle, axis="Z", camera_dist=camera_dist, elevation=elevation)
    elif camera_pose_mode == 'random':
        cam = randomize_camera_with_cache(radius_min=camera_dist, radius_max=camera_dist, maxz=114514., minz=-114514., idx=idx)
    else:
        raise ValueError(f"Unknown camera pose mode: {camera_pose_mode}")
    return cam


def setup_nodes(output_path, capturing_material_alpha: bool = False):
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]
    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    tree.links.new(color_socket, split_node.inputs[0])
    # Create separate file output nodes for every channel we care about.
    # The process calling this script must decide how to recombine these
    # channels, possibly into a single image.
    for i, channel in enumerate("rgba") if not capturing_material_alpha else [(0, "MatAlpha")]:
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"
        links.new(split_node.outputs[i], output_node.inputs[0])
    if capturing_material_alpha:
        # No need to re-write depth here.
        return

    depth_out = node_clamp(node_mul(input_sockets["Depth"], 1 / MAX_DEPTH))
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_node.format.file_format = 'OPEN_EXR'
    output_node.base_path = f"{output_path}_depth"
    links.new(depth_out, output_node.inputs[0])

    # Add normal map output
    normal_out = input_sockets["Normal"]

    # Scale normal by 0.5
    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(normal_out, scale_normal.inputs[1])

    # Bias normal by 0.5
    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    # Output the transformed normal map
    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.base_path = f"{output_path}_normal"
    normal_file_output.format.file_format = 'OPEN_EXR'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])


def setup_nodes_semantic(output_path, capturing_material_alpha: bool = False):
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]
        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz") or object_file.lower().endswith(".vrm"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures
    metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # normalize the scene
    normalize_scene()

    # cancel edge rim lighting in vrm files
    if object_file.endswith(".vrm"):
        for i in bpy.data.materials:
            i.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.rim_lighting_mix_factor = 0.0
            i.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.matcap_texture.index.source = None
            i.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.outline_width_factor = 0.0
            
    # rotate two arms to A-pose
    if object_file.endswith(".vrm"):
        armature = [ i for i in bpy.data.objects if 'Armature' in i.name ][0]
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        pbone1 = armature.pose.bones['J_Bip_L_UpperArm']
        pbone2 = armature.pose.bones['J_Bip_R_UpperArm']
        pbone1.rotation_mode = 'XYZ'
        pbone2.rotation_mode = 'XYZ'
        pbone1.rotation_euler.rotate_axis('X', math.radians(-45))
        pbone2.rotation_euler.rotate_axis('X', math.radians(-45))
        bpy.ops.object.mode_set(mode='OBJECT')

    def printInfo():
        print("====== Objects ======")
        for i in bpy.data.objects:
            print(i.name)
        print("====== Materials ======")
        for i in bpy.data.materials:
            print(i.name)

    def parse_material():
        hair_mats = []
        cloth_mats = []
        face_mats = []
        body_mats = []

        # main hair material
        if 'Hair' in bpy.data.objects:
            hair_mats = [i.name for i in bpy.data.objects['Hair'].data.materials if 'MToon Outline' not in i.name]
        else:
            flag = False
            for i in bpy.data.objects:
                if i.name[:4] == 'Hair' and bpy.data.objects[i.name].data:
                    hair_mats += [i.name for i in bpy.data.objects[i.name].data.materials if 'MToon Outline' not in i.name]
                    flag = True
            if not flag:
                if 'Hairs' in bpy.data.objects and bpy.data.objects['Hairs'].data:
                    hair_mats = [i.name for i in bpy.data.objects['Hairs'].data.materials if 'MToon Outline' not in i.name]
                else:
                    for i in bpy.data.materials:
                        if 'HAIR' in i.name and 'MToon Outline' not in i.name:
                            hair_mats.append(i.name)
                    if len(hair_mats) == 0:
                        printInfo()
                        with open('error.txt', 'a+') as f:
                            f.write(object_file + '\t' + 'Cannot find main hair material\t' + str([iii.name for iii in bpy.data.objects]) + '\n')
                        raise ValueError("Cannot find main hair material")
        
        # face material
        if 'Face' in bpy.data.objects:
            face_mats = [i.name for i in bpy.data.objects['Face'].data.materials if 'MToon Outline' not in i.name]
        else:
            for i in bpy.data.materials:
                if 'FACE' in i.name and 'MToon Outline' not in i.name:
                    face_mats.append(i.name)
                elif 'Face' in i.name and 'SKIN' in i.name and 'MToon Outline' not in i.name:
                    face_mats.append(i.name)
            if len(face_mats) == 0:
                printInfo()
                with open('error.txt', 'a+') as f:
                    f.write(object_file + '\t' + 'Cannot find face material\t' + str([iii.name for iii in bpy.data.objects]) + '\n')
                raise ValueError("Cannot find face material")
        
        # loop
        for i in bpy.data.materials:
            if 'MToon Outline' in i.name:
                continue
            elif 'CLOTH' in i.name:
                if 'Shoes' in i.name:
                    body_mats.append(i.name)
                elif 'Accessory' in i.name:
                    if 'CatEar' in i.name:
                        hair_mats.append(i.name)
                    else:
                        cloth_mats.append(i.name)
                elif any( name in i.name for name in ['Tops', 'Bottoms', 'Onepice'] ):
                    cloth_mats.append(i.name)
                else:
                    raise ValueError(f"Unknown cloth material: {i.name}")
            elif 'Body' in i.name and 'SKIN' in i.name:
                body_mats.append(i.name)
            elif i.name in hair_mats or i.name in face_mats:
                continue
            elif 'HairBack' in i.name and 'HAIR' in i.name:
                hair_mats.append(i.name)
            elif 'EYE' in i.name:
                face_mats.append(i.name)
            elif 'Face' in i.name and 'SKIN' in i.name:
                face_mats.append(i.name)
            else:
                print("hair_mats", hair_mats)
                print("cloth_mats", cloth_mats)
                print("face_mats", face_mats)
                print("body_mats", body_mats)
                with open('error.txt', 'a+') as f:
                    f.write(object_file + '\t' + 'Cannot find material\t' + i.name + '\n')
                raise ValueError(f"Unknown material: {i.name}")
            
        return hair_mats, cloth_mats, face_mats, body_mats
    
    hair_mats, cloth_mats, face_mats, body_mats = parse_material()

    # get bounding box of face
    def get_face_bbox():
        if 'Face' in bpy.data.objects:
            face = bpy.data.objects['Face']
            bbox_min, bbox_max = scene_bbox(face)
            return bbox_min, bbox_max
        else:
            bbox_min, bbox_max = scene_bbox()
            for i in bpy.data.objects:
                if i.data.materials and i.data.materials[0].name in face_mats:
                    face = i
                    cur_bbox_min, cur_bbox_max = scene_bbox(face)
                    bbox_min = np.minimum(bbox_min, cur_bbox_min)
                    bbox_max = np.maximum(bbox_max, cur_bbox_max)
            return bbox_min, bbox_max
    
    def assign_color(material_name, color):
        material = bpy.data.materials.get(material_name)
        if material:
            material.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (1, 1, 1, 1)
            image = material.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_texture.index.source
            if image:
                pixels = np.array(image.pixels[:])
                width, height = image.size
                num_channels = 4
                pixels = pixels.reshape((height, width, num_channels))
                srgb_pixels = np.clip(np.power(pixels, 1/2.2), 0.0, 1.0)
                print("Image converted to NumPy array")

                # Step 2: Edit the NumPy array
                srgb_pixels[..., 0] = color[0]
                srgb_pixels[..., 1] = color[1]
                srgb_pixels[..., 2] = color[2]
                edited_image_rgba = srgb_pixels

                # Step 3: Convert the edited NumPy array back to a Blender image
                edited_image_flat = edited_image_rgba.astype(np.float32)
                edited_image_flat = edited_image_flat.flatten()
                edited_image_name = "Edited_Texture"
                edited_blender_image = bpy.data.images.new(edited_image_name, width, height, alpha=True)
                edited_blender_image.pixels = edited_image_flat
                material.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_texture.index.source = edited_blender_image
                print(f"Edited image assigned to {material_name}")

            material.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.shade_color_factor = (1, 1, 1)
            image = material.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.shade_multiply_texture.index.source
            if image:
                pixels = np.array(image.pixels[:])
                width, height = image.size
                num_channels = 4
                pixels = pixels.reshape((height, width, num_channels))
                srgb_pixels = np.clip(np.power(pixels, 1/2.2), 0.0, 1.0)
                print("Image converted to NumPy array")

                # Step 2: Edit the NumPy array
                srgb_pixels[..., 0] = color[0]
                srgb_pixels[..., 1] = color[1]
                srgb_pixels[..., 2] = color[2]
                edited_image_rgba = srgb_pixels

                # Step 3: Convert the edited NumPy array back to a Blender image
                edited_image_flat = edited_image_rgba.astype(np.float32)
                edited_image_flat = edited_image_flat.flatten()
                edited_image_name = "Edited_Texture"
                edited_blender_image = bpy.data.images.new(edited_image_name, width, height, alpha=True)
                edited_blender_image.pixels = edited_image_flat
                material.vrm_addon_extension.mtoon1.extensions.vrmc_materials_mtoon.shade_multiply_texture.index.source = edited_blender_image
                print(f"Edited image assigned to {material_name}")
            material.vrm_addon_extension.mtoon1.extensions.khr_materials_emissive_strength.emissive_strength = 0.0
            
    def assign_transparency(material_name, alpha):
        material = bpy.data.materials.get(material_name)
        if material:
            material.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (1, 1, 1, alpha)

    # render the images
    use_workbench = bpy.context.scene.render.engine == "BLENDER_WORKBENCH"

    face_bbox_min, face_bbox_max = get_face_bbox()
    face_bbox_center = (face_bbox_min + face_bbox_max) / 2
    face_bbox_size = face_bbox_max - face_bbox_min
    print("face_bbox_center", face_bbox_center)
    print("face_bbox_size", face_bbox_size)

    config_names = ["custom2", "custom_top", "custom_bottom", "custom_face", "random"]

    # normal rendering
    for l in range(3):  # 3 levels: all; no hair; no hair and no cloth
        if l == 0:
            pass
        elif l == 1:
            for i in hair_mats:
                bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (0, 0, 0, 0)
        elif l == 2:
            for i in cloth_mats:
                bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (0, 0, 0, 0)

        for j in range(5): # 5 track
            config = configs[config_names[j]]
            if "render_num" in config:
                new_num_renders = config["render_num"]
            else:
                new_num_renders = num_renders

            for i in range(new_num_renders):
                camera_dist = 1.4
                if config_names[j] == "custom_face":
                    camera_dist = 0.6
                    if i not in [0, 1, 2, 6, 7]:
                        continue
                t = i / num_renders
                elevation_range = config["elevation_range"]
                init_elevation = elevation_range[0]
                # set camera
                camera = place_camera(
                t,
                camera_pose_mode=config["camera_pose"],
                camera_dist=camera_dist,
                rotate=config["rotate"],
                elevation=init_elevation,
                camera_offset=face_bbox_center if config_names[j] == "custom_face" else 0.0,
                idx=i
                )
                
                # set camera to ortho
                bpy.data.objects["Camera"].data.type = 'ORTHO'
                bpy.data.objects["Camera"].data.ortho_scale = 1.2 if config_names[j] != "custom_face" else np.max(face_bbox_size) * 1.2
                
                # render the image
                render_path = os.path.join(output_dir, f"{(i + j * 100 + l * 1000):05}.png")
                scene.render.filepath = render_path
                setup_nodes(render_path)
                bpy.ops.render.render(write_still=True)

                # save camera RT matrix
                rt_matrix = get_3x4_RT_matrix_from_blender(camera)
                rt_matrix_path = os.path.join(output_dir, f"{(i + j * 100 + l * 1000):05}.npy")
                np.save(rt_matrix_path, rt_matrix)

                for channel_name in ["r", "g", "b", "a", "depth", "normal"]:
                    sub_dir = f"{render_path}_{channel_name}"
                    if channel_name in ['r', 'g', 'b']:
                        # remove path
                        shutil.rmtree(sub_dir)
                        continue

                    image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])
                    name, ext = os.path.splitext(render_path)
                    if  channel_name == "a":
                        os.rename(image_path, f"{name}_{channel_name}.png")
                    elif channel_name == 'depth':
                        os.rename(image_path, f"{name}_{channel_name}.exr")
                    elif channel_name == "normal":
                        os.rename(image_path, f"{name}_{channel_name}.exr")
                    else:
                        os.remove(image_path)

                    os.removedirs(sub_dir)

    # reset
    for i in hair_mats:
        bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (1, 1, 1, 1)
    for i in cloth_mats:
        bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (1, 1, 1, 1)

    # switch to semantic rendering
    for i in hair_mats:
        assign_color(i, [1.0, 0.0, 0.0])
    for i in cloth_mats:
        assign_color(i, [0.0, 0.0, 1.0])
    for i in face_mats:
        assign_color(i, [0.0, 1.0, 1.0])
        if any( ii in i for ii in ['Eyeline', 'Eyelash', 'Brow', 'Highlight'] ):
            assign_transparency(i, 0.0)
    for i in body_mats:
        assign_color(i, [0.0, 1.0, 0.0])

    for l in range(3):  # 3 levels: all; no hair; no hair and no cloth
        if l == 0:
            pass
        elif l == 1:
            for i in hair_mats:
                bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (0, 0, 0, 0)
        elif l == 2:
            for i in cloth_mats:
                bpy.data.materials[i].vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor = (0, 0, 0, 0)
        for j in range(5): # 5 track
            config = configs[config_names[j]]
            if "render_num" in config:
                new_num_renders = config["render_num"]
            else:
                new_num_renders = num_renders

            for i in range(new_num_renders):
                camera_dist = 1.4
                if config_names[j] == "custom_face":
                    camera_dist = 0.6
                    if i not in [0, 1, 2, 6, 7]:
                        continue
                t = i / num_renders
                elevation_range = config["elevation_range"]
                init_elevation = elevation_range[0]
                # set camera
                camera = place_camera(
                t,
                camera_pose_mode=config["camera_pose"],
                camera_dist=camera_dist,
                rotate=config["rotate"],
                elevation=init_elevation,
                camera_offset=face_bbox_center if config_names[j] == "custom_face" else 0.0,
                idx=i
                )
                
                # set camera to ortho
                bpy.data.objects["Camera"].data.type = 'ORTHO'
                bpy.data.objects["Camera"].data.ortho_scale = 1.2 if config_names[j] != "custom_face" else np.max(face_bbox_size) * 1.2
                
                # render the image
                render_path = os.path.join(output_dir, f"{(i + j * 100 + l * 1000):05}_semantic.png")
                scene.render.filepath = render_path
                setup_nodes_semantic(render_path)
                bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=8,
        help="Number of renders to save of the object.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.resolution_x = 1024
    render.resolution_y = 1024
    render.resolution_percentage = 100
    
    # Set EEVEE settings
    scene.eevee.taa_render_samples = 64
    scene.eevee.use_taa_reprojection = True

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 9
    scene.cycles.glossy_bounces = 9
    scene.cycles.transparent_max_bounces = 9
    scene.cycles.transmission_bounces = 9
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

    bpy.context.view_layer.use_pass_normal = True
    render.image_settings.color_depth = "16"
    bpy.context.scene.use_nodes = True

    # Render the images
    render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
    )
