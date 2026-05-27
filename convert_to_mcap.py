import argparse
import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

# Import Foxglove Python SDK and its native message types
import foxglove
from foxglove import Channel, Schema
from foxglove.messages import (
    CameraCalibration,
    CompressedImage,
    CubePrimitive,
    FrameTransform,
    Grid,
    ImageAnnotations,
    LinePrimitive,
    LinePrimitiveLineType,
    LocationFix,
    ModelPrimitive,
    PackedElementField,
    PackedElementFieldNumericType,
    Point2,
    Point3,
    PointCloud,
    Pose,
    PoseInFrame,
    PointsAnnotation,
    PointsAnnotationType,
    Quaternion as foxglove_Quaternion,
    SceneEntity,
    SceneUpdate,
    TextAnnotation,
    Color,
    Vector2,
    Vector3,
    KeyValuePair,
    Timestamp,
)

with open(Path(__file__).parent / "turbomap.json") as f:
    TURBOMAP_DATA = np.array(json.load(f))


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md#imu
IMU_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "linear_accel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "q": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "rotation_rate": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
    },
}

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md#pose
ODOM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "accel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "orientation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "pos": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "rotation_rate": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "vel": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
    },
}

# Standard representation of a DiagnosticArray message structure
DIAGNOSTIC_ARRAY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "header": {
            "type": "object",
            "properties": {
                "stamp": {
                    "type": "object",
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"}
                    },
                    "required": ["sec", "nsec"]
                }
            },
            "required": ["stamp"]
        },
        "status": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer"},
                    "name": {"type": "string"},
                    "message": {"type": "string"},
                    "hardware_id": {"type": "string"},
                    "values": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"}
                            },
                            "required": ["key", "value"]
                        }
                    }
                },
                "required": ["level", "name", "message", "hardware_id", "values"]
            }
        }
    },
    "required": ["header", "status"]
}


def load_bitmap(dataroot: str, map_name: str, layer_name: str) -> np.ndarray:
    """render bitmap map layers. Currently these are:
    - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
    - basemap: The HD lidar basemap used for localization and as general context.

    :param dataroot: Path of the nuScenes dataset.
    :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
        'boston-seaport'.
    :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
    """
    # Load bitmap.
    if layer_name == "basemap":
        map_path = os.path.join(dataroot, "maps", "basemap", map_name + ".png")
    elif layer_name == "semantic_prior":
        map_hashes = {
            "singapore-onenorth": "53992ee3023e5494b90c316c183be829",
            "singapore-hollandvillage": "37819e65e09e5547b8a3ceaefba56bb2",
            "singapore-queenstown": "93406b464a165eaba6d9de76ca09f5da",
            "boston-seaport": "36092f0b03a857c6a3403e25b4b7aab3",
        }
        map_hash = map_hashes[map_name]
        map_path = os.path.join(dataroot, "maps", map_hash + ".png")
    else:
        raise Exception("Error: Invalid bitmap layer: %s" % layer_name)

    # Convert to numpy.
    if os.path.exists(map_path):
        image = np.array(Image.open(map_path).convert("L"))
    else:
        raise Exception("Error: Cannot find %s %s! Please make sure that the map is correctly installed." % (layer_name, map_path))

    # Invert semantic prior colors.
    if layer_name == "semantic_prior":
        image = image.max() - image

    return image


EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(math.sin(lat) * math.cos(angular_distance) + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing))
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(location: str, pose: Dict[str, float]):
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.

    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).
    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), "Error: Reference coordinate not found for location %s!" % location
    ref_lat, ref_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan2(x, y)
    yaw = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(ref_lat, ref_lon, bearing, yaw)
    return lat, lon


def get_translation(data):
    return Vector3(x=data["translation"][0], y=data["translation"][1], z=data["translation"][2])


def get_rotation(data):
    return foxglove_Quaternion(x=data["rotation"][1], y=data["rotation"][2], z=data["rotation"][3], w=data["rotation"][0])


def get_timestamp(timestamp_us) -> Timestamp:
    sec, msecs = divmod(timestamp_us, 1_000_000)
    nsec = msecs * 1000
    return Timestamp(sec=sec, nsec=nsec)


def get_timestamp_from_ns(timestamp_ns) -> Timestamp:
    sec, nsec = divmod(timestamp_ns, 1_000_000_000)
    return Timestamp(sec=sec, nsec=nsec)


# See:
# https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
def turbomap(x):
    np.clip(x, 0, 1, out=x)
    x *= 255
    a = x.astype(np.uint8)
    x -= a  # compute "f" in place
    b = np.minimum(254, a)
    b += 1
    color_a = TURBOMAP_DATA[a]
    color_b = TURBOMAP_DATA[b]
    color_b -= color_a
    color_b *= x[:, np.newaxis]
    return np.add(color_a, color_b, out=color_b)


def get_categories(nusc, first_sample):
    categories = set()
    sample_lidar = first_sample
    while sample_lidar is not None:
        sample = nusc.get("sample", sample_lidar["sample_token"])
        for annotation_id in sample["anns"]:
            ann = nusc.get("sample_annotation", annotation_id)
            categories.add(ann["category_name"])
        sample_lidar = nusc.get("sample_data", sample_lidar["next"]) if sample_lidar.get("next") != "" else None
    return categories


PCD_TO_PACKED_ELEMENT_TYPE_MAP = {
    ("I", 1): PackedElementFieldNumericType.Int8,
    ("U", 1): PackedElementFieldNumericType.Uint8,
    ("I", 2): PackedElementFieldNumericType.Int16,
    ("U", 2): PackedElementFieldNumericType.Uint16,
    ("I", 4): PackedElementFieldNumericType.Int32,
    ("U", 4): PackedElementFieldNumericType.Uint32,
    ("F", 4): PackedElementFieldNumericType.Float32,
    ("F", 8): PackedElementFieldNumericType.Float64,
}


def get_radar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]
    pc = pypcd.PointCloud.from_path(pc_filename)

    fields = []
    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        fields.append(
            PackedElementField(
                name=name,
                offset=offset,
                type=PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)]
            )
        )
        offset += size

    return PointCloud(
        timestamp=get_timestamp(sample_data["timestamp"]),
        frame_id=frame_id,
        point_stride=offset,
        fields=fields,
        data=pc.pc_data.tobytes(),
    )


def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data["filename"]
    with open(jpg_filename, "rb") as jpg_file:
        data = jpg_file.read()
    return CompressedImage(
        timestamp=get_timestamp(sample_data["timestamp"]),
        frame_id=frame_id,
        format="jpeg",
        data=data,
    )


def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    K = [calib["camera_intrinsic"][r][c] for r in range(3) for c in range(3)]
    R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    P = [K[0], K[1], K[2], 0.0, K[3], K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0]

    return CameraCalibration(
        timestamp=get_timestamp(sample_data["timestamp"]),
        frame_id=frame_id,
        height=sample_data["height"],
        width=sample_data["width"],
        K=K,
        R=R,
        P=P,
    )


def get_lidar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]

    with open(pc_filename, "rb") as pc_file:
        fields = [
            PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
            PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
            PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
            PackedElementField(name="intensity", offset=12, type=PackedElementFieldNumericType.Float32),
            PackedElementField(name="ring", offset=16, type=PackedElementFieldNumericType.Float32),
        ]
        return PointCloud(
            timestamp=get_timestamp(sample_data["timestamp"]),
            frame_id=frame_id,
            point_stride=len(fields) * 4,  # 20 bytes
            fields=fields,
            data=pc_file.read(),
        )


def get_lidar_image_annotations(nusc, sample_lidar, sample_data, frame_id):
    # lidar image markers in camera frame
    points, coloring, _ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=sample_lidar["token"],
        camera_token=sample_data["token"],
        render_intensity=True,
    )
    points = points.transpose()

    outline_colors = []
    for c in turbomap(coloring):
        outline_colors.append(Color(r=c[0], g=c[1], b=c[2], a=1.0))

    pts_list = [Point2(x=p[0], y=p[1]) for p in points]

    return ImageAnnotations(
        points=[
            PointsAnnotation(
                timestamp=get_timestamp(sample_data["timestamp"]),
                type=PointsAnnotationType.Points,
                thickness=2.0,
                points=pts_list,
                outline_colors=outline_colors,
            )
        ]
    )


def write_boxes_image_annotations(nusc, anns, sample_data, frame_id, topic_ns, stamp_ns):
    timestamp = get_timestamp_from_ns(stamp_ns)

    points_anns = []
    texts_anns = []

    # annotation boxes
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data["token"])
    for box in boxes:
        collector = Collector()
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(collector, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        box_pts = [Point2(x=p[0], y=p[1]) for p in collector.points]
        box_colors = [Color(r=col[0], g=col[1], b=col[2], a=1.0) for col in collector.colors]

        points_ann = PointsAnnotation(
            timestamp=timestamp,
            type=PointsAnnotationType.LineList,
            thickness=2.0,
            points=box_pts,
            outline_colors=box_colors,
            metadata=[KeyValuePair(key="category", value=box.name)]
        )
        points_anns.append(points_ann)

        min_x = min(pt[0] for pt in collector.points) if collector.points else 0.0
        min_y = min(pt[1] for pt in collector.points) if collector.points else 0.0

        texts_anns.append(
            TextAnnotation(
                timestamp=timestamp,
                font_size=24.0,
                position=Point2(x=min_x, y=min_y),
                text=box.name,
                text_color=Color(r=c[0], g=c[1], b=c[2], a=1.0),
                background_color=Color(r=1.0, g=1.0, b=1.0, a=0.0)
            )
        )

    msg = ImageAnnotations(
        points=points_anns,
        texts=texts_anns
    )

    foxglove.log(topic_ns + "/annotations", msg, log_time=stamp_ns)


def write_drivable_area(nusc_map, ego_pose, stamp_ns):
    translation = ego_pose["translation"]
    rotation = Quaternion(ego_pose["rotation"])
    yaw_radians = quaternion_yaw(rotation)
    yaw_degrees = yaw_radians / np.pi * 180
    patch_box = (translation[0], translation[1], 32, 32)
    canvas_size = (patch_box[2] * 10, patch_box[3] * 10)

    drivable_area = nusc_map.get_map_mask(patch_box, yaw_degrees, ["drivable_area"], canvas_size)[0]

    pos_x = translation[0] - (16 * math.cos(yaw_radians)) + (16 * math.sin(yaw_radians))
    pos_y = translation[1] - (16 * math.sin(yaw_radians)) - (16 * math.cos(yaw_radians))

    q = Quaternion(axis=(0, 0, 1), radians=yaw_radians)

    msg = Grid(
        timestamp=get_timestamp_from_ns(stamp_ns),
        frame_id="map",
        cell_size=Vector2(x=0.1, y=0.1),
        column_count=drivable_area.shape[1],
        row_stride=drivable_area.shape[1],
        cell_stride=1,
        fields=[
            PackedElementField(name="drivable_area", offset=0, type=PackedElementFieldNumericType.Uint8)
        ],
        pose=Pose(
            position=Vector3(x=pos_x, y=pos_y, z=0.01),
            orientation=foxglove_Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        ),
        data=drivable_area.astype(np.uint8).tobytes()
    )

    foxglove.log("/drivable_area", msg, log_time=stamp_ns)


def get_imu_msg(imu_data):
    timestamp_ns = int(imu_data["utime"]) * 1000

    msg = {
        "linear_accel": {
            "x": imu_data["linear_accel"][0],
            "y": imu_data["linear_accel"][1],
            "z": imu_data["linear_accel"][2],
        },
        "q": {
            "w": imu_data["q"][0],
            "x": imu_data["q"][1],
            "y": imu_data["q"][2],
            "z": imu_data["q"][3],
        },
        "rotation_rate": {
            "x": imu_data["rotation_rate"][0],
            "y": imu_data["rotation_rate"][1],
            "z": imu_data["rotation_rate"][2],
        },
    }

    return (timestamp_ns, "/imu", msg)


def get_odom_msg(pose_data):
    timestamp_ns = int(pose_data["utime"]) * 1000

    msg = {
        "accel": {
            "x": pose_data["accel"][0],
            "y": pose_data["accel"][1],
            "z": pose_data["accel"][2],
        },
        "orientation": {
            "w": pose_data["orientation"][0],
            "x": pose_data["orientation"][1],
            "y": pose_data["orientation"][2],
            "z": pose_data["orientation"][3],
        },
        "pos": {
            "x": pose_data["pos"][0],
            "y": pose_data["pos"][1],
            "z": pose_data["pos"][2],
        },
        "rotation_rate": {
            "x": pose_data["rotation_rate"][0],
            "y": pose_data["rotation_rate"][1],
            "z": pose_data["rotation_rate"][2],
        },
        "vel": {
            "x": pose_data["vel"][0],
            "y": pose_data["vel"][1],
            "z": pose_data["vel"][2],
        },
    }

    return (timestamp_ns, "/odom", msg)


def get_basic_can_msg(name, diag_data):
    values = []
    for (key, value) in diag_data.items():
        if key != "utime":
            values.append({"key": key, "value": str(round(value, 4))})

    sec, msecs = divmod(diag_data["utime"], 1_000_000)
    nsec = msecs * 1000

    msg = {
        "header": {
            "stamp": {
                "sec": int(sec),
                "nsec": int(nsec)
            }
        },
        "status": [
            {
                "level": 0,  # 0 matches OK
                "name": name,
                "message": "OK",
                "hardware_id": "",
                "values": values
            }
        ]
    }

    stamp_ns = int(diag_data["utime"]) * 1000
    return (stamp_ns, "/diagnostics", msg)


def get_ego_tf(ego_pose):
    return FrameTransform(
        parent_frame_id="map",
        child_frame_id="base_link",
        timestamp=get_timestamp(ego_pose["timestamp"]),
        translation=get_translation(ego_pose),
        rotation=get_rotation(ego_pose),
    )


def get_sensor_tf(nusc, sensor_id, sample_data):
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    return FrameTransform(
        parent_frame_id="base_link",
        child_frame_id=sensor_id,
        timestamp=get_timestamp(sample_data["timestamp"]),
        translation=get_translation(calibrated_sensor),
        rotation=get_rotation(calibrated_sensor),
    )


def scene_bounding_box(nusc, scene, nusc_map, padding=75.0):
    box = [np.inf, np.inf, -np.inf, -np.inf]
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        x, y = ego_pose["translation"][:2]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None
    box[0] = max(box[0] - padding, 0.0)
    box[1] = max(box[1] - padding, 0.0)
    box[2] = min(box[2] + padding, nusc_map.canvas_edge[0]) - box[0]
    box[3] = min(box[3] + padding, nusc_map.canvas_edge[1]) - box[1]
    return box


def get_scene_map(nusc, scene, nusc_map, image, stamp_ns):
    x, y, w, h = scene_bounding_box(nusc, scene, nusc_map)
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    img = np.flipud(image)[img_y : img_y + img_h, img_x : img_x + img_w]

    # img values are 0-255
    # convert to a color scale, 0=white and 255=black, in packed RGBA format: 0xFFFFFF00 to 0x00000000
    img = (255 - img) * 0x01010100
    # set alpha to 0xFF for all cells except those that are completely black
    img[img != 0x00000000] |= 0x000000FF

    return Grid(
        timestamp=get_timestamp_from_ns(stamp_ns),
        frame_id="map",
        cell_size=Vector2(x=0.1, y=0.1),
        column_count=img_w,
        row_stride=img_w * 4,
        cell_stride=4,
        fields=[
            PackedElementField(name="alpha", offset=0, type=PackedElementFieldNumericType.Uint8),
            PackedElementField(name="blue", offset=1, type=PackedElementFieldNumericType.Uint8),
            PackedElementField(name="green", offset=2, type=PackedElementFieldNumericType.Uint8),
            PackedElementField(name="red", offset=3, type=PackedElementFieldNumericType.Uint8),
        ],
        pose=Pose(
            position=Vector3(x=x, y=y, z=0.0),
            orientation=foxglove_Quaternion(x=0, y=0, z=0, w=1.0),
        ),
        data=img.astype("<u4").tobytes(),
    )


def rectContains(rect, point):
    a, b, c, d = rect
    x, y = point[:2]
    return a <= x < a + c and b <= y < b + d


def get_centerline_markers(nusc, scene, nusc_map, stamp_ns):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained_pose_lists = []
    for pose_list in pose_lists:
        new_pose_list = []
        for pose in pose_list:
            if rectContains(bbox, pose):
                new_pose_list.append(pose)
        if len(new_pose_list) > 1:
            contained_pose_lists.append(new_pose_list)

    timestamp = get_timestamp_from_ns(stamp_ns)

    entities = []
    for i, pose_list in enumerate(contained_pose_lists):
        points = [Point3(x=pose[0], y=pose[1], z=0.0) for pose in pose_list]
        line = LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            thickness=0.1,
            color=Color(r=51.0 / 255.0, g=160.0 / 255.0, b=44.0 / 255.0, a=1.0),
            points=points,
            pose=Pose(
                position=Vector3(x=0, y=0, z=0),
                orientation=foxglove_Quaternion(x=0, y=0, z=0, w=1.0),
            )
        )
        entity = SceneEntity(
            frame_id="map",
            timestamp=timestamp,
            id=f"{i}",
            frame_locked=True,
            lines=[line]
        )
        entities.append(entity)

    return SceneUpdate(entities=entities)


def find_closest_lidar(nusc, lidar_start_token, stamp_nsec):
    candidates = []

    next_lidar_token = nusc.get("sample_data", lidar_start_token)["next"]
    while next_lidar_token != "":
        lidar_data = nusc.get("sample_data", next_lidar_token)
        if lidar_data["is_key_frame"]:
            break

        dist_abs = abs(stamp_nsec - int(lidar_data["timestamp"]) * 1000)
        candidates.append((dist_abs, lidar_data))
        next_lidar_token = lidar_data["next"]

    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda x: x[0])[1]


def get_car_scene_update(stamp_ns) -> SceneUpdate:
    timestamp = get_timestamp_from_ns(stamp_ns)

    model = ModelPrimitive(
        pose=Pose(
            position=Vector3(x=1.0, y=0.0, z=0.0),
            orientation=foxglove_Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        ),
        scale=Vector3(x=1.0, y=1.0, z=1.0),
        url="https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    )

    entity = SceneEntity(
        frame_id="base_link",
        timestamp=timestamp,
        id="car",
        frame_locked=True,
        models=[model]
    )

    return SceneUpdate(entities=[entity])


class Collector:
    """
    Emulates the Matplotlib Axes class to collect line data.
    """

    def __init__(self):
        self.points = []
        self.colors = []

    def plot(self, xx, yy, color, linewidth):
        x1, x2 = xx
        y1, y2 = yy
        self.points.append((x1, y1))
        self.points.append((x2, y2))
        self.colors.append(color)


def get_num_sample_data(nusc: NuScenes, scene):
    num_sample_data = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sample_data = nusc.get("sample_data", sample_token)
        while sample_data is not None:
            num_sample_data += 1
            sample_data = nusc.get("sample_data", sample_data["next"]) if sample_data["next"] != "" else None
    return num_sample_data


def write_scene_to_mcap(nusc: NuScenes, nusc_can: NuScenesCanBus, scene, filepath):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    print(f'Loading map "{location}"')
    data_path = Path(nusc.dataroot)
    nusc_map = NuScenesMap(dataroot=data_path, map_name=location)
    print(f'Loading bitmap "{nusc_map.map_name}"')
    image = load_bitmap(nusc_map.dataroot, nusc_map.map_name, "basemap")
    print(f"Loaded {image.shape} bitmap")
    print(f"vehicle is {log['vehicle']}")

    cur_sample = nusc.get("sample", scene["first_sample_token"])
    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene_name} Sample Data", leave=False)

    can_parsers = [
        [nusc_can.get_messages(scene_name, "ms_imu"), 0, get_imu_msg],
        [nusc_can.get_messages(scene_name, "pose"), 0, get_odom_msg],
        [
            nusc_can.get_messages(scene_name, "steeranglefeedback"),
            0,
            lambda x: get_basic_can_msg("Steering Angle", x),
        ],
        [
            nusc_can.get_messages(scene_name, "vehicle_monitor"),
            0,
            lambda x: get_basic_can_msg("Vehicle Monitor", x),
        ],
        [
            nusc_can.get_messages(scene_name, "zoesensors"),
            0,
            lambda x: get_basic_can_msg("Zoe Sensors", x),
        ],
        [
            nusc_can.get_messages(scene_name, "zoe_veh_info"),
            0,
            lambda x: get_basic_can_msg("Zoe Vehicle Info", x),
        ],
    ]

    filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {filepath}")
    with foxglove.open_mcap(str(filepath), allow_overwrite=True) as writer:
        # Add MCAP Metadata
        writer.write_metadata(
            "scene-info",
            {
                "description": scene["description"],
                "name": scene["name"],
                "location": location,
                "vehicle": log["vehicle"],
                "date_captured": log["date_captured"],
            },
        )

        # Initialize named Schemas & JSON Channels for IMU and Odom dictionaries
        imu_schema = Schema(
            name="IMU",
            encoding="jsonschema",
            data=json.dumps(IMU_JSON_SCHEMA).encode("utf-8")
        )
        odom_schema = Schema(
            name="Pose",
            encoding="jsonschema",
            data=json.dumps(ODOM_JSON_SCHEMA).encode("utf-8")
        )
        diagnostic_schema = Schema(
            name="diagnostic_msgs/DiagnosticArray",
            encoding="jsonschema",
            data=json.dumps(DIAGNOSTIC_ARRAY_JSON_SCHEMA).encode("utf-8")
        )
        imu_channel = Channel("/imu", message_encoding="json", schema=imu_schema)
        odom_channel = Channel("/odom", message_encoding="json", schema=odom_schema)
        diagnostic_channel = Channel("/diagnostics", message_encoding="json", schema=diagnostic_schema)

        stamp_ns = int(
            nusc.get(
                "ego_pose",
                nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])["ego_pose_token"],
            )["timestamp"]
        ) * 1000

        map_msg = get_scene_map(nusc, scene, nusc_map, image, stamp_ns)
        centerlines_msg = get_centerline_markers(nusc, scene, nusc_map, stamp_ns)
        
        # Log baseline map and lane markers
        foxglove.log("/map", map_msg, log_time=stamp_ns)
        foxglove.log("/semantic_map", centerlines_msg, log_time=stamp_ns)

        while cur_sample is not None:
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            stamp_ns = int(ego_pose["timestamp"]) * 1000

            # write CAN messages to /pose, /odom, and /diagnostics
            can_msg_events = []
            for i in range(len(can_parsers)):
                (can_msgs, index, msg_func) = can_parsers[i]
                while index < len(can_msgs) and (int(can_msgs[index]["utime"]) * 1000) < stamp_ns:
                    can_msg_events.append(msg_func(can_msgs[index]))
                    index += 1
                    can_parsers[i][1] = index
            can_msg_events.sort(key=lambda x: x[0])
            for (msg_stamp_ns, topic, msg) in can_msg_events:
                if topic == "/imu":
                    imu_channel.log(msg, log_time=msg_stamp_ns)
                elif topic == "/odom":
                    odom_channel.log(msg, log_time=msg_stamp_ns)
                else:
                    diagnostic_channel.log(msg, log_time=msg_stamp_ns)

            # publish /tf
            foxglove.log("/tf", get_ego_tf(ego_pose), log_time=stamp_ns)

            # /driveable_area occupancy grid
            write_drivable_area(nusc_map, ego_pose, stamp_ns)

            # iterate sensors
            for (sensor_id, sample_token) in cur_sample["data"].items():
                pbar.update(1)
                sample_data = nusc.get("sample_data", sample_token)
                topic = ("/" + sensor_id).lower()

                # create sensor transform
                foxglove.log("/tf", get_sensor_tf(nusc, sensor_id.lower(), sample_data), log_time=stamp_ns)

                # write the sensor data
                if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id.lower())
                    foxglove.log(topic, msg, log_time=stamp_ns)
                elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id.lower())
                    foxglove.log(topic, msg, log_time=stamp_ns)
                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id.lower())
                    foxglove.log(topic + "/image_rect_compressed", msg, log_time=stamp_ns)
                    msg = get_camera_info(nusc, sample_data, sensor_id.lower())
                    foxglove.log(topic + "/camera_info", msg, log_time=stamp_ns)

                if sample_data["sensor_modality"] == "camera":
                    msg = get_lidar_image_annotations(nusc, sample_lidar, sample_data, sensor_id.lower())
                    foxglove.log(topic + "/lidar", msg, log_time=stamp_ns)
                    write_boxes_image_annotations(
                        nusc,
                        cur_sample["anns"],
                        sample_data,
                        sensor_id.lower(),
                        topic,
                        stamp_ns,
                    )

            # publish /pose
            pose_in_frame = PoseInFrame(
                timestamp=get_timestamp_from_ns(stamp_ns),
                frame_id="base_link",
                pose=Pose(
                    position=Vector3(x=0.0, y=0.0, z=0.0),
                    orientation=foxglove_Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )
            )
            foxglove.log("/pose", pose_in_frame, log_time=stamp_ns)

            # publish /gps
            lat, lon = derive_latlon(location, ego_pose)
            gps = LocationFix(
                latitude=lat,
                longitude=lon,
                altitude=ego_pose["translation"][2],
            )
            foxglove.log("/gps", gps, log_time=stamp_ns)

            # publish /markers/annotations
            entities = []
            for annotation_id in cur_sample["anns"]:
                ann = nusc.get("sample_annotation", annotation_id)
                marker_id = ann["instance_token"][:4]
                c = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

                entity = SceneEntity(
                    frame_id="map",
                    timestamp=get_timestamp_from_ns(stamp_ns),
                    id=marker_id,
                    frame_locked=True,
                    metadata=[KeyValuePair(key="category", value=ann["category_name"])],
                    cubes=[
                        CubePrimitive(
                            pose=Pose(
                                position=Vector3(
                                    x=ann["translation"][0],
                                    y=ann["translation"][1],
                                    z=ann["translation"][2],
                                ),
                                orientation=foxglove_Quaternion(
                                    w=ann["rotation"][0],
                                    x=ann["rotation"][1],
                                    y=ann["rotation"][2],
                                    z=ann["rotation"][3],
                                )
                            ),
                            size=Vector3(
                                x=ann["size"][1],
                                y=ann["size"][0],
                                z=ann["size"][2],
                            ),
                            color=Color(
                                r=c[0],
                                g=c[1],
                                b=c[2],
                                a=0.5,
                            )
                        )
                    ]
                )
                entities.append(entity)

            scene_update = SceneUpdate(entities=entities)
            foxglove.log("/markers/annotations", scene_update, log_time=stamp_ns)

            # publish /markers/car
            foxglove.log("/markers/car", get_car_scene_update(stamp_ns), log_time=stamp_ns)

            # collect all sensor frames after this sample but before the next sample
            non_keyframe_sensor_msgs = []
            for (sensor_id, sample_token) in cur_sample["data"].items():
                topic = ("/" + sensor_id).lower()

                next_sample_token = nusc.get("sample_data", sample_token)["next"]
                while next_sample_token != "":
                    next_sample_data = nusc.get("sample_data", next_sample_token)
                    if next_sample_data["is_key_frame"]:
                        break

                    pbar.update(1)
                    ego_pose = nusc.get("ego_pose", next_sample_data["ego_pose_token"])
                    ego_tf = get_ego_tf(ego_pose)
                    
                    # Convert FrameTransform timestamp back to nanoseconds
                    tf_stamp_ns = int(ego_pose["timestamp"]) * 1000
                    non_keyframe_sensor_msgs.append((tf_stamp_ns, "/tf", ego_tf))

                    if next_sample_data["sensor_modality"] == "radar":
                        msg = get_radar(data_path, next_sample_data, sensor_id.lower())
                        non_keyframe_sensor_msgs.append((int(next_sample_data["timestamp"]) * 1000, topic, msg))
                    elif next_sample_data["sensor_modality"] == "lidar":
                        msg = get_lidar(data_path, next_sample_data, sensor_id.lower())
                        non_keyframe_sensor_msgs.append((int(next_sample_data["timestamp"]) * 1000, topic, msg))
                    elif next_sample_data["sensor_modality"] == "camera":
                        msg = get_camera(data_path, next_sample_data, sensor_id.lower())
                        camera_stamp_nsec = int(next_sample_data["timestamp"]) * 1000
                        non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + "/image_rect_compressed", msg))

                        msg = get_camera_info(nusc, next_sample_data, sensor_id.lower())
                        non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + "/camera_info", msg))

                        closest_lidar = find_closest_lidar(nusc, cur_sample["data"]["LIDAR_TOP"], camera_stamp_nsec)
                        if closest_lidar is not None:
                            msg = get_lidar_image_annotations(nusc, closest_lidar, next_sample_data, sensor_id.lower())
                            non_keyframe_sensor_msgs.append(
                                (
                                    camera_stamp_nsec,
                                    topic + "/lidar",
                                    msg,
                                )
                            )

                    next_sample_token = next_sample_data["next"]

            # sort and publish the non-keyframe sensor msgs
            non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
            for (timestamp, topic, msg) in non_keyframe_sensor_msgs:
                foxglove.log(topic, msg, log_time=timestamp)

            # move to the next sample
            cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

        pbar.close()
        print(f"Finished writing {filepath}")


def convert_all(
    output_dir: Path,
    name: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus,
    selected_scenes,
):
    nusc.list_scenes()
    for scene in nusc.scene:
        scene_name = scene["name"]
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        mcap_name = f"NuScenes-{name}-{scene_name}.mcap"
        write_scene_to_mcap(nusc, nusc_can, scene, output_dir / mcap_name)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--data-dir",
        "-d",
        default=script_dir / "data",
        help="path to nuscenes data directory",
    )
    parser.add_argument(
        "--dataset-name",
        "-n",
        default=["v1.0-mini"],
        nargs="+",
        help="dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=script_dir / "output",
        help="path to write MCAP files into",
    )
    parser.add_argument("--scene", "-s", nargs="*", help="specific scene(s) to write")
    parser.add_argument("--list-only", action="store_true", help="lists the scenes and exits")

    args = parser.parse_args()

    nusc_can = NuScenesCanBus(dataroot=str(args.data_dir))

    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return
        convert_all(args.output_dir, name, nusc, nusc_can, args.scene)


if __name__ == "__main__":
    main()
