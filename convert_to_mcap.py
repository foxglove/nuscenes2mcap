import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped
from mcap.mcap0.writer import Writer
from nav_msgs.msg import OccupancyGrid, Odometry
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, NavSatFix
from std_msgs.msg import ColorRGBA
from tqdm import tqdm
from visualization_msgs.msg import ImageMarker, Marker, MarkerArray

from foxglove.FrameTransform_pb2 import FrameTransform
from foxglove.PackedElementField_pb2 import PackedElementField
from foxglove.PointCloud_pb2 import PointCloud
from foxglove.Quaternion_pb2 import Quaternion as foxglove_Quaternion
from foxglove.Vector3_pb2 import Vector3
from ProtobufWriter import ProtobufWriter
from RosmsgWriter import RosmsgWriter

with open(Path(__file__).parent / "turbomap.json") as f:
    TURBOMAP_DATA = np.array(json.load(f))


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
    assert location in REFERENCE_COORDINATES.keys(), f"Error: The given location: {location}, has no available reference."

    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    ts = get_time(pose)
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
    return lat, lon, ts


def get_translation(data):
    return Vector3(x=data["translation"][0], y=data["translation"][1], z=data["translation"][2])


def get_rotation(data):
    return foxglove_Quaternion(x=data["rotation"][1], y=data["rotation"][2], z=data["rotation"][3], w=data["rotation"][0])


def get_pose(data):
    p = Pose()
    p.position.x = data["translation"][0]
    p.position.y = data["translation"][1]
    p.position.z = data["translation"][2]

    p.orientation.w = data["rotation"][0]
    p.orientation.x = data["rotation"][1]
    p.orientation.y = data["rotation"][2]
    p.orientation.z = data["rotation"][3]

    return p


def get_time(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["timestamp"], 1_000_000)
    t.nsecs = msecs * 1000

    return t


def get_utime(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["utime"], 1_000_000)
    t.nsecs = msecs * 1000

    return t


def make_point(xyz):
    p = Point()
    p.x = xyz[0]
    p.y = xyz[1]
    p.z = xyz[2]
    return p


def make_point2d(xy):
    p = Point()
    p.x = xy[0]
    p.y = xy[1]
    p.z = 0.0
    return p


def make_color(rgb, a=1):
    c = ColorRGBA()
    c.r = rgb[0]
    c.g = rgb[1]
    c.b = rgb[2]
    c.a = a

    return c


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
    ("I", 1): PackedElementField.INT8,
    ("U", 1): PackedElementField.UINT8,
    ("I", 2): PackedElementField.INT16,
    ("U", 2): PackedElementField.UINT16,
    ("I", 4): PackedElementField.INT32,
    ("U", 4): PackedElementField.UINT32,
    ("F", 4): PackedElementField.FLOAT32,
    ("F", 8): PackedElementField.FLOAT64,
}


def get_radar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = PointCloud()
    msg.frame_id = frame_id
    msg.timestamp.FromMicroseconds(sample_data["timestamp"])
    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        msg.fields.add(name=name, offset=offset, type=PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)])
        offset += size

    msg.point_stride = offset
    msg.data = pc.pc_data.tobytes()
    return msg


def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data["filename"]
    msg = CompressedImage()
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    msg.format = "jpeg"
    with open(jpg_filename, "rb") as jpg_file:
        msg.data = jpg_file.read()
    return msg


def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    msg_info = CameraInfo()
    msg_info.header.frame_id = frame_id
    msg_info.header.stamp = get_time(sample_data)
    msg_info.height = sample_data["height"]
    msg_info.width = sample_data["width"]
    msg_info.K[0] = calib["camera_intrinsic"][0][0]
    msg_info.K[1] = calib["camera_intrinsic"][0][1]
    msg_info.K[2] = calib["camera_intrinsic"][0][2]
    msg_info.K[3] = calib["camera_intrinsic"][1][0]
    msg_info.K[4] = calib["camera_intrinsic"][1][1]
    msg_info.K[5] = calib["camera_intrinsic"][1][2]
    msg_info.K[6] = calib["camera_intrinsic"][2][0]
    msg_info.K[7] = calib["camera_intrinsic"][2][1]
    msg_info.K[8] = calib["camera_intrinsic"][2][2]

    msg_info.R[0] = 1
    msg_info.R[3] = 1
    msg_info.R[6] = 1

    msg_info.P[0] = msg_info.K[0]
    msg_info.P[1] = msg_info.K[1]
    msg_info.P[2] = msg_info.K[2]
    msg_info.P[3] = 0
    msg_info.P[4] = msg_info.K[3]
    msg_info.P[5] = msg_info.K[4]
    msg_info.P[6] = msg_info.K[5]
    msg_info.P[7] = 0
    msg_info.P[8] = 0
    msg_info.P[9] = 0
    msg_info.P[10] = 1
    msg_info.P[11] = 0
    return msg_info


def get_lidar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]

    with open(pc_filename, "rb") as pc_file:
        msg = PointCloud()
        msg.frame_id = frame_id
        msg.timestamp.FromMicroseconds(sample_data["timestamp"])
        msg.fields.add(name="x", offset=0, type=PackedElementField.FLOAT32),
        msg.fields.add(name="y", offset=4, type=PackedElementField.FLOAT32),
        msg.fields.add(name="z", offset=8, type=PackedElementField.FLOAT32),
        msg.fields.add(name="intensity", offset=12, type=PackedElementField.FLOAT32),
        msg.fields.add(name="ring", offset=16, type=PackedElementField.FLOAT32),
        msg.point_stride = len(msg.fields) * 4  # 4 bytes per field
        msg.data = pc_file.read()
        return msg


def get_lidar_imagemarkers(nusc, sample_lidar, sample_data, frame_id):
    # lidar image markers in camera frame
    points, coloring, _ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=sample_lidar["token"],
        camera_token=sample_data["token"],
        render_intensity=True,
    )
    points = points.transpose()

    marker = ImageMarker()
    marker.header.frame_id = frame_id
    marker.header.stamp = get_time(sample_data)
    marker.ns = "LIDAR_TOP"
    marker.id = 0
    marker.type = ImageMarker.POINTS
    marker.action = ImageMarker.ADD
    marker.scale = 2.0
    marker.points = [make_point2d(p) for p in points]
    marker.outline_colors = [make_color(c) for c in turbomap(coloring)]
    return marker


def get_remove_imagemarkers(frame_id, ns, stamp):
    marker = ImageMarker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = 0
    marker.action = ImageMarker.REMOVE
    return marker


def write_boxes_imagemarkers(nusc, rosmsg_writer, anns, sample_data, frame_id, topic_ns, stamp):
    # annotation boxes
    collector = Collector()
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data["token"])
    for box in boxes:
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(collector, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    marker = ImageMarker()
    marker.header.frame_id = frame_id
    marker.header.stamp = get_time(sample_data)
    marker.ns = "annotations"
    marker.id = 0
    marker.type = ImageMarker.LINE_LIST
    marker.action = ImageMarker.ADD
    marker.scale = 2.0
    marker.points = [make_point2d(p) for p in collector.points]
    marker.outline_colors = [make_color(c) for c in collector.colors]

    msg = ImageMarkerArray()
    msg.markers = [marker]

    rosmsg_writer.write_message(topic_ns + "/image_markers_annotations", msg, stamp)


def write_occupancy_grid(rosmsg_writer, nusc_map, ego_pose, stamp):
    translation = ego_pose["translation"]
    rotation = Quaternion(ego_pose["rotation"])
    yaw = quaternion_yaw(rotation) / np.pi * 180
    patch_box = (translation[0], translation[1], 32, 32)
    canvas_size = (patch_box[2] * 10, patch_box[3] * 10)

    drivable_area = nusc_map.get_map_mask(patch_box, yaw, ["drivable_area"], canvas_size)[0]
    drivable_area = (drivable_area * 100).astype(np.int8)

    msg = OccupancyGrid()
    msg.header.frame_id = "base_link"
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = 0.1
    msg.info.width = drivable_area.shape[1]
    msg.info.height = drivable_area.shape[0]
    msg.info.origin.position.x = -16.0
    msg.info.origin.position.y = -16.0
    msg.info.origin.orientation.w = 1
    msg.data = drivable_area.flatten().tolist()

    rosmsg_writer.write_message("/drivable_area", msg, stamp)


def get_imu_msg(imu_data):
    msg = Imu()
    msg.header.frame_id = "base_link"
    msg.header.stamp = get_utime(imu_data)
    msg.angular_velocity.x = imu_data["rotation_rate"][0]
    msg.angular_velocity.y = imu_data["rotation_rate"][1]
    msg.angular_velocity.z = imu_data["rotation_rate"][2]

    msg.linear_acceleration.x = imu_data["linear_accel"][0]
    msg.linear_acceleration.y = imu_data["linear_accel"][1]
    msg.linear_acceleration.z = imu_data["linear_accel"][2]

    msg.orientation.w = imu_data["q"][0]
    msg.orientation.x = imu_data["q"][1]
    msg.orientation.y = imu_data["q"][2]
    msg.orientation.z = imu_data["q"][3]

    return (msg.header.stamp, "/imu", msg)


def get_odom_msg(pose_data):
    msg = Odometry()
    msg.header.frame_id = "map"
    msg.header.stamp = get_utime(pose_data)
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = pose_data["pos"][0]
    msg.pose.pose.position.y = pose_data["pos"][1]
    msg.pose.pose.position.z = pose_data["pos"][2]
    msg.pose.pose.orientation.w = pose_data["orientation"][0]
    msg.pose.pose.orientation.x = pose_data["orientation"][1]
    msg.pose.pose.orientation.y = pose_data["orientation"][2]
    msg.pose.pose.orientation.z = pose_data["orientation"][3]
    msg.twist.twist.linear.x = pose_data["vel"][0]
    msg.twist.twist.linear.y = pose_data["vel"][1]
    msg.twist.twist.linear.z = pose_data["vel"][2]
    msg.twist.twist.angular.x = pose_data["rotation_rate"][0]
    msg.twist.twist.angular.y = pose_data["rotation_rate"][1]
    msg.twist.twist.angular.z = pose_data["rotation_rate"][2]

    return (msg.header.stamp, "/odom", msg)


def get_basic_can_msg(name, diag_data):
    values = []
    for (key, value) in diag_data.items():
        if key != "utime":
            values.append(KeyValue(key=key, value=str(round(value, 4))))

    msg = DiagnosticArray()
    msg.header.stamp = get_utime(diag_data)
    msg.status.append(DiagnosticStatus(name=name, level=0, message="OK", values=values))

    return (msg.header.stamp, "/diagnostics", msg)


def get_ego_tf(ego_pose):
    ego_tf = FrameTransform()
    ego_tf.parent_frame_id = "map"
    ego_tf.timestamp.FromMicroseconds(ego_pose["timestamp"])
    ego_tf.child_frame_id = "base_link"
    ego_tf.translation.CopyFrom(get_translation(ego_pose))
    ego_tf.rotation.CopyFrom(get_rotation(ego_pose))
    return ego_tf


def get_sensor_tf(nusc, sensor_id, sample_data):
    sensor_tf = FrameTransform()
    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(sample_data["timestamp"])
    sensor_tf.child_frame_id = sensor_id
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor_tf.translation.CopyFrom(get_translation(calibrated_sensor))
    sensor_tf.rotation.CopyFrom(get_rotation(calibrated_sensor))
    return sensor_tf


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


def get_scene_map(nusc, scene, nusc_map, image, stamp):
    x, y, w, h = scene_bounding_box(nusc, scene, nusc_map)
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    img = np.flipud(image)[img_y : img_y + img_h, img_x : img_x + img_w]
    img = (img * (100.0 / 255.0)).astype(np.int8)

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = 0.1
    msg.info.width = img_w
    msg.info.height = img_h
    msg.info.origin.position.x = x
    msg.info.origin.position.y = y
    msg.info.origin.orientation.w = 1
    msg.data = img.flatten().tolist()

    return msg


def rectContains(rect, point):
    a, b, c, d = rect
    x, y = point[:2]
    return a <= x < a + c and b <= y < b + d


def get_centerline_markers(nusc, scene, nusc_map, stamp):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained_pose_lists = []
    for pose_list in pose_lists:
        new_pose_list = []
        for pose in pose_list:
            if rectContains(bbox, pose):
                new_pose_list.append(pose)
        if len(new_pose_list) > 0:
            contained_pose_lists.append(new_pose_list)

    msg = MarkerArray()
    for i, pose_list in enumerate(contained_pose_lists):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "centerline"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.frame_locked = True
        marker.scale.x = 0.1
        marker.color.r = 51.0 / 255.0
        marker.color.g = 160.0 / 255.0
        marker.color.b = 44.0 / 255.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for pose in pose_list:
            marker.points.append(Point(pose[0], pose[1], 0))
        msg.markers.append(marker)

    return msg


def find_closest_lidar(nusc, lidar_start_token, stamp_nsec):
    candidates = []

    next_lidar_token = nusc.get("sample_data", lidar_start_token)["next"]
    while next_lidar_token != "":
        lidar_data = nusc.get("sample_data", next_lidar_token)
        if lidar_data["is_key_frame"]:
            break

        dist_abs = abs(stamp_nsec - get_time(lidar_data).to_nsec())
        candidates.append((dist_abs, lidar_data))
        next_lidar_token = lidar_data["next"]

    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda x: x[0])[1]


def get_car_marker(stamp):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = stamp
    marker.id = 99999999
    marker.ns = "car"
    marker.type = Marker.MESH_RESOURCE
    marker.pose.position.x = 1
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.w = 1
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.frame_locked = True
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.mesh_resource = "https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    marker.mesh_use_embedded_materials = True
    return marker


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

    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp)
        protobuf_writer = ProtobufWriter(writer)
        rosmsg_writer = RosmsgWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")

        writer.add_metadata(
            "scene-info",
            {
                "description": scene["description"],
                "name": scene["name"],
                "location": location,
                "vehicle": log["vehicle"],
                "date_captured": log["date_captured"],
            },
        )

        stamp = get_time(
            nusc.get(
                "ego_pose",
                nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])["ego_pose_token"],
            )
        )
        map_msg = get_scene_map(nusc, scene, nusc_map, image, stamp)
        centerlines_msg = get_centerline_markers(nusc, scene, nusc_map, stamp)
        rosmsg_writer.write_message("/map", map_msg, stamp)
        rosmsg_writer.write_message("/semantic_map", centerlines_msg, stamp)
        last_map_stamp = stamp

        while cur_sample is not None:
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            stamp = get_time(ego_pose)

            # write map topics every two seconds
            if stamp - rospy.Duration(2.0) >= last_map_stamp:
                map_msg.header.stamp = stamp
                for marker in centerlines_msg.markers:
                    marker.header.stamp = stamp
                rosmsg_writer.write_message("/map", map_msg, stamp)
                rosmsg_writer.write_message("/semantic_map", centerlines_msg, stamp)
                last_map_stamp = stamp

            # write CAN messages to /pose, /odom, and /diagnostics
            can_msg_events = []
            for i in range(len(can_parsers)):
                (can_msgs, index, msg_func) = can_parsers[i]
                while index < len(can_msgs) and get_utime(can_msgs[index]) < stamp:
                    can_msg_events.append(msg_func(can_msgs[index]))
                    index += 1
                    can_parsers[i][1] = index
            can_msg_events.sort(key=lambda x: x[0])
            for (msg_stamp, topic, msg) in can_msg_events:
                rosmsg_writer.write_message(topic, msg, msg_stamp)

            # publish /tf
            protobuf_writer.write_message("/tf", get_ego_tf(ego_pose), stamp.to_nsec())

            # /driveable_area occupancy grid
            write_occupancy_grid(rosmsg_writer, nusc_map, ego_pose, stamp)

            # iterate sensors
            for (sensor_id, sample_token) in cur_sample["data"].items():
                pbar.update(1)
                sample_data = nusc.get("sample_data", sample_token)
                topic = "/" + sensor_id

                # create sensor transform
                protobuf_writer.write_message("/tf", get_sensor_tf(nusc, sensor_id, sample_data), stamp.to_nsec())

                # write the sensor data
                if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message(topic, msg, stamp.to_nsec())
                elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message(topic, msg, stamp.to_nsec())
                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    rosmsg_writer.write_message(topic + "/image_rect_compressed", msg, stamp)
                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    rosmsg_writer.write_message(topic + "/camera_info", msg, stamp)

                if sample_data["sensor_modality"] == "camera":
                    msg = get_lidar_imagemarkers(nusc, sample_lidar, sample_data, sensor_id)
                    rosmsg_writer.write_message(topic + "/image_markers_lidar", msg, stamp)
                    write_boxes_imagemarkers(
                        nusc,
                        rosmsg_writer,
                        cur_sample["anns"],
                        sample_data,
                        sensor_id,
                        topic,
                        stamp,
                    )

            # publish /pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.header.stamp = stamp
            pose_stamped.pose.orientation.w = 1
            rosmsg_writer.write_message("/pose", pose_stamped, stamp)

            # publish /gps
            lat, lon, ts = derive_latlon(location, ego_pose)
            gps = NavSatFix()
            gps.header.frame_id = "base_link"
            gps.header.stamp = ts
            gps.status.status = 1
            gps.status.service = 1
            gps.latitude = lat
            gps.longitude = lon
            gps.altitude = get_translation(ego_pose).z
            rosmsg_writer.write_message("/gps", gps, stamp)

            # publish /markers/annotations
            marker_array = MarkerArray()
            for annotation_id in cur_sample["anns"]:
                ann = nusc.get("sample_annotation", annotation_id)
                marker_id = int(ann["instance_token"][:4], 16)
                c = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = stamp
                marker.id = marker_id
                marker.ns = ann["category_name"]
                marker.text = ann["instance_token"][:4]
                marker.type = Marker.CUBE
                marker.pose = get_pose(ann)
                marker.frame_locked = True
                marker.scale.x = ann["size"][1]
                marker.scale.y = ann["size"][0]
                marker.scale.z = ann["size"][2]
                marker.color = make_color(c, 0.5)
                marker_array.markers.append(marker)
            rosmsg_writer.write_message("/markers/annotations", marker_array, stamp)

            # publish /markers/car
            car_marker_array = MarkerArray()
            car_marker_array.markers.append(get_car_marker(stamp))
            rosmsg_writer.write_message("/markers/car", car_marker_array, stamp)

            # collect all sensor frames after this sample but before the next sample
            non_keyframe_sensor_msgs = []
            for (sensor_id, sample_token) in cur_sample["data"].items():
                topic = "/" + sensor_id

                next_sample_token = nusc.get("sample_data", sample_token)["next"]
                while next_sample_token != "":
                    next_sample_data = nusc.get("sample_data", next_sample_token)
                    # if next_sample_data['is_key_frame'] or get_time(next_sample_data).to_nsec() > next_stamp.to_nsec():
                    #     break
                    if next_sample_data["is_key_frame"]:
                        break

                    pbar.update(1)
                    ego_pose = nusc.get("ego_pose", next_sample_data["ego_pose_token"])
                    ego_tf = get_ego_tf(ego_pose)
                    non_keyframe_sensor_msgs.append((ego_tf.timestamp.ToNanoseconds(), "/tf", ego_tf))

                    if next_sample_data["sensor_modality"] == "radar":
                        msg = get_radar(data_path, next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append((msg.timestamp.ToNanoseconds(), topic, msg))
                    elif next_sample_data["sensor_modality"] == "lidar":
                        msg = get_lidar(data_path, next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append((msg.timestamp.ToNanoseconds(), topic, msg))
                    elif next_sample_data["sensor_modality"] == "camera":
                        msg = get_camera(data_path, next_sample_data, sensor_id)
                        camera_stamp_nsec = msg.header.stamp.to_nsec()
                        non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + "/image_rect_compressed", msg))

                        msg = get_camera_info(nusc, next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + "/camera_info", msg))

                        closest_lidar = find_closest_lidar(nusc, cur_sample["data"]["LIDAR_TOP"], camera_stamp_nsec)
                        if closest_lidar is not None:
                            msg = get_lidar_imagemarkers(nusc, closest_lidar, next_sample_data, sensor_id)
                            non_keyframe_sensor_msgs.append(
                                (
                                    msg.header.stamp.to_nsec(),
                                    topic + "/image_markers_lidar",
                                    msg,
                                )
                            )
                        else:
                            msg = get_remove_imagemarkers(sensor_id, "LIDAR_TOP", msg.header.stamp)
                            non_keyframe_sensor_msgs.append(
                                (
                                    msg.header.stamp.to_nsec(),
                                    topic + "/image_markers_lidar",
                                    msg,
                                )
                            )

                    next_sample_token = next_sample_data["next"]

            # sort and publish the non-keyframe sensor msgs
            non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
            for (_, topic, msg) in non_keyframe_sensor_msgs:
                if hasattr(msg, "header"):
                    rosmsg_writer.write_message(topic, msg, msg.header.stamp)
                else:
                    protobuf_writer.write_message(topic, msg, msg.timestamp.ToNanoseconds())

            # move to the next sample
            cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

        pbar.close()
        writer.finish()
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
