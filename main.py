from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import CameraInfo, CompressedImage, NavSatFix, PointCloud2, PointField
from typing import List, Tuple, Dict
from visualization_msgs.msg import ImageMarker, Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from mcap.mcap0.writer import Writer

import RosmsgWriter
import can

import math
import rospy
import numpy
import os
import json

EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def make_color(rgb, a=1):
    c = ColorRGBA()
    c.r = rgb[0]
    c.g = rgb[1]
    c.b = rgb[2]
    c.a = a

    return c


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


def get_lidar(sample_data, frame_id):
    pc_filename = "data/" + sample_data["filename"]
    pc_filesize = os.stat(pc_filename).st_size

    with open(pc_filename, "rb") as pc_file:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = get_time(sample_data)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = len(msg.fields) * 4  # 4 bytes per field
        msg.row_step = pc_filesize
        msg.width = round(pc_filesize / msg.point_step)
        msg.height = 1  # unordered
        msg.data = pc_file.read()
        return msg


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
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat), math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat)
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

    coordinates = []
    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    ts = pose["timestamp"]
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
    return {"latitude": lat, "longitude": lon}


def get_transform(data):
    t = Transform()
    t.translation.x = data["translation"][0]
    t.translation.y = data["translation"][1]
    t.translation.z = data["translation"][2]

    t.rotation.w = data["rotation"][0]
    t.rotation.x = data["rotation"][1]
    t.rotation.y = data["rotation"][2]
    t.rotation.z = data["rotation"][3]

    return t


def get_time(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["timestamp"], 1_000_000)
    t.nsecs = msecs * 1000

    return t


NUSCENES_VERSION = "v1.0-mini"

nusc = NuScenes(version=NUSCENES_VERSION, dataroot="data", verbose=True)
nusc_can = NuScenesCanBus(dataroot="data")

nusc.list_scenes()

scene = nusc.scene[0]

scene_name = scene["name"]
log = nusc.get("log", scene["log_token"])
location = log["location"]

cur_sample = nusc.get("sample", scene["first_sample_token"])

# the sensor messages we want to record
lidar_top_data = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
cam_front_data = nusc.get("sample_data", cur_sample["data"]["CAM_FRONT"])

sensors = [lidar_top_data, cam_front_data]

mcap_file_name = f"output.mcap"
mcap_file_path = os.path.join(os.path.abspath(os.curdir), mcap_file_name)
print(f"Writing to {mcap_file_path}")

stream = open(mcap_file_path, "wb")
writer = Writer(stream)

rosmsgWriter = RosmsgWriter.RosmsgWriter(writer)

writer.start(profile="ros1", library="nuscenes2bag")

event_schema_id = writer.register_schema(
    name="Event",
    encoding="jsonschema",
    data=json.dumps(
        {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "string",
                }
            },
        }
    ).encode(),
)

events_channel_id = writer.register_channel(
    schema_id=event_schema_id,
    topic="/events",
    message_encoding="json",
)

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

can_parsers = can.get_can_parsers(nusc_can, scene_name)

while cur_sample is not None:
    print("----")

    messages = []

    sample_stamp = get_time(cur_sample)

    for idx, sensor in enumerate(sensors):
        # there is one ego pose entry for every sample_data entry
        while sensor is not None and sensor["sample_token"] == cur_sample["token"]:
            ego_pose = nusc.get("ego_pose", sensor["ego_pose_token"])
            stamp = get_time(ego_pose)

            # publish /pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.header.stamp = stamp
            pose_stamped.pose.orientation.w = 1
            messages.append([pose_stamped.header.stamp.to_nsec(), pose_stamped.header.stamp, "/pose", pose_stamped])

            # create tf
            transforms = []

            ego_transform = get_transform(ego_pose)

            # create ego transform
            ego_tf = TransformStamped()
            ego_tf.header.frame_id = "map"
            ego_tf.header.stamp = stamp
            ego_tf.child_frame_id = "base_link"
            ego_tf.transform = ego_transform
            transforms.append(ego_tf)

            tf_array = TFMessage()
            tf_array.transforms = transforms

            messages.append([stamp.to_nsec(), stamp, "/tf", tf_array])

            # publish /gps
            coordinates = derive_latlon(location, ego_pose)
            gps = NavSatFix()
            gps.header.frame_id = "base_link"
            gps.header.stamp = stamp
            gps.status.status = 1
            gps.status.service = 1
            gps.latitude = coordinates["latitude"]
            gps.longitude = coordinates["longitude"]
            gps.altitude = ego_transform.translation.z
            messages.append([stamp.to_nsec(), stamp, "/gps", gps])

            if sensor["sensor_modality"] == "lidar":
                sensor_id = sensor["channel"]
                sensor_msg = get_lidar(sensor, sensor_id)
                messages.append([sensor_msg.header.stamp.to_nsec(), sensor_msg.header.stamp, "/" + sensor_id, sensor_msg])

                # create sensor transform
                sensor_tf = TransformStamped()
                sensor_tf.header.frame_id = "base_link"
                sensor_tf.header.stamp = sensor_msg.header.stamp
                sensor_tf.child_frame_id = sensor_id
                sensor_tf.transform = get_transform(nusc.get("calibrated_sensor", sensor["calibrated_sensor_token"]))
                transforms.append(sensor_tf)

                tf_array = TFMessage()
                tf_array.transforms = transforms

                messages.append([sensor_msg.header.stamp.to_nsec(), sensor_msg.header.stamp, "/tf", tf_array])

            print(stamp)
            print(sensor)
            sensor = nusc.get("sample_data", sensor["next"]) if sensor["next"] != "" else None
            sensors[idx] = sensor

    # publish /markers/annotations
    annotation_summary = dict()

    marker_array = MarkerArray()
    for annotation_id in cur_sample["anns"]:
        ann = nusc.get("sample_annotation", annotation_id)
        marker_id = int(ann["instance_token"][:4], 16)
        c = numpy.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = sample_stamp
        marker.id = marker_id
        marker.text = ann["instance_token"][:4]
        marker.type = Marker.CUBE
        marker.pose = get_pose(ann)
        marker.frame_locked = True
        marker.scale.x = ann["size"][1]
        marker.scale.y = ann["size"][0]
        marker.scale.z = ann["size"][2]
        marker.color = make_color(c, 0.5)
        marker_array.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = sample_stamp
        marker.id = marker_id
        marker.ns = "category_name"
        marker.text = ann["category_name"]
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose = get_pose(ann)
        marker.frame_locked = True
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color = make_color(c, 0.9)
        marker_array.markers.append(marker)

        key = "obj." + ann["category_name"]
        category_count = annotation_summary.get(key, 0)
        annotation_summary[key] = category_count + 1

    messages.append([sample_stamp.to_nsec(), sample_stamp, "/markers/annotations", marker_array])

    # Write annotation summary event
    annotation_summary["type"] = "annotation_summary"
    writer.add_message(
        channel_id=events_channel_id,
        log_time=sample_stamp.to_nsec(),
        data=json.dumps(annotation_summary).encode("utf-8"),
        publish_time=sample_stamp.to_nsec(),
    )

    # get all the CAN messages through with stamp <= our last stamp
    for i in range(len(can_parsers)):
        (can_msgs, index, msg_func) = can_parsers[i]
        while index < len(can_msgs) and can.get_utime(can_msgs[index]) < stamp:
            res_tuple = msg_func(can_msgs[index])
            messages.append([res_tuple[0].to_nsec(), res_tuple[0], res_tuple[1], res_tuple[2]])

            index += 1
            can_parsers[i][1] = index

    messages.sort(key=lambda x: x[0])
    for (stamp, _, topic, msg) in messages:
        rosmsgWriter.write_message(topic, msg, stamp)

    cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample["next"] != "" else None

writer.finish()
stream.close()

print("done")
