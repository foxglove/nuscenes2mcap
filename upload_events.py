import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

from foxglove_data_platform.client import Client
from mcap.mcap0.reader import make_reader

from sensor_msgs.msg import Imu
from foxglove.SceneUpdate_pb2 import SceneUpdate

from device_name import make_device_name
from event_helpers.annotators import Annotator
from event_helpers.client_utils import get_all_events_for_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="MCAP files to annotate")
    parser.add_argument(
        "--token",
        "-t",
        help="data platform secret token (if not provided, FOXGLOVE_DATA_PLATFORM_TOKEN from environment is used)",
    )
    parser.add_argument("--host", default="api.foxglove.dev", help="custom host to direct API requests to")
    parser.add_argument("--commit", "-y", action="store_true", help="actually send the events")
    args = parser.parse_args()
    if args.token is None:
        token = os.environ.get("FOXGLOVE_DATA_PLATFORM_TOKEN")
        if token is None:
            print("FOXGLOVE_DATA_PLATFORM_TOKEN not in environment", file=sys.stderr)
            return 1
        args.token = token

    client = Client(token=args.token, host=args.host)
    device_ids = {resp["name"]: resp["id"] for resp in client.get_devices()}

    filepaths = []
    for name in args.files:
        path = Path(name)
        if path.is_dir():
            filepaths.extend(path.glob("*.mcap"))
        elif path.is_file():
            filepaths.append(path)
        else:
            raise RuntimeError(f"path does not exist: {name}")

    for filepath in filepaths:
        print(f"scanning {filepath} for events...")
        annotator = Annotator()
        events = []
        device_id = None
        summary = None
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            scene_info = next(
                (metadata for metadata in reader.iter_metadata() if metadata.name == "scene-info"),
                None,
            )
            device_name = make_device_name(scene_info.metadata)
            try:
                device_id = device_ids[device_name]
            except KeyError:
                print(
                    f"device ID not found for '{device_name}' - has this MCAP been uploaded?",
                    file=sys.stderr,
                )
                return 1

            events.extend(annotator.on_mcap_start(summary, scene_info))
            for schema, _, message in reader.iter_messages(topics=["/markers/annotations", "/imu"]):
                if schema.name == "sensor_msgs/Imu":
                    imu = Imu()
                    imu.deserialize(message.data)
                    events.extend(annotator.on_imu(imu))

                if schema.name == "foxglove.SceneUpdate":
                    scene_update = SceneUpdate()
                    scene_update.ParseFromString(message.data)
                    events.extend(annotator.on_scene_update(scene_update))

            events.extend(annotator.on_mcap_end())

        # save existing events
        old_events = get_all_events_for_device(
            client=client,
            device_id=device_id,
            start=datetime.fromtimestamp(float(summary.statistics.message_start_time) / 1e9),
            end=datetime.fromtimestamp(float(summary.statistics.message_end_time) / 1e9),
        )

        print(f"uploading {len(events)} events for {filepath} ...")
        for event in events:
            # create new events
            if args.commit:
                client.create_event(
                    device_id=device_id,
                    time=datetime.fromtimestamp(float(event.timestamp_ns) / 1e9),
                    duration=event.duration_ns,
                    metadata=event.metadata,
                )
            else:
                print(f"would upload: {event}")

        # destroy old events once new events have been uploaded
        print(f"deleting {len(old_events)} old events for {filepath} ...")
        for old_event in old_events:
            if args.commit:
                client.delete_event(event_id=old_event["id"])
            else:
                print(f"would delete: {old_event}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
