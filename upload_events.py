import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

from foxglove_data_platform.client import Client
from mcap.mcap0.reader import make_reader

from sensor_msgs.msg import Imu
from visualization_msgs.msg import MarkerArray

from events import Annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="MCAP files to annotate")
    parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("FOXGLOVE_CONSOLE_TOKEN"),
        help="data platform secret token (if not provided, FOXGLOVE_CONSOLE_TOKEN from environment is used)",
    )
    args = parser.parse_args()
    if args.token is None:
        print("FOXGLOVE_CONSOLE_TOKEN not in environment", file=sys.stderr)
        return 1

    client = Client(token=args.token)
    device_ids = {resp["name"]: resp["id"] for resp in client.get_devices()}
    
    for filename in args.files:
        if Path(filename).is_dir:
        print(f"scanning {filename} for events...")
        annotator = Annotator()
        events = []
        device_id = None
        summary = None
        with open(filename, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            scene_info = next(
                (metadata for metadata in reader.iter_metadata() if metadata.name == "scene-info"),
                None
            )
            try:
                vehicle = scene_info.metadata["vehicle"]
                device_id = device_ids[vehicle]
            except KeyError:
                print(f"device ID not found for vehicle '{vehicle}' - has this MCAP been uploaded?", file=sys.stderr)
                return 1
        
            events.extend(annotator.on_mcap_start(summary, scene_info))
            for schema, _, message in reader.iter_messages(topics=["/markers/annotations", "/imu"]):
                if schema.name == "sensor_msgs/Imu":
                    imu = Imu()
                    imu.deserialize(message.data)
                    events.extend(annotator.on_imu(imu))
                        
                if schema.name == "visualization_msgs/MarkerArray":
                    marker_array = MarkerArray()
                    marker_array.deserialize(message.data)
                    events.extend(annotator.on_marker_array(marker_array))

            events.extend(annotator.on_mcap_end())

        # save existing events
        old_events = client.get_events(
            device_id=device_id,
            start=datetime.fromtimestamp(float(summary.statistics.message_start_time) / 1e9),
            end=datetime.fromtimestamp(float(summary.statistics.message_end_time) / 1e9),
        )
        
        print(f"uploading {len(events)} events for {filename} ...")
        for event in events:
            # create new events
            client.create_event(
                device_id=device_id,
                time=datetime.fromtimestamp(float(event.timestamp_ns) / 1e9),
                duration=event.duration_ns,
                metadata=event.metadata,
            )
        
        # destroy old events once new events have been uploaded
        print(f"deleting {len(old_events)} old events for {filename} ...")
        for old_event in old_events:
            client.delete_event(event_id=old_event["id"])
        return 0


if __name__ == "__main__":
    sys.exit(main())
