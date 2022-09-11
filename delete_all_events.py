import argparse
import os
import sys

from foxglove_data_platform.client import Client
from event_helpers.client_utils import get_all_events_for_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        "-t",
        help="data platform secret token (if not provided, FOXGLOVE_DATA_PLATFORM_TOKEN from environment is used)",
    )
    parser.add_argument("--commit", "-y", action="store_true", help="actually perform deletions")
    parser.add_argument("--host", default="api.foxglove.dev", help="custom host to send data to")
    args = parser.parse_args()
    if args.token is None:
        token = os.environ.get("FOXGLOVE_DATA_PLATFORM_TOKEN")
        if token is None:
            print("FOXGLOVE_DATA_PLATFORM_TOKEN not in environment", file=sys.stderr)
            return 1
        args.token = token

    client = Client(token=args.token, host=args.host)
    device_ids = [resp["id"] for resp in client.get_devices()]
    print(f"Found events for devices: {device_ids}")

    # find all the events
    events_to_delete = []
    for device_id in device_ids:
        events_to_delete.extend(get_all_events_for_device(client, device_id))

    # destroy all events
    for old_event in events_to_delete:
        if not args.commit:
            print(f"Would delete: {old_event}")
        else:
            client.delete_event(event_id=old_event["id"])
    print(f"would have deleted {len(events_to_delete)} events for devices: {device_ids}")


if __name__ == "__main__":
    sys.exit(main())
