from foxglove_data_platform.client import Client
from datetime import datetime
from typing import Optional

PAGE_LENGTH = 100


def get_all_events_for_device(client: Client, device_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None):
    """The client.get_events API is paginated, meaning to successfully find all events
    we must call the API repeatedly with a changing offset.
    """
    all_events = []
    offset = 0
    while True:
        events_returned = client.get_events(device_id=device_id, start=start, end=end, limit=PAGE_LENGTH, offset=offset)
        all_events.extend(events_returned)
        offset += len(events_returned)
        if len(events_returned) != PAGE_LENGTH:
            return all_events
