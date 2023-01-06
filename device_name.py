import hashlib
from typing import Dict


def make_device_name(metadata: Dict[str, str]) -> str:
    """constructs a synthetic device name from vehicle name and location."""
    vehicle = metadata["vehicle"]
    location = metadata["location"]
    return hashlib.sha256((vehicle + location).encode("utf8")).hexdigest()[:6]
