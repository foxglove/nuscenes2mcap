from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Event:
    timestamp_ns: int
    duration_ns: int
    metadata: Dict[str, str] = field(default_factory=dict)
