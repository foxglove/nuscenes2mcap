from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Event:
    timestamp_ns: int
    duration_ns: int
    metadata: Dict[str, str] = field(default_factory=dict)


def to_ns(rospy_time):
    return rospy_time.nsecs + (1_000_000_000 * rospy_time.secs)


class Annotator:
    ACCELERATION_THRESHOLD = 1.5
    PEDESTRIAN_THRESHOLD = 10

    def __init__(self):
        self.jerk_start_time = None
        self.max_acceleration = 0.0
        self.ped_event_start_time = None
        self.max_num_peds = 0
        self.summary = None

    def on_mcap_start(self, summary, scene_info = None) -> List[Event]:
        self.summary = summary
        if scene_info is None:
            return []
        tags = [tag.strip() for tag in scene_info.metadata["description"].split(",")]
        metadata = {tag: "true" for tag in tags}
        metadata["category"] = "scene_description"
        return [
            Event(
                timestamp_ns=summary.statistics.message_start_time,
                duration_ns=(summary.statistics.message_end_time - summary.statistics.message_start_time),
                metadata=metadata,
            )
        ]

    def on_imu(self, imu) -> List[Event]:
        longitudinal_acceleration = abs(imu.linear_acceleration.x)
        if longitudinal_acceleration >= self.ACCELERATION_THRESHOLD:
            if self.jerk_start_time is None:
                self.jerk_start_time = imu.header.stamp
            self.max_acceleration = max(longitudinal_acceleration, self.max_acceleration)
        if longitudinal_acceleration < self.ACCELERATION_THRESHOLD and self.jerk_start_time is not None:
            event = Event(
                timestamp_ns=to_ns(self.jerk_start_time),
                duration_ns=to_ns(imu.header.stamp - self.jerk_start_time),
                metadata={
                    "category": "large_acceleration",
                    "max": f"{self.max_acceleration:.2f}"
                },
            ) 
            self.jerk_start_time = None
            self.max_acceleration = 0
            return [event]
        return []

    def on_marker_array(self, marker_array) -> List[Event]:
        num_peds = sum(1 for marker in marker_array.markers if marker.ns.startswith("human.pedestrian"))
        stamp = next((marker.header.stamp for marker in marker_array.markers), None)
        if num_peds > self.PEDESTRIAN_THRESHOLD:
            if self.ped_event_start_time is None:
                self.ped_event_start_time = stamp
            self.max_num_peds = max(self.max_num_peds, num_peds)
        if num_peds < self.PEDESTRIAN_THRESHOLD and self.ped_event_start_time is not None:
            event = Event(
                timestamp_ns=to_ns(self.ped_event_start_time),
                duration_ns=to_ns(stamp - self.ped_event_start_time),
                metadata={
                    "category": "many_pedestrians",
                    "max": str(self.max_num_peds)
                },
            ) 
            self.ped_event_start_time = None
            self.max_num_peds = 0
            return [event]
        return []

    def on_mcap_end(self) -> List[Event]:
        if self.summary is None:
            return []
        events = []

        if self.jerk_start_time is not None:
            events.append(Event(
                timestamp_ns=to_ns(self.jerk_start_time),
                duration_ns=self.summary.statistics.message_end_time - to_ns(self.jerk_start_time),
                metadata={
                    "category": "large_acceleration",
                    "max": f"{self.max_acceleration:.2f}"
                },
            ))

        if self.ped_event_start_time is not None:
            events.append(Event(
                timestamp_ns=to_ns(self.ped_event_start_time),
                duration_ns=self.summary.statistics.message_end_time - to_ns(self.ped_event_start_time),
                metadata={
                    "category": "many_pedestrians",
                    "max": str(self.max_num_peds),
                },
            ))
        return events
