from typing import List, Dict

from . import Event


def to_ns(rospy_time) -> int:
    """Converts a rospy.Time object to an integer nanosecond count."""
    return rospy_time.nsecs + (1_000_000_000 * rospy_time.secs)


class LatchingEventSource:
    def __init__(self, cooldown_period=100_000_000):
        """base class for any event source that implements"""
        self._start_time = None
        self._last_activated_time = None
        self._cooldown_period = cooldown_period

    def activate(self, value) -> bool:
        raise NotImplementedError(f"Implement this method, using {value}")

    def event_metadata(self) -> Dict[str, str]:
        raise NotImplementedError("Implement this method")

    def reset(self):
        raise NotImplementedError("Implement this method")

    def tick(self, value, timestamp_ns):
        if self.activate(value):
            if self._start_time is None:
                self._start_time = timestamp_ns
            self._last_activated_time = timestamp_ns
            return []
        if self._last_activated_time is None:
            return []
        if (timestamp_ns - self._last_activated_time) < self._cooldown_period:
            return []
        result = Event(
            timestamp_ns=self._start_time,
            duration_ns=(self._last_activated_time - self._start_time),
            metadata=self.event_metadata(),
        )
        self._start_time = None
        self._last_activated_time = None
        self.reset()
        return [result]

    def finish(self, end_timestamp_ns):
        if self._start_time is not None:
            return [
                Event(
                    timestamp_ns=self._start_time,
                    duration_ns=(end_timestamp_ns - self._start_time),
                    metadata=self.event_metadata(),
                )
            ]
        return []


class PedestrianEventSource(LatchingEventSource):
    def __init__(self):
        super().__init__()
        self.max_pedestrians = 0

    def activate(self, value):
        self.max_pedestrians = max(self.max_pedestrians, value)
        return value > 10

    def event_metadata(self):
        return {
            "category": "many_pedestrians",
            "max_pedestrians": str(self.max_pedestrians),
        }

    def reset(self):
        self.max_pedestrians = 0


class AccelerationEventSource(LatchingEventSource):
    def __init__(self):
        super().__init__()
        self.max_acceleration = 0

    def activate(self, value):
        if abs(value) > abs(self.max_acceleration):
            self.max_acceleration = value
        return abs(value) > 1.5

    def event_metadata(self):
        return {
            "category": "large_acceleration",
            "max": f"{self.max_acceleration:.2f}",
        }

    def reset(self):
        self.max_acceleration = 0


class Annotator:
    def __init__(self):
        self.summary = None
        self.ped_event_source = PedestrianEventSource()
        self.acc_event_source = AccelerationEventSource()

    def on_mcap_start(self, summary, scene_info=None) -> List[Event]:
        self.summary = summary
        if scene_info is None:
            return []
        tags = [tag.strip() for tag in scene_info.metadata["description"].split(",")]
        metadata = {tag: "true" for tag in tags}
        metadata["category"] = "scene_description"
        metadata["nuscene"] = scene_info.metadata["name"]
        metadata["city"] = scene_info.metadata["location"]

        return [
            Event(
                timestamp_ns=summary.statistics.message_start_time,
                duration_ns=(summary.statistics.message_end_time - summary.statistics.message_start_time),
                metadata=metadata,
            )
        ]

    def on_imu(self, imu) -> List[Event]:
        longitudinal_acceleration = imu.linear_acceleration.x
        return self.acc_event_source.tick(longitudinal_acceleration, to_ns(imu.header.stamp))

    def on_scene_update(self, scene_update) -> List[Event]:
        num_peds = 0
        timestamp = None
        for entity in scene_update.entities:
            if timestamp is None:
                timestamp = entity.timestamp
            for metadata in entity.metadata:
                if metadata.key == "category" and metadata.value.startswith("human.pedestrian"):
                    num_peds += 1

        stamp_ns = timestamp.nanos + (1_000_000_000 * timestamp.seconds)

        return self.ped_event_source.tick(num_peds, stamp_ns)

    def on_mcap_end(self) -> List[Event]:
        if self.summary is None:
            return []
        final_ped_events = self.ped_event_source.finish(self.summary.statistics.message_end_time)
        final_acc_events = self.acc_event_source.finish(self.summary.statistics.message_end_time)
        return final_ped_events + final_acc_events
