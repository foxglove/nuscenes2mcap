from typing import Any, Dict, Optional
from io import BytesIO

import rospy


class RosmsgWriter:
    def __init__(self, output):
        self.__writer = output
        self.__schema_ids: Dict[str, int] = {}
        self.__channel_ids: Dict[str, int] = {}

    def write_message(
        self,
        topic: str,
        message: Any,
        log_time: Optional[int] = None,
        publish_time: Optional[int] = None,
        sequence: int = 0,
    ):
        """
        Writes a message to the MCAP stream, automatically registering schemas and channels as
        needed.
        @param topic: The topic of the message.
        @param message: The message to write.
        @param log_time: The time at which the message was logged.
            Will default to the current time if not specified.
        @param publish_time: The time at which the message was published.
            Will default to the current time if not specified.
        @param sequence: An optional sequence number.
        """
        if message._type not in self.__schema_ids.keys():
            schema_id = self.__writer.register_schema(
                name=message._type,
                data=message.__class__._full_text.encode(),
                encoding="ros1msg",
            )
            self.__schema_ids[message._type] = schema_id
        schema_id = self.__schema_ids[message._type]

        if topic not in self.__channel_ids.keys():
            channel_id = self.__writer.register_channel(
                topic=topic,
                message_encoding="ros1",
                schema_id=schema_id,
            )
            self.__channel_ids[topic] = channel_id
        channel_id = self.__channel_ids[topic]

        buffer = BytesIO()
        message.serialize(buffer)
        if isinstance(log_time, rospy.Time):
            log_time = int((log_time.secs * int(1e9)) + log_time.nsecs)
        self.__writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time or log_time,
            sequence=sequence,
            data=buffer.getvalue(),
        )
