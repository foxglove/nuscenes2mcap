from typing import Any, Dict, Optional
from mcap_protobuf.schema import register_schema


class ProtobufWriter:
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
        msg_typename = type(message).DESCRIPTOR.full_name
        schema_id = self.__schema_ids.get(msg_typename)
        if schema_id is None:
            schema_id = register_schema(self.__writer, type(message))
            self.__schema_ids[msg_typename] = schema_id

        channel_id = self.__channel_ids.get(topic)
        if channel_id is None:
            channel_id = self.__writer.register_channel(
                topic=topic,
                message_encoding="protobuf",
                schema_id=schema_id,
            )
            self.__channel_ids[topic] = channel_id

        self.__writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time or log_time,
            sequence=sequence,
            data=message.SerializeToString(),
        )
