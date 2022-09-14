"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.timestamp_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class SceneEntityDeletion(google.protobuf.message.Message):
    """(Experimental, subject to change) Command to remove previously published entities"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _Type:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _TypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[SceneEntityDeletion._Type.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        MATCHING_ID: SceneEntityDeletion._Type.ValueType  # 0
        """Delete the existing entity on the same topic that has the provided `id`"""

        ALL: SceneEntityDeletion._Type.ValueType  # 1
        """Delete all existing entities on the same topic"""

    class Type(_Type, metaclass=_TypeEnumTypeWrapper):
        """(Experimental, subject to change) An enumeration indicating which entities should match a SceneEntityDeletion command"""
        pass

    MATCHING_ID: SceneEntityDeletion.Type.ValueType  # 0
    """Delete the existing entity on the same topic that has the provided `id`"""

    ALL: SceneEntityDeletion.Type.ValueType  # 1
    """Delete all existing entities on the same topic"""


    TIMESTAMP_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    ID_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Timestamp of the deletion. Only matching entities earlier than this timestamp will be deleted."""
        pass
    type: global___SceneEntityDeletion.Type.ValueType
    """Type of deletion action to perform"""

    id: typing.Text
    """Identifier which must match if `type` is `MATCHING_ID`."""

    def __init__(self,
        *,
        timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        type: global___SceneEntityDeletion.Type.ValueType = ...,
        id: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["timestamp",b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["id",b"id","timestamp",b"timestamp","type",b"type"]) -> None: ...
global___SceneEntityDeletion = SceneEntityDeletion
