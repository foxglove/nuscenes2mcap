"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Generated by https://github.com/foxglove/schemas"""
import builtins
import foxglove.Quaternion_pb2
import foxglove.Vector3_pb2
import google.protobuf.descriptor
import google.protobuf.message
import google.protobuf.timestamp_pb2
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class FrameTransform(google.protobuf.message.Message):
    """A transform between two reference frames in 3D space"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TIMESTAMP_FIELD_NUMBER: builtins.int
    PARENT_FRAME_ID_FIELD_NUMBER: builtins.int
    CHILD_FRAME_ID_FIELD_NUMBER: builtins.int
    TRANSLATION_FIELD_NUMBER: builtins.int
    ROTATION_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Timestamp of transform"""
    parent_frame_id: builtins.str
    """Name of the parent frame"""
    child_frame_id: builtins.str
    """Name of the child frame"""
    @property
    def translation(self) -> foxglove.Vector3_pb2.Vector3:
        """Translation component of the transform"""
    @property
    def rotation(self) -> foxglove.Quaternion_pb2.Quaternion:
        """Rotation component of the transform"""
    def __init__(
        self,
        *,
        timestamp: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        parent_frame_id: builtins.str = ...,
        child_frame_id: builtins.str = ...,
        translation: foxglove.Vector3_pb2.Vector3 | None = ...,
        rotation: foxglove.Quaternion_pb2.Quaternion | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["rotation", b"rotation", "timestamp", b"timestamp", "translation", b"translation"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["child_frame_id", b"child_frame_id", "parent_frame_id", b"parent_frame_id", "rotation", b"rotation", "timestamp", b"timestamp", "translation", b"translation"]) -> None: ...

global___FrameTransform = FrameTransform
