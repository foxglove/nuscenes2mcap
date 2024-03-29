"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Generated by https://github.com/foxglove/schemas"""
import builtins
import collections.abc
import foxglove.PackedElementField_pb2
import foxglove.Pose_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import google.protobuf.timestamp_pb2
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class PointCloud(google.protobuf.message.Message):
    """A collection of N-dimensional points, which may contain additional fields with information like normals, intensity, etc."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TIMESTAMP_FIELD_NUMBER: builtins.int
    FRAME_ID_FIELD_NUMBER: builtins.int
    POSE_FIELD_NUMBER: builtins.int
    POINT_STRIDE_FIELD_NUMBER: builtins.int
    FIELDS_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Timestamp of point cloud"""
    frame_id: builtins.str
    """Frame of reference"""
    @property
    def pose(self) -> foxglove.Pose_pb2.Pose:
        """The origin of the point cloud relative to the frame of reference"""
    point_stride: builtins.int
    """Number of bytes between points in the `data`"""
    @property
    def fields(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[foxglove.PackedElementField_pb2.PackedElementField]:
        """Fields in `data`. At least 2 coordinate fields from `x`, `y`, and `z` are required for each point's position; `red`, `green`, `blue`, and `alpha` are optional for customizing each point's color."""
    data: builtins.bytes
    """Point data, interpreted using `fields`"""
    def __init__(
        self,
        *,
        timestamp: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        frame_id: builtins.str = ...,
        pose: foxglove.Pose_pb2.Pose | None = ...,
        point_stride: builtins.int = ...,
        fields: collections.abc.Iterable[foxglove.PackedElementField_pb2.PackedElementField] | None = ...,
        data: builtins.bytes = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["pose", b"pose", "timestamp", b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["data", b"data", "fields", b"fields", "frame_id", b"frame_id", "point_stride", b"point_stride", "pose", b"pose", "timestamp", b"timestamp"]) -> None: ...

global___PointCloud = PointCloud
