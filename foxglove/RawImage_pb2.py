# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: foxglove/RawImage.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x66oxglove/RawImage.proto\x12\x08\x66oxglove\x1a\x1fgoogle/protobuf/timestamp.proto\"\x98\x01\n\x08RawImage\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08\x66rame_id\x18\x07 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x07\x12\x0e\n\x06height\x18\x03 \x01(\x07\x12\x10\n\x08\x65ncoding\x18\x04 \x01(\t\x12\x0c\n\x04step\x18\x05 \x01(\x07\x12\x0c\n\x04\x64\x61ta\x18\x06 \x01(\x0c\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'foxglove.RawImage_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RAWIMAGE._serialized_start=71
  _RAWIMAGE._serialized_end=223
# @@protoc_insertion_point(module_scope)
