// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: caffe2_legacy.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "caffe2_legacy.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace caffe2 {

namespace {

const ::google::protobuf::EnumDescriptor* LegacyPadding_descriptor_ = NULL;

}  // namespace


void protobuf_AssignDesc_caffe2_5flegacy_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AssignDesc_caffe2_5flegacy_2eproto() {
  protobuf_AddDesc_caffe2_5flegacy_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "caffe2_legacy.proto");
  GOOGLE_CHECK(file != NULL);
  LegacyPadding_descriptor_ = file->enum_type(0);
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_caffe2_5flegacy_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
}

}  // namespace

void protobuf_ShutdownFile_caffe2_5flegacy_2eproto() {
}

void protobuf_AddDesc_caffe2_5flegacy_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AddDesc_caffe2_5flegacy_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\023caffe2_legacy.proto\022\006caffe2*J\n\rLegacyP"
    "adding\022\n\n\006NOTSET\020\000\022\t\n\005VALID\020\001\022\010\n\004SAME\020\002\022"
    "\030\n\024CAFFE_LEGACY_POOLING\020\003", 105);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "caffe2_legacy.proto", &protobuf_RegisterTypes);
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_caffe2_5flegacy_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_caffe2_5flegacy_2eproto {
  StaticDescriptorInitializer_caffe2_5flegacy_2eproto() {
    protobuf_AddDesc_caffe2_5flegacy_2eproto();
  }
} static_descriptor_initializer_caffe2_5flegacy_2eproto_;
const ::google::protobuf::EnumDescriptor* LegacyPadding_descriptor() {
  protobuf_AssignDescriptorsOnce();
  return LegacyPadding_descriptor_;
}
bool LegacyPadding_IsValid(int value) {
  switch(value) {
    case 0:
    case 1:
    case 2:
    case 3:
      return true;
    default:
      return false;
  }
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace caffe2

// @@protoc_insertion_point(global_scope)
