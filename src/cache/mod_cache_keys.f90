module mod_cache_keys
  use mod_kinds,  only: i32, i64, MAX_TENSOR_RANK, MAX_NAME_LEN
  use mod_types,  only: MIZU_ABI_VERSION, MIZU_MODEL_FAMILY_UNKNOWN, MIZU_STAGE_NONE, &
                        MIZU_BACKEND_FAMILY_NONE, MIZU_EXEC_ROUTE_NONE, &
                        MIZU_DTYPE_UNKNOWN
  use mod_model_manifest, only: model_manifest

  implicit none

  private
  public :: MAX_CACHE_KEY_LEN, CACHE_KEY_SCHEMA_VERSION
  public :: invalidation_version_fields
  public :: plan_cache_key, weight_cache_key, session_cache_key, multimodal_cache_key
  public :: initialize_invalidation_fields
  public :: build_plan_cache_key, build_weight_cache_key
  public :: build_session_cache_key, build_multimodal_cache_key

  integer(i32), parameter :: MAX_CACHE_KEY_LEN = 512_i32
  integer(i32), parameter :: CACHE_KEY_SCHEMA_VERSION = 1_i32
  integer(i32), parameter :: MAX_SHAPE_SIGNATURE_LEN = 160_i32

  type :: invalidation_version_fields
    integer(i32) :: schema_version  = CACHE_KEY_SCHEMA_VERSION
    integer(i32) :: abi_version     = MIZU_ABI_VERSION
    integer(i32) :: planner_version = 0_i32
    integer(i32) :: pack_version    = 0_i32
    integer(i32) :: backend_version = 0_i32
  end type invalidation_version_fields

  type :: plan_cache_key
    type(invalidation_version_fields) :: versions
    integer(i64)                      :: logical_model_hash = 0_i64
    integer(i64)                      :: projector_revision = 0_i64
    integer(i32)                      :: model_family       = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i32)                      :: stage_kind         = MIZU_STAGE_NONE
    integer(i32)                      :: backend_family     = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                      :: execution_route    = MIZU_EXEC_ROUTE_NONE
    integer(i32)                      :: dtype              = MIZU_DTYPE_UNKNOWN
    integer(i32)                      :: rank               = 0_i32
    integer(i64)                      :: shape(MAX_TENSOR_RANK) = 0_i64
    character(len=MAX_NAME_LEN)       :: device_key         = ""
    character(len=MAX_NAME_LEN)       :: pack_format        = ""
    character(len=MAX_CACHE_KEY_LEN)  :: key_text           = ""
  end type plan_cache_key

  type :: weight_cache_key
    type(invalidation_version_fields) :: versions
    integer(i64)                      :: logical_model_hash = 0_i64
    integer(i64)                      :: projector_revision = 0_i64
    integer(i32)                      :: model_family       = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i32)                      :: backend_family     = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                      :: execution_route    = MIZU_EXEC_ROUTE_NONE
    character(len=MAX_NAME_LEN)       :: device_key         = ""
    character(len=MAX_NAME_LEN)       :: pack_format        = ""
    character(len=MAX_CACHE_KEY_LEN)  :: key_text           = ""
  end type weight_cache_key

  type :: session_cache_key
    type(invalidation_version_fields) :: versions
    integer(i64)                      :: logical_model_hash = 0_i64
    integer(i32)                      :: model_family       = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i32)                      :: backend_family     = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                      :: execution_route    = MIZU_EXEC_ROUTE_NONE
    integer(i64)                      :: max_context_tokens = 0_i64
    integer(i64)                      :: max_decode_tokens  = 0_i64
    character(len=MAX_NAME_LEN)       :: device_key         = ""
    character(len=MAX_CACHE_KEY_LEN)  :: key_text           = ""
  end type session_cache_key

  type :: multimodal_cache_key
    type(invalidation_version_fields) :: versions
    integer(i64)                      :: logical_model_hash = 0_i64
    integer(i64)                      :: projector_revision = 0_i64
    integer(i32)                      :: model_family       = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i32)                      :: modality_kind      = 0_i32
    integer(i32)                      :: dtype              = MIZU_DTYPE_UNKNOWN
    integer(i64)                      :: byte_count         = 0_i64
    character(len=MAX_NAME_LEN)       :: device_key         = ""
    character(len=MAX_NAME_LEN)       :: slot_name          = ""
    character(len=MAX_CACHE_KEY_LEN)  :: key_text           = ""
  end type multimodal_cache_key

contains

  subroutine initialize_invalidation_fields(fields)
    type(invalidation_version_fields), intent(out) :: fields

    fields = invalidation_version_fields()
  end subroutine initialize_invalidation_fields

  subroutine build_plan_cache_key(manifest, device_key, pack_format, stage_kind, backend_family, &
                                  execution_route, dtype, rank, shape, key, versions)
    type(model_manifest), intent(in)                :: manifest
    character(len=*), intent(in)                    :: device_key
    character(len=*), intent(in)                    :: pack_format
    integer(i32), intent(in)                        :: stage_kind
    integer(i32), intent(in)                        :: backend_family
    integer(i32), intent(in)                        :: execution_route
    integer(i32), intent(in)                        :: dtype
    integer(i32), intent(in)                        :: rank
    integer(i64), intent(in)                        :: shape(:)
    type(plan_cache_key), intent(out)               :: key
    type(invalidation_version_fields), intent(in), optional :: versions
    character(len=MAX_SHAPE_SIGNATURE_LEN)          :: shape_text

    call resolve_versions(key%versions, versions)
    key%logical_model_hash = manifest%logical_model_hash
    key%projector_revision = manifest%projector%revision_identity
    key%model_family       = manifest%model_family
    key%stage_kind         = stage_kind
    key%backend_family     = backend_family
    key%execution_route    = execution_route
    key%dtype              = dtype
    key%rank               = max(rank, 0_i32)
    key%shape              = 0_i64
    if (key%rank > 0_i32) then
      key%shape(1:min(key%rank, int(size(shape), kind=i32))) = &
        shape(1:min(key%rank, int(size(shape), kind=i32)))
    end if
    key%device_key         = trim(device_key)
    key%pack_format        = trim(pack_format)
    shape_text = shape_signature_to_text(key%shape, key%rank)

    key%key_text = "plan:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":planner=" // trim(i32_to_text(key%versions%planner_version)) // &
      ":packv=" // trim(i32_to_text(key%versions%pack_version)) // &
      ":backendv=" // trim(i32_to_text(key%versions%backend_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":proj=" // trim(i64_to_text(key%projector_revision)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":stage=" // trim(i32_to_text(key%stage_kind)) // &
      ":backend=" // trim(i32_to_text(key%backend_family)) // &
      ":route=" // trim(i32_to_text(key%execution_route)) // &
      ":dtype=" // trim(i32_to_text(key%dtype)) // &
      ":device=" // trim(key%device_key) // &
      ":pack=" // trim(key%pack_format) // &
      ":shape=" // trim(shape_text)
  end subroutine build_plan_cache_key

  subroutine build_weight_cache_key(manifest, device_key, pack_format, backend_family, &
                                    execution_route, key, versions)
    type(model_manifest), intent(in)                :: manifest
    character(len=*), intent(in)                    :: device_key
    character(len=*), intent(in)                    :: pack_format
    integer(i32), intent(in)                        :: backend_family
    integer(i32), intent(in)                        :: execution_route
    type(weight_cache_key), intent(out)             :: key
    type(invalidation_version_fields), intent(in), optional :: versions

    call resolve_versions(key%versions, versions)
    key%logical_model_hash = manifest%logical_model_hash
    key%projector_revision = manifest%projector%revision_identity
    key%model_family       = manifest%model_family
    key%backend_family     = backend_family
    key%execution_route    = execution_route
    key%device_key         = trim(device_key)
    key%pack_format        = trim(pack_format)

    key%key_text = "weight:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":planner=" // trim(i32_to_text(key%versions%planner_version)) // &
      ":packv=" // trim(i32_to_text(key%versions%pack_version)) // &
      ":backendv=" // trim(i32_to_text(key%versions%backend_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":proj=" // trim(i64_to_text(key%projector_revision)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":backend=" // trim(i32_to_text(key%backend_family)) // &
      ":route=" // trim(i32_to_text(key%execution_route)) // &
      ":device=" // trim(key%device_key) // &
      ":pack=" // trim(key%pack_format)
  end subroutine build_weight_cache_key

  subroutine build_session_cache_key(manifest, device_key, backend_family, execution_route, &
                                     max_context_tokens, max_decode_tokens, key, versions)
    type(model_manifest), intent(in)                :: manifest
    character(len=*), intent(in)                    :: device_key
    integer(i32), intent(in)                        :: backend_family
    integer(i32), intent(in)                        :: execution_route
    integer(i64), intent(in)                        :: max_context_tokens
    integer(i64), intent(in)                        :: max_decode_tokens
    type(session_cache_key), intent(out)            :: key
    type(invalidation_version_fields), intent(in), optional :: versions

    call resolve_versions(key%versions, versions)
    key%logical_model_hash = manifest%logical_model_hash
    key%model_family       = manifest%model_family
    key%backend_family     = backend_family
    key%execution_route    = execution_route
    key%max_context_tokens = max_context_tokens
    key%max_decode_tokens  = max_decode_tokens
    key%device_key         = trim(device_key)

    key%key_text = "session:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":planner=" // trim(i32_to_text(key%versions%planner_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":backend=" // trim(i32_to_text(key%backend_family)) // &
      ":route=" // trim(i32_to_text(key%execution_route)) // &
      ":ctx=" // trim(i64_to_text(key%max_context_tokens)) // &
      ":decode=" // trim(i64_to_text(key%max_decode_tokens)) // &
      ":device=" // trim(key%device_key)
  end subroutine build_session_cache_key

  subroutine build_multimodal_cache_key(manifest, device_key, slot_name, modality_kind, dtype, &
                                        byte_count, key, versions)
    type(model_manifest), intent(in)                :: manifest
    character(len=*), intent(in)                    :: device_key
    character(len=*), intent(in)                    :: slot_name
    integer(i32), intent(in)                        :: modality_kind
    integer(i32), intent(in)                        :: dtype
    integer(i64), intent(in)                        :: byte_count
    type(multimodal_cache_key), intent(out)         :: key
    type(invalidation_version_fields), intent(in), optional :: versions

    call resolve_versions(key%versions, versions)
    key%logical_model_hash = manifest%logical_model_hash
    key%projector_revision = manifest%projector%revision_identity
    key%model_family       = manifest%model_family
    key%modality_kind      = modality_kind
    key%dtype              = dtype
    key%byte_count         = byte_count
    key%device_key         = trim(device_key)
    key%slot_name          = trim(slot_name)

    key%key_text = "mm:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":planner=" // trim(i32_to_text(key%versions%planner_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":proj=" // trim(i64_to_text(key%projector_revision)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":kind=" // trim(i32_to_text(key%modality_kind)) // &
      ":dtype=" // trim(i32_to_text(key%dtype)) // &
      ":bytes=" // trim(i64_to_text(key%byte_count)) // &
      ":device=" // trim(key%device_key) // &
      ":slot=" // trim(key%slot_name)
  end subroutine build_multimodal_cache_key

  subroutine resolve_versions(resolved, provided)
    type(invalidation_version_fields), intent(out) :: resolved
    type(invalidation_version_fields), intent(in), optional :: provided

    call initialize_invalidation_fields(resolved)
    if (present(provided)) then
      resolved = provided
      if (resolved%schema_version <= 0_i32) resolved%schema_version = CACHE_KEY_SCHEMA_VERSION
      if (resolved%abi_version <= 0_i32) resolved%abi_version = MIZU_ABI_VERSION
    end if
  end subroutine resolve_versions

  pure function shape_signature_to_text(shape, rank) result(signature)
    integer(i64), intent(in) :: shape(MAX_TENSOR_RANK)
    integer(i32), intent(in) :: rank
    character(len=MAX_SHAPE_SIGNATURE_LEN) :: signature
    character(len=32) :: dim_text
    integer(i32) :: index
    integer(i32) :: cursor
    integer(i32) :: dim_length
    integer(i32) :: limit

    signature = ""
    if (rank <= 0_i32) then
      signature = "scalar"
      return
    end if

    cursor = 1_i32
    do index = 1_i32, min(rank, MAX_TENSOR_RANK)
      write(dim_text, "(I0)") shape(index)
      dim_length = int(len_trim(dim_text), kind=i32)
      if (index > 1_i32) then
        if (cursor <= len(signature)) then
          signature(cursor:cursor) = "x"
          cursor = cursor + 1_i32
        end if
      end if
      limit = min(cursor + dim_length - 1_i32, int(len(signature), kind=i32))
      if (limit >= cursor) then
        signature(cursor:limit) = dim_text(1:limit - cursor + 1_i32)
        cursor = limit + 1_i32
      end if
    end do
  end function shape_signature_to_text

  pure function i32_to_text(value) result(text)
    integer(i32), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i32_to_text

  pure function i64_to_text(value) result(text)
    integer(i64), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i64_to_text

end module mod_cache_keys
