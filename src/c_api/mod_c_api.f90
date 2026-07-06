module mod_c_api
  use iso_c_binding, only: c_ptr, c_null_ptr, c_associated, c_f_pointer, c_loc, &
                           c_size_t, c_int32_t, c_int64_t, c_char, c_float, &
                           c_null_char, c_sizeof
  use mod_kinds,     only: i8, i32, i64, c_i8, r32, MAX_NAME_LEN, MAX_PATH_LEN
  use mod_status,    only: MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE, &
                           MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_INVALID_STATE, &
                           MIZU_STATUS_BUFFER_TOO_SMALL, MIZU_STATUS_ABI_MISMATCH, &
                           MIZU_STATUS_BUSY, MIZU_STATUS_UNSUPPORTED_MODEL, &
                           MIZU_STATUS_UNSUPPORTED_MODALITY, MIZU_STATUS_NO_VALID_PLAN, &
                           MIZU_STATUS_SESSION_EVICTED
  use mod_types,     only: MIZU_ABI_VERSION, MIZU_OPTIMIZATION_MODE_DISABLED, &
                           MIZU_OPTIMIZATION_MODE_MEASURE_ONLY, &
                           MIZU_OPTIMIZATION_MODE_LEARN_AND_REUSE, &
                           MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_MASK_APPLE_ANE, &
                           MIZU_BACKEND_MASK_APPLE_METAL, MIZU_BACKEND_MASK_CUDA, &
                           MIZU_MODEL_FLAG_NONE, MIZU_SESSION_FLAG_NONE, &
                           MIZU_ATTACH_FLAG_NONE, MIZU_MODEL_FEATURE_MULTIMODAL, &
                           MIZU_MODEL_FEATURE_PROJECTOR, MIZU_MODEL_FAMILY_UNKNOWN, &
                           MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                           MIZU_STAGE_NONE, MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, &
                           MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, MIZU_STAGE_PARK, &
                           MIZU_STAGE_RESUME, MIZU_SELECTION_MODE_NONE, &
                           MIZU_SELECTION_MODE_DIRECT, MIZU_SELECTION_MODE_EXPLORATORY, &
                           MIZU_SELECTION_MODE_REUSE, MIZU_COLD_STATE_UNKNOWN, &
                           MIZU_COLD_STATE_COLD, MIZU_COLD_STATE_WARM, &
                           MIZU_FALLBACK_REASON_NONE, MIZU_OUTPUT_KIND_TOKEN_IDS, &
                           MIZU_STOP_REASON_NONE, MIZU_STOP_REASON_TOKEN_BUDGET, &
                           MIZU_BACKEND_FAMILY_NONE, &
                           MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                           MIZU_EXEC_ROUTE_NONE, MIZU_EXEC_ROUTE_ANE, &
                           MIZU_EXEC_ROUTE_METAL, MIZU_EXEC_ROUTE_CUDA, &
                           MIZU_CACHE_FLAG_NONE, MIZU_CACHE_FLAG_WEIGHT_HIT, &
                           MIZU_CACHE_FLAG_PLAN_HIT, MIZU_CACHE_FLAG_SESSION_HIT, &
                           MIZU_CACHE_FLAG_MM_HIT, MIZU_CACHE_FLAG_WINNER_REUSED, &
                           MIZU_MODALITY_KIND_IMAGE, MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS, &
                           MIZU_STORAGE_KIND_ENCODED_BYTES, MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS, &
                           MIZU_DTYPE_U8, MIZU_DTYPE_I32, MIZU_DTYPE_F16, MIZU_DTYPE_BF16, &
                           MIZU_DTYPE_F32, &
                           MIZU_LIFETIME_POLICY_COPY, MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL, &
                           SOURCE_FORMAT_MIZU_IMPORT_BUNDLE, runtime_handle, model_handle, &
                           session_handle, runtime_config, model_open_config, session_config, &
                           model_info, session_info, execution_report, runtime_state, &
                           model_state, session_state, import_tensor_state, &
                           MAX_RECENT_OUTPUT_TOKENS, MAX_LIVE_CONTEXT_BYTES
  use mod_runtime,   only: initialize_runtime_state, reset_runtime_state, &
                           initialize_model_state, reset_model_state, register_model, &
                           unregister_model, register_session, unregister_session, &
                           validate_runtime_destroy, validate_model_close, &
                           set_runtime_error
  use mod_workspace, only: reserve_workspace_bytes, release_workspace_bytes
  use mod_session,   only: initialize_session_state, reset_session_state, &
                           validate_prefill, validate_decode, validate_park, validate_resume, &
                           stage_tokens, stage_modal_input, clear_pending_inputs, &
                           complete_prefill, complete_decode, park_session_state, &
                           resume_session_state, evict_parked_session, &
                           build_session_info, validate_read_output, store_live_context_record, &
                           update_live_context_record, offload_live_context_record
  use mod_optimization_store, only: runtime_optimization_store, &
                                     initialize_runtime_optimization_store, &
                                     reset_runtime_optimization_store, &
                                     record_execution_sample, lookup_winner_candidate, &
                                     lookup_optimization_entry_stats, &
                                     invalidate_stale_optimization_candidates, &
                                     load_runtime_optimization_store, &
                                     save_runtime_optimization_store
  use mod_backend_registry, only: runtime_backend_registry, initialize_runtime_backend_registry, &
                                  probe_runtime_backend_registry, apply_backend_registry_to_runtime
  use mod_backend_probe_support, only: read_boolean_env_override
  use mod_backend_contract, only: plan_request, planner_result, initialize_plan_request, &
                                  planner_result_is_success, OP_FAMILY_NONE, OP_FAMILY_PROJECTOR, &
                                  OP_FAMILY_PREFILL, OP_FAMILY_DECODE
  use mod_model_manifest, only: model_manifest, populate_model_info_from_manifest, &
                                manifest_tensor_count, manifest_modality_count, &
                                hash_text64
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_apple_planner,  only: APPLE_ARTIFACT_PAYLOAD_LEN, plan_apple_stage, &
                                build_apple_artifact_payload_text
  use mod_apple_executor, only: execute_apple_projector, execute_apple_prefill, execute_apple_decode, &
                                apple_context_bytes_are_valid, extract_apple_context_lineage
  use mod_cuda_planner,   only: CUDA_ARTIFACT_PAYLOAD_LEN, plan_cuda_stage, &
                                build_cuda_artifact_payload_text
  use mod_cuda_executor,  only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode, &
                                cuda_context_bytes_are_valid, extract_cuda_context_lineage
  use mod_cache_keys,     only: MAX_CACHE_KEY_LEN, invalidation_version_fields, &
                                plan_cache_key, weight_cache_key, &
                                session_cache_key, multimodal_cache_key, build_plan_cache_key, &
                                build_weight_cache_key, build_session_cache_key, &
                                build_multimodal_cache_key
  use mod_cache_store,    only: artifact_metadata_record, runtime_cache_bundle, &
                                initialize_runtime_cache_bundle, reset_runtime_cache_bundle, &
                                touch_weight_cache_key, touch_plan_cache_key, &
                                touch_session_cache_key, touch_multimodal_cache_key, &
                                record_weight_artifact_metadata, record_plan_artifact_metadata, &
                                record_session_artifact_metadata, record_multimodal_artifact_metadata, &
                                lookup_session_artifact_metadata, load_runtime_cache_bundle, &
                                save_runtime_cache_bundle

  implicit none

  private
  public :: mizu_get_abi_version
  public :: mizu_runtime_create, mizu_runtime_destroy, mizu_runtime_copy_last_error
  public :: mizu_model_open, mizu_model_close, mizu_model_get_info
  public :: mizu_model_get_last_report
  public :: mizu_session_open, mizu_session_close, mizu_session_park
  public :: mizu_session_resume, mizu_session_get_info
  public :: mizu_session_attach_tokens, mizu_session_attach_modal_input
  public :: mizu_session_clear_pending_inputs, mizu_session_prefill
  public :: mizu_session_decode_step, mizu_session_read_output
  public :: mizu_session_get_last_report

  integer(i64), parameter :: INITIAL_REGISTRY_CAPACITY = 8_i64
  integer(i64), parameter :: MAX_RETIRED_HANDLE_BOXES_PER_KIND = 4096_i64
  integer(i32), parameter :: MAX_IMPORT_STAGE_PACK_DISPATCH = 4_i32
  integer(i32), parameter :: MAX_CUDA_PACK_PAGE_WORDS = 8_i32
  integer(i32), parameter :: MAX_CUDA_PACK_TILE_BYTES = 32_i32
  integer(i64), parameter :: IMPORT_STAGE_SPAN_SAMPLE_BYTES = 64_i64

  type, bind(c) :: c_runtime_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    type(c_ptr)        :: cache_root_z
    integer(c_int32_t) :: optimization_mode
    integer(c_int32_t) :: exploration_budget
    integer(c_int64_t) :: runtime_flags
  end type c_runtime_config

  type, bind(c) :: c_model_open_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    type(c_ptr)        :: model_root_z
    integer(c_int64_t) :: allowed_backend_mask
    integer(c_int64_t) :: model_flags
  end type c_model_open_config

  type, bind(c) :: c_session_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    integer(c_int64_t) :: max_context_tokens
    integer(c_int64_t) :: max_decode_tokens
    integer(c_int32_t) :: sampler_kind
    integer(c_int64_t) :: seed
    real(c_float)      :: temperature
    integer(c_int32_t) :: top_k
    real(c_float)      :: top_p
    integer(c_int64_t) :: session_flags
  end type c_session_config

  type, bind(c) :: c_model_info
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: model_family
    integer(c_int64_t) :: allowed_backend_mask
    integer(c_int64_t) :: model_features
    integer(c_int32_t) :: projector_slot_count
    integer(c_int32_t) :: reserved_u32
  end type c_model_info

  type, bind(c) :: c_session_info
    integer(c_size_t)  :: struct_size
    integer(c_int64_t) :: session_state_flags
    integer(c_int64_t) :: kv_token_count
    integer(c_int64_t) :: staged_token_count
    integer(c_int32_t) :: staged_modal_count
    integer(c_int32_t) :: reserved_u32
  end type c_session_info

  type, bind(c) :: c_modal_input_desc
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: slot_name_z
    integer(c_int32_t) :: placeholder_ordinal
    integer(c_int32_t) :: modality_kind
    integer(c_int32_t) :: storage_kind
    integer(c_int32_t) :: dtype
    integer(c_int32_t) :: rank
    type(c_ptr)        :: shape
    type(c_ptr)        :: data
    integer(c_size_t)  :: byte_count
    integer(c_int32_t) :: lifetime_policy
    integer(c_int64_t) :: input_flags
  end type c_modal_input_desc

  type, bind(c) :: c_decode_options
    integer(c_size_t)  :: struct_size
    integer(c_int64_t) :: token_budget
    integer(c_int64_t) :: stop_flags
    integer(c_int64_t) :: decode_flags
  end type c_decode_options

  type, bind(c) :: c_decode_result
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: token_buffer
    integer(c_size_t)  :: token_capacity
    integer(c_size_t)  :: token_count
    integer(c_int32_t) :: stop_reason
    integer(c_int64_t) :: result_flags
  end type c_decode_result

  type, bind(c) :: c_output_buffer
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: output_kind
    type(c_ptr)        :: data
    integer(c_size_t)  :: byte_capacity
    integer(c_size_t)  :: bytes_written
    integer(c_int64_t) :: output_flags
  end type c_output_buffer

  type, bind(c) :: c_execution_report
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: stage_kind
    integer(c_int32_t) :: backend_family
    integer(c_int32_t) :: execution_route
    integer(c_int64_t) :: plan_id
    integer(c_int32_t) :: selection_mode
    integer(c_int32_t) :: cold_state
    integer(c_int32_t) :: fallback_reason
    integer(c_int64_t) :: cache_flags
    integer(c_int64_t) :: elapsed_us
  end type c_execution_report

  type, bind(c) :: c_report_buffer
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: reports
    integer(c_size_t)  :: report_capacity
    integer(c_size_t)  :: report_count
  end type c_report_buffer

  type, bind(c) :: runtime_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type runtime_box

  type, bind(c) :: model_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type model_box

  type, bind(c) :: session_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type session_box

  type(runtime_state), allocatable, target, save :: runtime_registry(:)
  type(runtime_cache_bundle), allocatable, target, save :: runtime_cache_registry(:)
  type(runtime_optimization_store), allocatable, target, save :: runtime_optimization_registry(:)
  type(model_state), allocatable, target, save   :: model_registry(:)
  type(session_state), allocatable, target, save :: session_registry(:)
  type(c_ptr), allocatable, save :: runtime_handle_ptrs(:)
  type(c_ptr), allocatable, save :: model_handle_ptrs(:)
  type(c_ptr), allocatable, save :: session_handle_ptrs(:)

  logical, allocatable, save :: runtime_used(:)
  logical, allocatable, save :: model_used(:)
  logical, allocatable, save :: session_used(:)
  integer(i64), save :: retired_runtime_box_count = 0_i64
  integer(i64), save :: retired_model_box_count = 0_i64
  integer(i64), save :: retired_session_box_count = 0_i64

  interface
    function c_strlen(str) bind(c, name="strlen") result(length)
      import c_ptr, c_size_t
      type(c_ptr), value :: str
      integer(c_size_t)  :: length
    end function c_strlen
  end interface

contains

  integer(c_int32_t) function mizu_get_abi_version() bind(c, name="mizu_get_abi_version")
    mizu_get_abi_version = int(MIZU_ABI_VERSION, kind=c_int32_t)
  end function mizu_get_abi_version

  integer(c_int32_t) function mizu_runtime_create(config_ptr, out_runtime_ptr) &
      bind(c, name="mizu_runtime_create")
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_runtime_ptr
    type(c_runtime_config), pointer :: c_config
    type(runtime_box), pointer      :: box
    type(runtime_config)            :: config
    type(runtime_backend_registry)  :: backend_registry
    integer(i32)                    :: status_code
    integer(i64)                    :: slot_id

    out_runtime_ptr = c_null_ptr

    if (.not. c_associated(config_ptr)) then
      mizu_runtime_create = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      mizu_runtime_create = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_input_struct_size(c_config%struct_size, c_sizeof(c_config))
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_create = int(status_code, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      mizu_runtime_create = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version        = int(c_config%abi_version, kind=i32)
    config%optimization_mode  = int(c_config%optimization_mode, kind=i32)
    config%exploration_budget = int(c_config%exploration_budget, kind=i32)
    config%runtime_flags      = int(c_config%runtime_flags, kind=i64)
    call copy_c_string_ptr_to_fortran(c_config%cache_root_z, config%cache_root)

    status_code = require_retired_handle_capacity(retired_runtime_box_count)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_create = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = acquire_runtime_slot()
    call initialize_runtime_state(runtime_registry(slot_id), config)
    call initialize_runtime_cache_bundle(runtime_cache_registry(slot_id))
    call initialize_runtime_optimization_store(runtime_optimization_registry(slot_id))
    call initialize_runtime_backend_registry(backend_registry)
    call probe_runtime_backend_registry(backend_registry)
    call apply_backend_registry_to_runtime(backend_registry, runtime_registry(slot_id))
    runtime_registry(slot_id)%handle%value = slot_id
    call hydrate_runtime_cache_state(runtime_registry(slot_id), runtime_cache_registry(slot_id))
    call hydrate_runtime_optimization_state(runtime_registry(slot_id), runtime_optimization_registry(slot_id))

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    runtime_handle_ptrs(slot_id) = c_loc(box)
    out_runtime_ptr = runtime_handle_ptrs(slot_id)

    mizu_runtime_create = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_create

  integer(c_int32_t) function mizu_runtime_destroy(runtime_ptr) bind(c, name="mizu_runtime_destroy")
    type(c_ptr), value :: runtime_ptr
    type(runtime_box), pointer  :: box
    type(runtime_state), pointer :: runtime
    integer(i32) :: status_code
    integer(i64) :: slot_id

    if (.not. c_associated(runtime_ptr)) then
      mizu_runtime_destroy = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_destroy = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_runtime_destroy(runtime)
    if (status_code /= MIZU_STATUS_OK) then
      if (status_code == MIZU_STATUS_BUSY) then
        call set_runtime_error(runtime, status_code, "runtime cannot destroy while models are live")
      else
        call set_runtime_error(runtime, status_code, "runtime cannot destroy in current state")
      end if
      mizu_runtime_destroy = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = int(box%id, kind=i64)
    call persist_runtime_cache_state(runtime, runtime_cache_registry(slot_id))
    call persist_runtime_optimization_state(runtime, runtime_optimization_registry(slot_id))
    call reset_runtime_state(runtime)
    call reset_runtime_cache_bundle(runtime_cache_registry(slot_id))
    call reset_runtime_optimization_store(runtime_optimization_registry(slot_id))
    call retire_runtime_box(slot_id, box)

    mizu_runtime_destroy = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_destroy

  integer(c_int32_t) function mizu_runtime_copy_last_error(runtime_ptr, buffer_ptr, capacity, &
                                                           out_required_ptr) &
      bind(c, name="mizu_runtime_copy_last_error")
    type(c_ptr), value         :: runtime_ptr
    type(c_ptr), value         :: buffer_ptr
    integer(c_size_t), value   :: capacity
    type(c_ptr), value         :: out_required_ptr
    type(runtime_box), pointer  :: box
    type(runtime_state), pointer :: runtime
    integer(i32)                :: status_code
    integer(i64)                :: required_len

    call resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_copy_last_error = int(status_code, kind=c_int32_t)
      return
    end if

    required_len = int(len_trim(runtime%last_error_message), kind=i64) + 1_i64
    call write_size_t_pointer(out_required_ptr, required_len)
    call copy_fortran_string_to_c(runtime%last_error_message, buffer_ptr, capacity)

    mizu_runtime_copy_last_error = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_copy_last_error

  integer(c_int32_t) function mizu_model_open(runtime_ptr, config_ptr, out_model_ptr) &
      bind(c, name="mizu_model_open")
    type(c_ptr), value :: runtime_ptr
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_model_ptr
    type(runtime_box), pointer      :: runtime_box_ptr
    type(runtime_state), pointer    :: runtime
    type(c_model_open_config), pointer :: c_config
    type(model_box), pointer        :: box
    type(model_open_config)         :: config
    type(model_manifest)            :: manifest
    type(model_info)                :: info
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64)                    :: slot_id
    integer(i64)                    :: load_cache_flags
    integer(i64)                    :: model_plan_id
    integer(i64)                    :: load_elapsed_us
    integer(i32)                    :: selection_mode
    integer(i32)                    :: report_backend_family
    integer(i32)                    :: report_route
    integer(i32)                    :: status_code
    integer(i64)                    :: stage_started_us

    out_model_ptr = c_null_ptr

    call resolve_runtime_handle(runtime_ptr, runtime_box_ptr, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(config_ptr)) then
      call set_runtime_error(runtime, MIZU_STATUS_INVALID_ARGUMENT, "model config pointer is null")
      mizu_model_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      call set_runtime_error(runtime, MIZU_STATUS_INVALID_ARGUMENT, "model config pointer is invalid")
      mizu_model_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_input_struct_size(c_config%struct_size, c_sizeof(c_config))
    if (status_code /= MIZU_STATUS_OK) then
      call set_runtime_error(runtime, status_code, "model config struct_size is too small")
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      call set_runtime_error(runtime, MIZU_STATUS_ABI_MISMATCH, "model config ABI version mismatch")
      mizu_model_open = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version         = int(c_config%abi_version, kind=i32)
    config%allowed_backend_mask = int(c_config%allowed_backend_mask, kind=i64)
    config%model_flags         = int(c_config%model_flags, kind=i64)
    call copy_c_string_ptr_to_fortran(c_config%model_root_z, config%model_root)

    stage_started_us = monotonic_timestamp_us()
    status_code = build_model_info(config, runtime%detected_backend_mask, manifest, info)
    if (status_code /= MIZU_STATUS_OK) then
      if (status_code == MIZU_STATUS_NO_VALID_PLAN) then
        call set_runtime_error(runtime, status_code, "no requested backend is available on this runtime")
      else
        call set_runtime_error(runtime, status_code, "model manifest load failed")
      end if
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = require_retired_handle_capacity(retired_model_box_count)
    if (status_code /= MIZU_STATUS_OK) then
      call set_runtime_error(runtime, status_code, "model handle arena is exhausted")
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = acquire_model_slot()
    call initialize_model_state(model_registry(slot_id), config, info)
    model_registry(slot_id)%handle%value = slot_id
    model_registry(slot_id)%runtime_owner%value = runtime%handle%value
    model_registry(slot_id)%source_format = manifest%provenance%source_format
    model_registry(slot_id)%logical_model_hash = manifest%logical_model_hash
    model_registry(slot_id)%projector_revision = manifest%projector%revision_identity
    model_registry(slot_id)%tensor_count = manifest_tensor_count(manifest)
    model_registry(slot_id)%modality_count = manifest_modality_count(manifest)
    model_registry(slot_id)%source_model_id = manifest%provenance%source_model_id
    call copy_model_import_snapshot(manifest, model_registry(slot_id))
    runtime_cache => runtime_cache_registry(runtime%handle%value)
    optimization_store => runtime_optimization_registry(runtime%handle%value)
    load_elapsed_us = elapsed_since_us(stage_started_us)
    load_cache_flags = resolve_weight_cache_flags(runtime, runtime_cache, optimization_store, &
      manifest, model_registry(slot_id), info%allowed_backend_mask, load_elapsed_us, &
      model_plan_id, selection_mode, &
      report_backend_family, report_route)
    model_registry(slot_id)%last_report = make_stage_report(MIZU_STAGE_MODEL_LOAD, report_backend_family, &
      report_route, MIZU_FALLBACK_REASON_NONE, selection_mode, &
      resolve_stage_cold_state(MIZU_COLD_STATE_COLD, selection_mode, load_cache_flags), &
      load_cache_flags, model_plan_id, load_elapsed_us)

    call register_model(runtime)

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    model_handle_ptrs(slot_id) = c_loc(box)
    out_model_ptr = model_handle_ptrs(slot_id)

    mizu_model_open = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_open

  integer(c_int32_t) function mizu_model_close(model_ptr) bind(c, name="mizu_model_close")
    type(c_ptr), value :: model_ptr
    type(model_box), pointer   :: box
    type(model_state), pointer :: model
    type(runtime_state), pointer :: runtime
    integer(i32) :: status_code
    integer(i64) :: runtime_id
    integer(i64) :: slot_id

    if (.not. c_associated(model_ptr)) then
      mizu_model_close = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_close = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_model_close(model)
    if (status_code /= MIZU_STATUS_OK) then
      if (status_code == MIZU_STATUS_BUSY) then
        call set_model_owner_runtime_error(model, status_code, "model cannot close while sessions are live")
      else
        call set_model_owner_runtime_error(model, status_code, "model cannot close in current state")
      end if
      mizu_model_close = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = int(box%id, kind=i64)
    runtime_id = model%runtime_owner%value
    if (is_runtime_slot_valid(runtime_id)) then
      runtime => runtime_registry(runtime_id)
      call unregister_model(runtime)
    end if

    call reset_model_state(model)
    call retire_model_box(slot_id, box)

    mizu_model_close = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_close

  integer(c_int32_t) function mizu_model_get_info(model_ptr, out_info_ptr) bind(c, name="mizu_model_get_info")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: out_info_ptr
    type(model_box), pointer   :: box
    type(model_state), pointer :: model
    type(c_model_info), pointer :: c_info
    integer(i32) :: status_code

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_info_ptr)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "model info output pointer is null")
      mizu_model_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_info_ptr, c_info)
    if (.not. associated(c_info)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "model info output pointer is invalid")
      mizu_model_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_output_struct_size(c_info%struct_size, c_sizeof(c_info))
    if (status_code /= MIZU_STATUS_OK) then
      call set_model_owner_runtime_error(model, status_code, "model info output struct_size is too small")
      mizu_model_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    c_info%model_family         = int(model%info%model_family, kind=c_int32_t)
    c_info%allowed_backend_mask = int(model%info%allowed_backend_mask, kind=c_int64_t)
    c_info%model_features       = int(model%info%model_features, kind=c_int64_t)
    c_info%projector_slot_count = int(model%info%projector_slot_count, kind=c_int32_t)
    c_info%reserved_u32         = 0_c_int32_t

    mizu_model_get_info = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_get_info

  integer(c_int32_t) function mizu_model_get_last_report(model_ptr, out_report_ptr) &
      bind(c, name="mizu_model_get_last_report")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: out_report_ptr
    type(model_box), pointer    :: box
    type(model_state), pointer  :: model
    type(c_execution_report), pointer :: c_report
    integer(i32) :: status_code

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_report_ptr)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "model report output pointer is null")
      mizu_model_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_report_ptr, c_report)
    if (.not. associated(c_report)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "model report output pointer is invalid")
      mizu_model_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_output_struct_size(c_report%struct_size, c_sizeof(c_report))
    if (status_code /= MIZU_STATUS_OK) then
      call set_model_owner_runtime_error(model, status_code, "model report output struct_size is too small")
      mizu_model_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    call copy_internal_report_to_c(model%last_report, c_report)
    mizu_model_get_last_report = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_get_last_report

  integer(c_int32_t) function mizu_session_open(model_ptr, config_ptr, out_session_ptr) &
      bind(c, name="mizu_session_open")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_session_ptr
    type(model_box), pointer     :: model_box_ptr
    type(model_state), pointer   :: model
    type(c_session_config), pointer :: c_config
    type(session_box), pointer   :: box
    type(session_config)         :: config
    integer(i64)                 :: slot_id
    integer(i32)                 :: status_code

    out_session_ptr = c_null_ptr

    call resolve_model_handle(model_ptr, model_box_ptr, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(config_ptr)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "session config pointer is null")
      mizu_session_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_INVALID_ARGUMENT, "session config pointer is invalid")
      mizu_session_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_input_struct_size(c_config%struct_size, c_sizeof(c_config))
    if (status_code /= MIZU_STATUS_OK) then
      call set_model_owner_runtime_error(model, status_code, "session config struct_size is too small")
      mizu_session_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      call set_model_owner_runtime_error(model, MIZU_STATUS_ABI_MISMATCH, "session config ABI version mismatch")
      mizu_session_open = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version        = int(c_config%abi_version, kind=i32)
    config%max_context_tokens = int(c_config%max_context_tokens, kind=i64)
    config%max_decode_tokens  = int(c_config%max_decode_tokens, kind=i64)
    config%sampler_kind       = int(c_config%sampler_kind, kind=i32)
    config%seed               = int(c_config%seed, kind=i64)
    config%temperature        = real(c_config%temperature, kind=r32)
    config%top_k              = int(c_config%top_k, kind=i32)
    config%top_p              = real(c_config%top_p, kind=r32)
    config%session_flags      = int(c_config%session_flags, kind=i64)

    status_code = require_retired_handle_capacity(retired_session_box_count)
    if (status_code /= MIZU_STATUS_OK) then
      call set_model_owner_runtime_error(model, status_code, "session handle arena is exhausted")
      mizu_session_open = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = acquire_session_slot()
    call initialize_session_state(session_registry(slot_id), config)
    session_registry(slot_id)%handle%value      = slot_id
    session_registry(slot_id)%model_owner%value = model%handle%value

    call register_session(model)

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    session_handle_ptrs(slot_id) = c_loc(box)
    out_session_ptr = session_handle_ptrs(slot_id)

    mizu_session_open = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_open

  integer(c_int32_t) function mizu_session_close(session_ptr) bind(c, name="mizu_session_close")
    type(c_ptr), value :: session_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    type(model_state), pointer   :: model
    integer(i32) :: status_code
    integer(i64) :: model_id
    integer(i64) :: slot_id

    if (.not. c_associated(session_ptr)) then
      mizu_session_close = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_close = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = int(box%id, kind=i64)
    model_id = session%model_owner%value
    if (is_model_slot_valid(model_id)) then
      model => model_registry(model_id)
      call unregister_session(model)
    end if

    call reset_session_state(session)
    call retire_session_box(slot_id, box)

    mizu_session_close = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_close

  integer(c_int32_t) function mizu_session_park(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_park")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(model_state), pointer    :: model
    type(runtime_state), pointer  :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64) :: cache_flags
    integer(i64) :: stage_plan_id
    integer(i64) :: stage_elapsed_us
    integer(i64) :: stage_started_us
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    logical      :: checkpoint_offloaded
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_report_buffer_error(session, out_reports_ptr, status_code)
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_park(session)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session cannot park in current state")
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    call park_session_state(session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call persist_session_checkpoint(runtime, runtime_cache, model, session, checkpoint_offloaded)
    if (checkpoint_offloaded) call offload_live_context_record(session)

    stage_elapsed_us = elapsed_since_us(stage_started_us)
    call resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
      stage_elapsed_us, stage_plan_id, selection_mode, report_backend_family, report_route, cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_PARK, report_backend_family, report_route, &
      MIZU_FALLBACK_REASON_NONE, selection_mode, &
      resolve_stage_cold_state(MIZU_COLD_STATE_WARM, selection_mode, cache_flags), &
      cache_flags, stage_plan_id, stage_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())
    if (checkpoint_offloaded .and. force_session_eviction_requested()) call evict_parked_session(session)

    mizu_session_park = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_park

  integer(c_int32_t) function mizu_session_resume(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_resume")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(model_state), pointer    :: model
    type(runtime_state), pointer  :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64) :: cache_flags
    integer(i64) :: stage_plan_id
    integer(i64) :: stage_elapsed_us
    integer(i64) :: stage_started_us
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    logical      :: restored_ok
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_report_buffer_error(session, out_reports_ptr, status_code)
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_resume(session)
    if (status_code /= MIZU_STATUS_OK) then
      if (status_code == MIZU_STATUS_SESSION_EVICTED) then
        call set_runtime_error(runtime, status_code, "session parked state was evicted")
      else
        call set_runtime_error(runtime, status_code, "session cannot resume in current state")
      end if
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    restored_ok = .true.
    if (session%live_context_byte_count > 0_i32 .and. .not. session%has_resident_live_context) then
      call restore_session_checkpoint(runtime%config%cache_root, runtime_cache, model, session, restored_ok)
      if (.not. restored_ok) then
        call set_runtime_error(runtime, MIZU_STATUS_INVALID_STATE, "session checkpoint restore failed")
        mizu_session_resume = int(MIZU_STATUS_INVALID_STATE, kind=c_int32_t)
        return
      end if
    end if
    call resume_session_state(session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    stage_elapsed_us = elapsed_since_us(stage_started_us)
    call resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
      stage_elapsed_us, stage_plan_id, selection_mode, report_backend_family, report_route, cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_RESUME, report_backend_family, report_route, &
      MIZU_FALLBACK_REASON_NONE, selection_mode, &
      resolve_stage_cold_state(MIZU_COLD_STATE_WARM, selection_mode, cache_flags), &
      cache_flags, stage_plan_id, stage_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())

    mizu_session_resume = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_resume

  integer(c_int32_t) function mizu_session_get_info(session_ptr, out_info_ptr) &
      bind(c, name="mizu_session_get_info")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_info_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(c_session_info), pointer :: c_info
    type(session_info)            :: info
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_info_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "session info output pointer is null")
      mizu_session_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_info_ptr, c_info)
    if (.not. associated(c_info)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "session info output pointer is invalid")
      mizu_session_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_output_struct_size(c_info%struct_size, c_sizeof(c_info))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session info output struct_size is too small")
      mizu_session_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    info = build_session_info(session)
    c_info%session_state_flags = int(info%session_state_flags, kind=c_int64_t)
    c_info%kv_token_count      = int(info%kv_token_count, kind=c_int64_t)
    c_info%staged_token_count  = int(info%staged_token_count, kind=c_int64_t)
    c_info%staged_modal_count  = int(info%staged_modal_count, kind=c_int32_t)
    c_info%reserved_u32        = 0_c_int32_t

    mizu_session_get_info = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_get_info

  integer(c_int32_t) function mizu_session_attach_tokens(session_ptr, tokens_ptr, token_count, &
                                                         attach_flags) &
      bind(c, name="mizu_session_attach_tokens")
    type(c_ptr), value       :: session_ptr
    type(c_ptr), value       :: tokens_ptr
    integer(c_size_t), value :: token_count
    integer(c_int32_t), value :: attach_flags
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    integer(c_int32_t), pointer  :: token_values(:)
    integer(i32) :: status_code

    if (attach_flags /= int(MIZU_ATTACH_FLAG_NONE, kind=c_int32_t)) then
      ! Reserved for future policies.
    end if

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_attach_tokens = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(tokens_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "token input pointer is null")
      mizu_session_attach_tokens = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (token_count > int(huge(0), kind=c_size_t)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
                                           "token count exceeds supported interop range")
      mizu_session_attach_tokens = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(tokens_ptr, token_values, [int(token_count)])
    if (.not. associated(token_values)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "token input pointer is invalid")
      mizu_session_attach_tokens = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call stage_tokens(session, int(token_count, kind=i64), status_code, int(token_values, kind=i32))
    if (status_code == MIZU_STATUS_INVALID_STATE) then
      call set_session_owner_runtime_error(session, status_code, "session cannot attach tokens in current state")
    else if (status_code == MIZU_STATUS_INVALID_ARGUMENT) then
      call set_session_owner_runtime_error(session, status_code, "token count must be positive")
    end if
    mizu_session_attach_tokens = int(status_code, kind=c_int32_t)
  end function mizu_session_attach_tokens

  integer(c_int32_t) function mizu_session_attach_modal_input(session_ptr, input_ptr) &
      bind(c, name="mizu_session_attach_modal_input")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: input_ptr
    type(session_box), pointer        :: box
    type(session_state), pointer      :: session
    type(c_modal_input_desc), pointer :: input
    integer(c_i8), pointer            :: modal_bytes(:)
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(input_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
                                           "modal input descriptor pointer is null")
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(input_ptr, input)
    if (.not. associated(input)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
                                           "modal input descriptor pointer is invalid")
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_input_struct_size(input%struct_size, c_sizeof(input))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "modal input descriptor struct_size is too small")
      mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_modal_input_descriptor_c(input)
    if (status_code /= MIZU_STATUS_OK) then
      if (status_code == MIZU_STATUS_UNSUPPORTED_MODALITY) then
        call set_session_owner_runtime_error(session, status_code, &
                                             "modal input modality or storage is unsupported")
      else
        call set_session_owner_runtime_error(session, status_code, "modal input descriptor is invalid")
      end if
      mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
      return
    end if

    if (input%byte_count > int(huge(0), kind=c_size_t)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
                                           "modal input byte count exceeds supported interop range")
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(input%data) .and. input%byte_count > 0_c_size_t) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "modal input data pointer is null")
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (input%byte_count > 0_c_size_t) then
      call c_f_pointer(input%data, modal_bytes, [int(input%byte_count)])
      if (.not. associated(modal_bytes)) then
        call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
                                             "modal input data pointer is invalid")
        mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
        return
      end if

      call stage_modal_input(session, status_code, int(input%byte_count, kind=i64), &
        int(input%modality_kind, kind=i32), int(input%dtype, kind=i32), &
        copy_c_string_ptr(input%slot_name_z, "image"), int(modal_bytes, kind=i8))
    else
      call stage_modal_input(session, status_code, int(input%byte_count, kind=i64), &
        int(input%modality_kind, kind=i32), int(input%dtype, kind=i32), &
        copy_c_string_ptr(input%slot_name_z, "image"))
    end if
    if (status_code == MIZU_STATUS_INVALID_STATE) then
      call set_session_owner_runtime_error(session, status_code, "session cannot attach modal input in current state")
    end if
    mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
  end function mizu_session_attach_modal_input

  integer(c_int32_t) function mizu_session_clear_pending_inputs(session_ptr) &
      bind(c, name="mizu_session_clear_pending_inputs")
    type(c_ptr), value :: session_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_clear_pending_inputs = int(status_code, kind=c_int32_t)
      return
    end if

    call clear_pending_inputs(session, status_code)
    mizu_session_clear_pending_inputs = int(status_code, kind=c_int32_t)
  end function mizu_session_clear_pending_inputs

  integer(c_int32_t) function mizu_session_prefill(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_prefill")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    type(model_state), pointer   :: model
    type(runtime_state), pointer :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i32) :: status_code
    integer(i64) :: required_reports
    integer(i64) :: kv_before
    integer(i64) :: staged_tokens_before
    integer(i64) :: staged_token_hash_before
    integer(i64) :: staged_modal_byte_count_before
    integer(i64) :: staged_modal_hash_before
    integer(i32) :: staged_modal_before
    integer(i32) :: staged_modal_kind_before
    integer(i32) :: staged_modal_dtype_before
    integer(i64) :: projector_cache_flags
    integer(i64) :: prefill_cache_flags
    integer(i64) :: projector_plan_id
    integer(i64) :: prefill_plan_id
    integer(i64) :: projector_elapsed_us
    integer(i64) :: prefill_elapsed_us
    integer(i64) :: stage_started_us
    integer(i64) :: projector_embedding_count
    integer(i64) :: consumed_token_count
    integer(i32) :: prefill_cold_state
    integer(i32) :: projector_selection_mode
    integer(i32) :: prefill_selection_mode
    integer(i32) :: projector_backend_family
    integer(i32) :: projector_route
    integer(i32) :: projector_fallback_reason
    integer(i32) :: projector_placeholder_count
    integer(i32) :: prefill_context_byte_count
    integer(i8)  :: prefill_context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64) :: prefill_context_artifact_hash
    integer(i32) :: prefill_backend_family
    integer(i32) :: prefill_route
    integer(i32) :: prefill_fallback_reason
    character(len=MAX_PATH_LEN) :: staged_modal_slot_name_before
    character(len=MAX_CACHE_KEY_LEN) :: projector_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: projector_candidate_key_text
    character(len=MAX_CACHE_KEY_LEN) :: prefill_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: prefill_candidate_key_text
    logical      :: has_modal_inputs
    logical      :: needs_projector_stage
    logical      :: projector_workspace_reserved
    logical      :: prefill_workspace_reserved
    type(execution_report) :: projector_report
    type(artifact_metadata_record) :: projector_artifact_metadata
    type(artifact_metadata_record) :: prefill_artifact_metadata

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    has_modal_inputs = (session%staged_modal_count > 0_i32)
    kv_before = session%kv_token_count
    staged_tokens_before = session%staged_token_count
    staged_token_hash_before = session%staged_token_hash
    staged_modal_before = session%staged_modal_count
    staged_modal_byte_count_before = session%staged_modal_byte_count
    staged_modal_hash_before = session%staged_modal_hash
    staged_modal_kind_before = session%staged_modal_kind
    staged_modal_dtype_before = session%staged_modal_dtype
    staged_modal_slot_name_before = session%staged_modal_slot_name
    needs_projector_stage = has_modal_inputs .and. &
      staged_modal_kind_before /= MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS
    required_reports = merge(2_i64, 1_i64, needs_projector_stage)
    prefill_cold_state = merge(MIZU_COLD_STATE_WARM, MIZU_COLD_STATE_COLD, session%has_live_context)

    status_code = prepare_report_buffer(out_reports_ptr, required_reports)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_report_buffer_error(session, out_reports_ptr, status_code)
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_prefill(session)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session cannot prefill in current state")
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    projector_embedding_count = 0_i64
    if (needs_projector_stage) then
      call prepare_projector_stage_candidate(runtime, optimization_store, model, staged_modal_byte_count_before, &
        staged_modal_kind_before, staged_modal_dtype_before, staged_modal_slot_name_before, &
        projector_optimization_key_text, projector_candidate_key_text, projector_plan_id, &
        projector_selection_mode, projector_backend_family, projector_route, projector_fallback_reason, &
        projector_artifact_metadata, projector_placeholder_count, status_code)
      if (status_code /= MIZU_STATUS_OK) then
        call set_session_owner_runtime_error(session, status_code, "no valid projector route is available")
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if

      call reserve_stage_workspace(runtime, projector_artifact_metadata, projector_workspace_reserved, status_code)
      if (status_code /= MIZU_STATUS_OK) then
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if

      stage_started_us = monotonic_timestamp_us()
      if (projector_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
          (projector_route == MIZU_EXEC_ROUTE_ANE .or. projector_route == MIZU_EXEC_ROUTE_METAL)) then
        call execute_apple_projector(runtime%config%cache_root, trim(projector_artifact_metadata%payload_path), &
          projector_route, staged_modal_byte_count_before, projector_placeholder_count, staged_modal_hash_before, &
          projector_embedding_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use)
        if (status_code /= MIZU_STATUS_OK) then
          call release_stage_workspace(runtime, projector_workspace_reserved)
          mizu_session_prefill = int(status_code, kind=c_int32_t)
          return
        end if
      else if (projector_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. projector_route == MIZU_EXEC_ROUTE_CUDA) then
        call execute_cuda_projector(runtime%config%cache_root, trim(projector_artifact_metadata%payload_path), &
          staged_modal_byte_count_before, projector_placeholder_count, staged_modal_hash_before, &
          projector_embedding_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use)
        if (status_code /= MIZU_STATUS_OK) then
          call release_stage_workspace(runtime, projector_workspace_reserved)
          mizu_session_prefill = int(status_code, kind=c_int32_t)
          return
        end if
      end if
      call release_stage_workspace(runtime, projector_workspace_reserved)
      projector_elapsed_us = elapsed_since_us(stage_started_us)
      call finalize_projector_stage_cache(runtime_cache, optimization_store, trim(projector_optimization_key_text), &
        trim(projector_candidate_key_text), projector_plan_id, projector_selection_mode, projector_elapsed_us, &
        projector_artifact_metadata, projector_cache_flags)
      projector_report = make_stage_report(MIZU_STAGE_PROJECTOR, projector_backend_family, &
        projector_route, projector_fallback_reason, projector_selection_mode, &
        resolve_stage_cold_state(prefill_cold_state, projector_selection_mode, projector_cache_flags), &
        projector_cache_flags, projector_plan_id, projector_elapsed_us)
    else
      projector_report = execution_report()
      if (has_modal_inputs) projector_embedding_count = max(1_i64, int(staged_modal_before, kind=i64))
    end if

    call prepare_plan_stage_candidate(runtime, optimization_store, model, MIZU_STAGE_PREFILL, &
      OP_FAMILY_PREFILL, [max(0_i64, kv_before), max(0_i64, staged_tokens_before), &
      max(0_i64, int(staged_modal_before, kind=i64))], max(0_i64, staged_tokens_before), &
      model%info%allowed_backend_mask, prefill_optimization_key_text, prefill_candidate_key_text, &
      prefill_plan_id, prefill_selection_mode, prefill_backend_family, prefill_route, &
      prefill_fallback_reason, prefill_artifact_metadata, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "no valid prefill route is available")
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    call reserve_stage_workspace(runtime, prefill_artifact_metadata, prefill_workspace_reserved, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    consumed_token_count = staged_tokens_before
    prefill_context_byte_count = 0_i32
    prefill_context_bytes = 0_i8
    prefill_context_artifact_hash = 0_i64
    if (prefill_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
        (prefill_route == MIZU_EXEC_ROUTE_ANE .or. prefill_route == MIZU_EXEC_ROUTE_METAL)) then
      if (allocated(session%staged_tokens)) then
        if (allocated(session%staged_modal_bytes)) then
          call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
            staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
            runtime%workspace%bytes_in_use, token_values=session%staged_tokens, &
            modal_bytes=session%staged_modal_bytes, context_bytes=prefill_context_bytes, &
            context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
        else
          call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
            staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
            runtime%workspace%bytes_in_use, token_values=session%staged_tokens, &
            context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
            context_artifact_hash=prefill_context_artifact_hash)
        end if
      else if (allocated(session%staged_modal_bytes)) then
        call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
          staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use, modal_bytes=session%staged_modal_bytes, &
          context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
          context_artifact_hash=prefill_context_artifact_hash)
      else
        call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
          staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use, context_bytes=prefill_context_bytes, &
          context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
      end if
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, prefill_workspace_reserved)
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if
    else if (prefill_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. prefill_route == MIZU_EXEC_ROUTE_CUDA) then
      if (allocated(session%staged_tokens)) then
        if (allocated(session%staged_modal_bytes)) then
          call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
            consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
            token_values=session%staged_tokens, modal_bytes=session%staged_modal_bytes, &
            context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
            context_artifact_hash=prefill_context_artifact_hash)
        else
          call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
            consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
            token_values=session%staged_tokens, context_bytes=prefill_context_bytes, &
            context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
        end if
      else if (allocated(session%staged_modal_bytes)) then
        call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
          consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
          modal_bytes=session%staged_modal_bytes, context_bytes=prefill_context_bytes, &
          context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
      else
        call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
          consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
          context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
          context_artifact_hash=prefill_context_artifact_hash)
      end if
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, prefill_workspace_reserved)
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if
    end if
    call release_stage_workspace(runtime, prefill_workspace_reserved)
    call complete_prefill(session, consumed_token_count=consumed_token_count, status_code=status_code, &
      token_content_hash=staged_token_hash_before, modal_content_hash=staged_modal_hash_before, &
      projector_embedding_count=projector_embedding_count)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call store_live_context_record(session, prefill_backend_family, prefill_route, prefill_context_bytes, &
      prefill_context_byte_count, producer_stage=MIZU_STAGE_PREFILL, artifact_hash=prefill_context_artifact_hash)
    prefill_elapsed_us = elapsed_since_us(stage_started_us)

    call finalize_plan_stage_cache(runtime_cache, optimization_store, trim(prefill_optimization_key_text), &
      trim(prefill_candidate_key_text), prefill_plan_id, prefill_selection_mode, prefill_elapsed_us, &
      prefill_artifact_metadata, prefill_cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_PREFILL, prefill_backend_family, prefill_route, &
      prefill_fallback_reason, prefill_selection_mode, &
      resolve_stage_cold_state(prefill_cold_state, prefill_selection_mode, prefill_cache_flags), &
      prefill_cache_flags, prefill_plan_id, prefill_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, projector_report)

    mizu_session_prefill = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_prefill

  integer(c_int32_t) function mizu_session_decode_step(session_ptr, options_ptr, out_result_ptr, &
                                                       out_reports_ptr) &
      bind(c, name="mizu_session_decode_step")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: options_ptr
    type(c_ptr), value :: out_result_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer       :: box
    type(session_state), pointer     :: session
    type(model_state), pointer       :: model
    type(runtime_state), pointer     :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    type(c_decode_options), pointer  :: options
    type(c_decode_result), pointer   :: result
    integer(c_int32_t), pointer      :: token_buffer(:)
    integer(c_int32_t)               :: token_value
    integer(i32) :: status_code
    integer(i64) :: emitted_token_count
    integer(i64) :: kv_before
    integer(i64) :: remaining_context_tokens
    integer(i64) :: decode_cache_flags
    integer(i64) :: decode_plan_id
    integer(i64) :: decode_elapsed_us
    integer(i64) :: stage_started_us
    integer(i64) :: decode_shape_band
    integer(i64) :: decode_allowed_backend_mask
    integer(i32) :: decode_stop_reason
    integer(i32) :: updated_context_byte_count
    integer(i8)  :: updated_context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64) :: decode_context_artifact_hash
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    integer(i32) :: decode_fallback_reason
    integer(i32) :: emitted_tokens_local(MAX_RECENT_OUTPUT_TOKENS)
    character(len=MAX_CACHE_KEY_LEN) :: decode_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: decode_candidate_key_text
    logical      :: decode_progressed
    logical      :: terminal_decode
    logical      :: decode_workspace_reserved
    type(artifact_metadata_record) :: decode_artifact_metadata

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(options_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "decode options pointer is null")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if
    if (.not. c_associated(out_result_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "decode result pointer is null")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(options_ptr, options)
    call c_f_pointer(out_result_ptr, result)
    if (.not. associated(options)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "decode options pointer is invalid")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if
    if (.not. associated(result)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "decode result pointer is invalid")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_input_struct_size(options%struct_size, c_sizeof(options))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "decode options struct_size is too small")
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    status_code = require_output_struct_size(result%struct_size, c_sizeof(result))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "decode result struct_size is too small")
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_report_buffer_error(session, out_reports_ptr, status_code)
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_decode(session)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session cannot decode in current state")
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    if (options%token_budget <= 0_c_int64_t) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, "decode token budget must be positive")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    kv_before = session%kv_token_count
    remaining_context_tokens = 1_i64
    if (session%config%max_context_tokens > 0_i64) then
      remaining_context_tokens = session%config%max_context_tokens - kv_before
      if (remaining_context_tokens <= 0_i64) then
        result%token_count = 0_c_size_t
        result%stop_reason = int(MIZU_STOP_REASON_TOKEN_BUDGET, kind=c_int32_t)
        result%result_flags = 0_c_int64_t
        session%last_report = make_stage_report(MIZU_STAGE_DECODE, session%live_context_backend_family, &
          session%live_context_execution_route, MIZU_FALLBACK_REASON_NONE, MIZU_SELECTION_MODE_NONE, &
          MIZU_COLD_STATE_WARM, MIZU_CACHE_FLAG_NONE, 0_i64, 0_i64)
        call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())
        mizu_session_decode_step = int(MIZU_STATUS_END_OF_SEQUENCE, kind=c_int32_t)
        return
      end if
    end if
    emitted_tokens_local = 0_i32
    updated_context_byte_count = 0_i32
    updated_context_bytes = 0_i8
    decode_context_artifact_hash = 0_i64
    decode_shape_band = decode_kv_shape_band(kv_before)
    decode_allowed_backend_mask = model%info%allowed_backend_mask
    if (session%has_live_context .and. session%live_context_execution_route /= MIZU_EXEC_ROUTE_NONE) then
      decode_allowed_backend_mask = iand(decode_allowed_backend_mask, &
        execution_route_backend_mask(session%live_context_execution_route))
    end if
    if (decode_allowed_backend_mask == MIZU_BACKEND_MASK_NONE) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_STATE, &
        "live context route is no longer allowed for decode")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_STATE, kind=c_int32_t)
      return
    end if
    call prepare_plan_stage_candidate(runtime, optimization_store, model, MIZU_STAGE_DECODE, &
      OP_FAMILY_DECODE, [decode_shape_band, max(0_i64, int(options%token_budget, kind=i64)), 1_i64], &
      max(0_i64, int(options%token_budget, kind=i64)), decode_allowed_backend_mask, &
      decode_optimization_key_text, decode_candidate_key_text, decode_plan_id, selection_mode, &
      report_backend_family, report_route, decode_fallback_reason, decode_artifact_metadata, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "no valid decode route is available")
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    if (session%has_live_context .and. session%live_context_producer_stage == MIZU_STAGE_DECODE) then
      if (report_backend_family /= session%live_context_backend_family .or. &
          report_route /= session%live_context_execution_route) then
        call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_STATE, &
          "decode route no longer matches live decode context")
        mizu_session_decode_step = int(MIZU_STATUS_INVALID_STATE, kind=c_int32_t)
        return
      end if
    end if

    call reserve_stage_workspace(runtime, decode_artifact_metadata, decode_workspace_reserved, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    emitted_token_count = min(int(options%token_budget, kind=i64), 1_i64)
    if (session%config%max_context_tokens > 0_i64) then
      emitted_token_count = min(emitted_token_count, remaining_context_tokens)
    end if
    decode_stop_reason = MIZU_STOP_REASON_NONE
    token_value = int(mod(session%kv_token_count, 4096_i64), kind=c_int32_t)
    if (token_value == 0_c_int32_t) token_value = 1_c_int32_t
    decode_progressed = (emitted_token_count > 0_i64)
    if (decode_progressed .and. report_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
        (report_route == MIZU_EXEC_ROUTE_ANE .or. report_route == MIZU_EXEC_ROUTE_METAL)) then
      call execute_apple_decode(runtime%config%cache_root, trim(decode_artifact_metadata%payload_path), &
        report_route, kv_before, int(options%token_budget, kind=i64), emitted_token_count, token_value, &
        decode_stop_reason, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
        session%live_context_bytes, session%live_context_byte_count, updated_context_bytes, &
        updated_context_byte_count, context_artifact_hash=decode_context_artifact_hash)
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, decode_workspace_reserved)
        mizu_session_decode_step = int(status_code, kind=c_int32_t)
        return
      end if
    else if (decode_progressed .and. report_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. &
             report_route == MIZU_EXEC_ROUTE_CUDA) then
      call execute_cuda_decode(runtime%config%cache_root, trim(decode_artifact_metadata%payload_path), &
        kv_before, int(options%token_budget, kind=i64), emitted_token_count, token_value, &
        decode_stop_reason, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
        session%live_context_bytes, session%live_context_byte_count, updated_context_bytes, &
        updated_context_byte_count, context_artifact_hash=decode_context_artifact_hash)
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, decode_workspace_reserved)
        mizu_session_decode_step = int(status_code, kind=c_int32_t)
        return
      end if
    end if
    call release_stage_workspace(runtime, decode_workspace_reserved)
    if (decode_progressed .and. session%config%max_context_tokens > 0_i64) then
      if (kv_before + emitted_token_count >= session%config%max_context_tokens) then
        decode_stop_reason = MIZU_STOP_REASON_TOKEN_BUDGET
      end if
    end if
    terminal_decode = (decode_stop_reason /= MIZU_STOP_REASON_NONE)

    result%token_count  = int(emitted_token_count, kind=c_size_t)
    result%stop_reason  = int(decode_stop_reason, kind=c_int32_t)
    result%result_flags = 0_c_int64_t

    if (result%token_capacity < int(emitted_token_count, kind=c_size_t)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_BUFFER_TOO_SMALL, "decode result token buffer is too small")
      mizu_session_decode_step = int(MIZU_STATUS_BUFFER_TOO_SMALL, kind=c_int32_t)
      return
    end if

    if (emitted_token_count > 0_i64 .and. .not. c_associated(result%token_buffer)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "decode result token storage pointer is null")
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (emitted_token_count > 0_i64) emitted_tokens_local(1) = int(token_value, kind=i32)
    call complete_decode(session, emitted_token_count, decode_stop_reason, status_code, emitted_tokens_local)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    if (decode_progressed .and. ((report_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
         (report_route == MIZU_EXEC_ROUTE_ANE .or. report_route == MIZU_EXEC_ROUTE_METAL)) .or. &
        (report_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. report_route == MIZU_EXEC_ROUTE_CUDA))) then
      call update_live_context_record(session, updated_context_bytes, updated_context_byte_count, &
        producer_stage=MIZU_STAGE_DECODE, artifact_hash=decode_context_artifact_hash, &
        backend_family=report_backend_family, execution_route=report_route)
    end if
    decode_elapsed_us = elapsed_since_us(stage_started_us)

    if (emitted_token_count > 0_i64) then
      call c_f_pointer(result%token_buffer, token_buffer, [int(emitted_token_count)])
      token_buffer(1) = token_value
    end if

    call finalize_plan_stage_cache(runtime_cache, optimization_store, trim(decode_optimization_key_text), &
      trim(decode_candidate_key_text), decode_plan_id, selection_mode, decode_elapsed_us, &
      decode_artifact_metadata, decode_cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_DECODE, report_backend_family, report_route, &
      decode_fallback_reason, selection_mode, &
      resolve_stage_cold_state(MIZU_COLD_STATE_WARM, selection_mode, decode_cache_flags), &
      decode_cache_flags, decode_plan_id, decode_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())

    if (terminal_decode) then
      mizu_session_decode_step = int(MIZU_STATUS_END_OF_SEQUENCE, kind=c_int32_t)
    else
      mizu_session_decode_step = int(MIZU_STATUS_OK, kind=c_int32_t)
    end if
  end function mizu_session_decode_step

  integer(c_int32_t) function mizu_session_read_output(session_ptr, out_output_ptr) &
      bind(c, name="mizu_session_read_output")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_output_ptr
    type(session_box), pointer      :: box
    type(session_state), pointer    :: session
    type(c_output_buffer), pointer  :: output
    integer(c_int32_t), pointer     :: token_buffer(:)
    integer(c_int32_t)              :: token_example
    integer(i32) :: status_code
    integer(i64) :: bytes_required

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_read_output = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_output_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "session output buffer pointer is null")
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_output_ptr, output)
    if (.not. associated(output)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "session output buffer pointer is invalid")
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_output_struct_size(output%struct_size, c_sizeof(output))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session output buffer struct_size is too small")
      mizu_session_read_output = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_read_output(session)
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session has no decode output to read")
      mizu_session_read_output = int(status_code, kind=c_int32_t)
      return
    end if

    if (int(output%output_kind, kind=i32) /= MIZU_OUTPUT_KIND_TOKEN_IDS) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_UNSUPPORTED_MODALITY, &
        "session output kind is unsupported")
      mizu_session_read_output = int(MIZU_STATUS_UNSUPPORTED_MODALITY, kind=c_int32_t)
      return
    end if

    bytes_required = session%last_output_token_count * int(c_sizeof(token_example), kind=i64)
    output%bytes_written = int(bytes_required, kind=c_size_t)

    if (output%byte_capacity < int(bytes_required, kind=c_size_t)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_BUFFER_TOO_SMALL, "session output buffer is too small")
      mizu_session_read_output = int(MIZU_STATUS_BUFFER_TOO_SMALL, kind=c_int32_t)
      return
    end if

    if (bytes_required > 0_i64 .and. .not. c_associated(output%data)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "session output storage pointer is null")
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (session%last_output_token_count > 0_i64) then
      call c_f_pointer(output%data, token_buffer, [int(session%last_output_token_count)])
      token_buffer(1:int(session%last_output_token_count)) = int( &
        session%last_output_tokens(1:int(session%last_output_token_count)), kind=c_int32_t)
    end if

    mizu_session_read_output = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_read_output

  integer(c_int32_t) function mizu_session_get_last_report(session_ptr, out_report_ptr) &
      bind(c, name="mizu_session_get_last_report")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_report_ptr
    type(session_box), pointer      :: box
    type(session_state), pointer    :: session
    type(c_execution_report), pointer :: c_report
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_report_ptr)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "session report output pointer is null")
      mizu_session_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_report_ptr, c_report)
    if (.not. associated(c_report)) then
      call set_session_owner_runtime_error(session, MIZU_STATUS_INVALID_ARGUMENT, &
        "session report output pointer is invalid")
      mizu_session_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = require_output_struct_size(c_report%struct_size, c_sizeof(c_report))
    if (status_code /= MIZU_STATUS_OK) then
      call set_session_owner_runtime_error(session, status_code, "session report output struct_size is too small")
      mizu_session_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    call copy_internal_report_to_c(session%last_report, c_report)
    mizu_session_get_last_report = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_get_last_report

  subroutine ensure_runtime_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(runtime_state), allocatable :: new_registry(:)
    type(runtime_cache_bundle), allocatable :: new_cache_registry(:)
    type(runtime_optimization_store), allocatable :: new_optimization_registry(:)
    type(c_ptr), allocatable         :: new_handle_ptrs(:)
    logical, allocatable             :: new_used(:)
    integer(i64)                     :: current_capacity, new_capacity

    if (.not. allocated(runtime_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(runtime_registry(new_capacity), runtime_cache_registry(new_capacity), &
               runtime_optimization_registry(new_capacity), runtime_handle_ptrs(new_capacity), &
               runtime_used(new_capacity))
      runtime_registry = runtime_state()
      runtime_cache_registry = runtime_cache_bundle()
      runtime_optimization_registry = runtime_optimization_store()
      runtime_handle_ptrs = c_null_ptr
      runtime_used     = .false.
      return
    end if

    current_capacity = int(size(runtime_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_cache_registry(new_capacity), &
             new_optimization_registry(new_capacity), &
             new_handle_ptrs(new_capacity), new_used(new_capacity))
    new_registry = runtime_state()
    new_cache_registry = runtime_cache_bundle()
    new_optimization_registry = runtime_optimization_store()
    new_handle_ptrs = c_null_ptr
    new_used     = .false.
    new_registry(1:current_capacity) = runtime_registry
    new_cache_registry(1:current_capacity) = runtime_cache_registry
    new_optimization_registry(1:current_capacity) = runtime_optimization_registry
    new_handle_ptrs(1:current_capacity) = runtime_handle_ptrs
    new_used(1:current_capacity)     = runtime_used
    call move_alloc(new_registry, runtime_registry)
    call move_alloc(new_cache_registry, runtime_cache_registry)
    call move_alloc(new_optimization_registry, runtime_optimization_registry)
    call move_alloc(new_handle_ptrs, runtime_handle_ptrs)
    call move_alloc(new_used, runtime_used)
  end subroutine ensure_runtime_registry_capacity

  subroutine ensure_model_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(model_state), allocatable :: new_registry(:)
    type(c_ptr), allocatable       :: new_handle_ptrs(:)
    logical, allocatable           :: new_used(:)
    integer(i64)                   :: current_capacity, new_capacity

    if (.not. allocated(model_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(model_registry(new_capacity), model_handle_ptrs(new_capacity), &
               model_used(new_capacity))
      model_registry = model_state()
      model_handle_ptrs = c_null_ptr
      model_used     = .false.
      return
    end if

    current_capacity = int(size(model_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_handle_ptrs(new_capacity), new_used(new_capacity))
    new_registry = model_state()
    new_handle_ptrs = c_null_ptr
    new_used     = .false.
    new_registry(1:current_capacity) = model_registry
    new_handle_ptrs(1:current_capacity) = model_handle_ptrs
    new_used(1:current_capacity)     = model_used
    call move_alloc(new_registry, model_registry)
    call move_alloc(new_handle_ptrs, model_handle_ptrs)
    call move_alloc(new_used, model_used)
  end subroutine ensure_model_registry_capacity

  subroutine ensure_session_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(session_state), allocatable :: new_registry(:)
    type(c_ptr), allocatable         :: new_handle_ptrs(:)
    logical, allocatable             :: new_used(:)
    integer(i64)                     :: current_capacity, new_capacity

    if (.not. allocated(session_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(session_registry(new_capacity), session_handle_ptrs(new_capacity), &
               session_used(new_capacity))
      session_registry = session_state()
      session_handle_ptrs = c_null_ptr
      session_used     = .false.
      return
    end if

    current_capacity = int(size(session_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_handle_ptrs(new_capacity), new_used(new_capacity))
    new_registry = session_state()
    new_handle_ptrs = c_null_ptr
    new_used     = .false.
    new_registry(1:current_capacity) = session_registry
    new_handle_ptrs(1:current_capacity) = session_handle_ptrs
    new_used(1:current_capacity)     = session_used
    call move_alloc(new_registry, session_registry)
    call move_alloc(new_handle_ptrs, session_handle_ptrs)
    call move_alloc(new_used, session_used)
  end subroutine ensure_session_registry_capacity

  integer(i64) function acquire_runtime_slot() result(slot_id)
    integer(i64) :: index

    call ensure_runtime_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(runtime_used), kind=i64)
      if (.not. runtime_used(index)) then
        runtime_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(runtime_used), kind=i64) + 1_i64
    call ensure_runtime_registry_capacity(index)
    runtime_used(index) = .true.
    slot_id = index
  end function acquire_runtime_slot

  integer(i64) function acquire_model_slot() result(slot_id)
    integer(i64) :: index

    call ensure_model_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(model_used), kind=i64)
      if (.not. model_used(index)) then
        model_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(model_used), kind=i64) + 1_i64
    call ensure_model_registry_capacity(index)
    model_used(index) = .true.
    slot_id = index
  end function acquire_model_slot

  integer(i64) function acquire_session_slot() result(slot_id)
    integer(i64) :: index

    call ensure_session_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(session_used), kind=i64)
      if (.not. session_used(index)) then
        session_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(session_used), kind=i64) + 1_i64
    call ensure_session_registry_capacity(index)
    session_used(index) = .true.
    slot_id = index
  end function acquire_session_slot

  subroutine resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    type(c_ptr), value             :: runtime_ptr
    type(runtime_box), pointer     :: box
    type(runtime_state), pointer   :: runtime
    integer(i32), intent(out)      :: status_code
    integer(i64)                   :: slot_id

    nullify(box)
    nullify(runtime)
    if (.not. c_associated(runtime_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = find_runtime_handle_slot(runtime_ptr)
    if (.not. is_runtime_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(runtime_handle_ptrs(slot_id), box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    runtime => runtime_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_runtime_handle

  subroutine resolve_model_handle(model_ptr, box, model, status_code)
    type(c_ptr), value           :: model_ptr
    type(model_box), pointer     :: box
    type(model_state), pointer   :: model
    integer(i32), intent(out)    :: status_code
    integer(i64)                 :: slot_id

    nullify(box)
    nullify(model)
    if (.not. c_associated(model_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = find_model_handle_slot(model_ptr)
    if (.not. is_model_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(model_handle_ptrs(slot_id), box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    model => model_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_handle

  subroutine resolve_session_handle(session_ptr, box, session, status_code)
    type(c_ptr), value             :: session_ptr
    type(session_box), pointer     :: box
    type(session_state), pointer   :: session
    integer(i32), intent(out)      :: status_code
    integer(i64)                   :: slot_id

    nullify(box)
    nullify(session)
    if (.not. c_associated(session_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = find_session_handle_slot(session_ptr)
    if (.not. is_session_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(session_handle_ptrs(slot_id), box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session => session_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_session_handle

  integer(i64) function find_runtime_handle_slot(runtime_ptr) result(slot_id)
    type(c_ptr), value :: runtime_ptr
    integer(i64)       :: index

    slot_id = 0_i64
    if (.not. allocated(runtime_handle_ptrs)) return

    do index = 1_i64, int(size(runtime_handle_ptrs), kind=i64)
      if (runtime_used(index) .and. c_associated(runtime_ptr, runtime_handle_ptrs(index))) then
        slot_id = index
        return
      end if
    end do
  end function find_runtime_handle_slot

  integer(i64) function find_model_handle_slot(model_ptr) result(slot_id)
    type(c_ptr), value :: model_ptr
    integer(i64)       :: index

    slot_id = 0_i64
    if (.not. allocated(model_handle_ptrs)) return

    do index = 1_i64, int(size(model_handle_ptrs), kind=i64)
      if (model_used(index) .and. c_associated(model_ptr, model_handle_ptrs(index))) then
        slot_id = index
        return
      end if
    end do
  end function find_model_handle_slot

  integer(i64) function find_session_handle_slot(session_ptr) result(slot_id)
    type(c_ptr), value :: session_ptr
    integer(i64)       :: index

    slot_id = 0_i64
    if (.not. allocated(session_handle_ptrs)) return

    do index = 1_i64, int(size(session_handle_ptrs), kind=i64)
      if (session_used(index) .and. c_associated(session_ptr, session_handle_ptrs(index))) then
        slot_id = index
        return
      end if
    end do
  end function find_session_handle_slot

  subroutine retire_runtime_box(slot_id, box)
    integer(i64), intent(in)       :: slot_id
    type(runtime_box), pointer     :: box

    ! Retire the wrapper box without freeing it so stale opaque pointers cannot
    ! alias a future handle after allocator reuse.
    box%id = 0_c_int64_t
    runtime_handle_ptrs(slot_id) = c_null_ptr
    runtime_used(slot_id) = .false.
    retired_runtime_box_count = retired_runtime_box_count + 1_i64
  end subroutine retire_runtime_box

  subroutine retire_model_box(slot_id, box)
    integer(i64), intent(in)     :: slot_id
    type(model_box), pointer     :: box

    ! Retire the wrapper box without freeing it so stale opaque pointers cannot
    ! alias a future handle after allocator reuse.
    box%id = 0_c_int64_t
    model_handle_ptrs(slot_id) = c_null_ptr
    model_used(slot_id) = .false.
    retired_model_box_count = retired_model_box_count + 1_i64
  end subroutine retire_model_box

  subroutine retire_session_box(slot_id, box)
    integer(i64), intent(in)       :: slot_id
    type(session_box), pointer     :: box

    ! Retire the wrapper box without freeing it so stale opaque pointers cannot
    ! alias a future handle after allocator reuse.
    box%id = 0_c_int64_t
    session_handle_ptrs(slot_id) = c_null_ptr
    session_used(slot_id) = .false.
    retired_session_box_count = retired_session_box_count + 1_i64
  end subroutine retire_session_box

  pure logical function is_runtime_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = .false.
    if (.not. allocated(runtime_used)) return
    if (.not. allocated(runtime_handle_ptrs)) return
    if (slot_id < 1_i64) return
    if (slot_id > int(size(runtime_used), kind=i64)) return
    if (.not. runtime_used(slot_id)) return
    is_valid = c_associated(runtime_handle_ptrs(slot_id))
  end function is_runtime_slot_valid

  pure logical function is_model_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = .false.
    if (.not. allocated(model_used)) return
    if (.not. allocated(model_handle_ptrs)) return
    if (slot_id < 1_i64) return
    if (slot_id > int(size(model_used), kind=i64)) return
    if (.not. model_used(slot_id)) return
    is_valid = c_associated(model_handle_ptrs(slot_id))
  end function is_model_slot_valid

  pure logical function is_session_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = .false.
    if (.not. allocated(session_used)) return
    if (.not. allocated(session_handle_ptrs)) return
    if (slot_id < 1_i64) return
    if (slot_id > int(size(session_used), kind=i64)) return
    if (.not. session_used(slot_id)) return
    is_valid = c_associated(session_handle_ptrs(slot_id))
  end function is_session_slot_valid

  integer(i32) function build_model_info(config, available_backend_mask, manifest, info) result(status_code)
    type(model_open_config), intent(in) :: config
    integer(i64), intent(in)            :: available_backend_mask
    type(model_manifest), intent(out)   :: manifest
    type(model_info), intent(out)       :: info
    integer(i64)                        :: effective_backend_mask

    info = model_info()

    if (config%allowed_backend_mask == MIZU_BACKEND_MASK_NONE) then
      status_code = MIZU_STATUS_NO_VALID_PLAN
      return
    end if

    status_code = load_model_manifest_from_root(config%model_root, manifest)
    if (status_code /= MIZU_STATUS_OK) return

    effective_backend_mask = iand(config%allowed_backend_mask, available_backend_mask)
    if (effective_backend_mask == MIZU_BACKEND_MASK_NONE) then
      status_code = MIZU_STATUS_NO_VALID_PLAN
      return
    end if

    call populate_model_info_from_manifest(manifest, info)
    info%allowed_backend_mask = effective_backend_mask
    status_code = MIZU_STATUS_OK
  end function build_model_info

  subroutine resolve_session_owner_model(session, model, status_code)
    type(session_state), intent(in)   :: session
    type(model_state), pointer        :: model
    integer(i32), intent(out)         :: status_code
    integer(i64)                      :: model_id

    nullify(model)
    model_id = session%model_owner%value
    if (.not. is_model_slot_valid(model_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    model => model_registry(model_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_session_owner_model

  subroutine resolve_model_owner_runtime(model, runtime, status_code)
    type(model_state), intent(in) :: model
    type(runtime_state), pointer  :: runtime
    integer(i32), intent(out)     :: status_code
    integer(i64)                  :: runtime_id

    nullify(runtime)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    runtime => runtime_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_runtime

  subroutine resolve_model_owner_cache(model, runtime_cache, status_code)
    type(model_state), intent(in)            :: model
    type(runtime_cache_bundle), pointer      :: runtime_cache
    integer(i32), intent(out)                :: status_code
    integer(i64)                             :: runtime_id

    nullify(runtime_cache)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    runtime_cache => runtime_cache_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_cache

  subroutine resolve_model_owner_optimizer(model, optimization_store, status_code)
    type(model_state), intent(in)                 :: model
    type(runtime_optimization_store), pointer     :: optimization_store
    integer(i32), intent(out)                     :: status_code
    integer(i64)                                  :: runtime_id

    nullify(optimization_store)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    optimization_store => runtime_optimization_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_optimizer

  subroutine set_model_owner_runtime_error(model, status_code, message)
    type(model_state), intent(in)       :: model
    integer(i32), intent(in)            :: status_code
    character(len=*), intent(in)        :: message
    type(runtime_state), pointer        :: runtime
    integer(i32)                        :: owner_status

    call resolve_model_owner_runtime(model, runtime, owner_status)
    if (owner_status /= MIZU_STATUS_OK) return

    call set_runtime_error(runtime, status_code, message)
  end subroutine set_model_owner_runtime_error

  subroutine set_session_owner_runtime_error(session, status_code, message)
    type(session_state), intent(in)     :: session
    integer(i32), intent(in)            :: status_code
    character(len=*), intent(in)        :: message
    type(model_state), pointer          :: model
    integer(i32)                        :: owner_status

    call resolve_session_owner_model(session, model, owner_status)
    if (owner_status /= MIZU_STATUS_OK) return

    call set_model_owner_runtime_error(model, status_code, message)
  end subroutine set_session_owner_runtime_error

  subroutine set_session_report_buffer_error(session, report_buffer_ptr, status_code)
    type(session_state), intent(in) :: session
    type(c_ptr), value              :: report_buffer_ptr
    integer(i32), intent(in)        :: status_code
    type(c_report_buffer), pointer  :: report_buffer
    integer(i32)                    :: size_status

    if (status_code == MIZU_STATUS_BUFFER_TOO_SMALL) then
      call c_f_pointer(report_buffer_ptr, report_buffer)
      if (associated(report_buffer)) then
        size_status = require_output_struct_size(report_buffer%struct_size, c_sizeof(report_buffer))
        if (size_status /= MIZU_STATUS_OK) then
          call set_session_owner_runtime_error(session, status_code, "session report buffer struct_size is too small")
        else
          call set_session_owner_runtime_error(session, status_code, "session report buffer is too small")
        end if
      else
        call set_session_owner_runtime_error(session, status_code, "session report buffer is too small")
      end if
      return
    end if

    if (status_code /= MIZU_STATUS_INVALID_ARGUMENT) return

    call c_f_pointer(report_buffer_ptr, report_buffer)
    if (.not. associated(report_buffer)) then
      call set_session_owner_runtime_error(session, status_code, "session report buffer pointer is invalid")
    else if (.not. c_associated(report_buffer%reports)) then
      call set_session_owner_runtime_error(session, status_code, "session report storage pointer is null")
    else
      call set_session_owner_runtime_error(session, status_code, "session report buffer is invalid")
    end if
  end subroutine set_session_report_buffer_error

  pure subroutine populate_manifest_identity(model, manifest)
    type(model_state), intent(in)   :: model
    type(model_manifest), intent(out) :: manifest

    manifest = model_manifest()
    manifest%model_family = model%info%model_family
    manifest%model_features = model%info%model_features
    manifest%logical_model_hash = model%logical_model_hash
    manifest%provenance%source_model_id = model%source_model_id
    manifest%projector%is_present = (model%projector_revision /= 0_i64)
    manifest%projector%revision_identity = model%projector_revision
    if (manifest%projector%is_present) then
      manifest%projector%slot_name = "image"
      manifest%projector%placeholder_count = 1_i32
      manifest%projector%input_dtype = MIZU_DTYPE_U8
      manifest%projector%embedding_dtype = MIZU_DTYPE_BF16
    end if
  end subroutine populate_manifest_identity

  pure integer(i32) function manifest_runtime_version_code(manifest) result(version_code)
    type(model_manifest), intent(in) :: manifest

    version_code = max(0_i32, manifest%runtime_version%manifest_major) * 1000_i32 + &
      max(0_i32, manifest%runtime_version%manifest_minor)
  end function manifest_runtime_version_code

  pure integer(i32) function clamp_i64_to_i32_nonnegative(value) result(clamped_value)
    integer(i64), intent(in) :: value
    integer(i64)             :: upper_bound

    upper_bound = int(huge(0_i32), kind=i64)
    clamped_value = int(min(max(0_i64, value), upper_bound), kind=i32)
  end function clamp_i64_to_i32_nonnegative

  pure function default_backend_device_key(backend_family, execution_route) result(device_key)
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    character(len=MAX_NAME_LEN) :: device_key

    device_key = "logical"
    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      select case (execution_route)
      case (MIZU_EXEC_ROUTE_ANE)
        device_key = "apple_ane"
      case (MIZU_EXEC_ROUTE_METAL)
        device_key = "apple_metal"
      case default
        device_key = "apple"
      end select
    case (MIZU_BACKEND_FAMILY_CUDA)
      device_key = "cuda"
    end select
  end function default_backend_device_key

  pure subroutine initialize_cache_key_identity(manifest, backend_family, execution_route, device_key, versions)
    type(model_manifest), intent(in)                  :: manifest
    integer(i32), intent(in)                          :: backend_family
    integer(i32), intent(in)                          :: execution_route
    character(len=*), intent(out)                     :: device_key
    type(invalidation_version_fields), intent(out)    :: versions

    versions = invalidation_version_fields()
    versions%abi_version = max(MIZU_ABI_VERSION, manifest%runtime_version%abi_version)
    versions%planner_version = max(0_i32, manifest%runtime_version%planner_version)
    versions%pack_version = max(0_i32, manifest%runtime_version%pack_version)
    versions%backend_version = manifest_runtime_version_code(manifest)
    device_key = default_backend_device_key(backend_family, execution_route)
  end subroutine initialize_cache_key_identity

  pure subroutine resolve_cache_key_identity(runtime, manifest, backend_family, execution_route, device_key, versions)
    type(runtime_state), intent(in)                   :: runtime
    type(model_manifest), intent(in)                  :: manifest
    integer(i32), intent(in)                          :: backend_family
    integer(i32), intent(in)                          :: execution_route
    character(len=*), intent(out)                     :: device_key
    type(invalidation_version_fields), intent(out)    :: versions
    integer(i32)                                      :: backend_index

    call initialize_cache_key_identity(manifest, backend_family, execution_route, device_key, versions)
    if (backend_family == MIZU_BACKEND_FAMILY_NONE) return

    do backend_index = 1_i32, runtime%detected_backend_count
      if (runtime%detected_backends(backend_index)%family /= backend_family) cycle
      if (len_trim(runtime%detected_backends(backend_index)%device_name) > 0) then
        device_key = trim(runtime%detected_backends(backend_index)%device_name)
      end if
      if (runtime%detected_backends(backend_index)%planner_version > 0_i64) then
        versions%planner_version = clamp_i64_to_i32_nonnegative( &
          runtime%detected_backends(backend_index)%planner_version)
      end if
      return
    end do
  end subroutine resolve_cache_key_identity

  subroutine copy_model_import_snapshot(manifest, model)
    type(model_manifest), intent(in)    :: manifest
    type(model_state), intent(inout)    :: model
    character(len=MAX_PATH_LEN + 3 * MAX_NAME_LEN + 96) :: lineage_entry
    integer(i32)                        :: tensor_index
    integer(i32)                        :: preview_index
    integer(i64)                        :: entry_hash

    model%has_import_bundle = (manifest%provenance%source_format == SOURCE_FORMAT_MIZU_IMPORT_BUNDLE)
    model%import_inventory_hash = 0_i64
    model%import_tensor_bytes = 0_i64
    model%import_weight_pack_bytes = 0_i64
    model%import_weight_pack_hash = 0_i64
    model%import_projector_bytes = 0_i64
    model%import_weight_pack_count = 0_i32
    model%import_preview_count = 0_i32
    model%import_projector_artifact_path = ""
    model%import_tensor_names = ""
    model%import_tensor_roles = ""
    model%import_tensor_paths = ""
    if (allocated(model%import_tensors)) deallocate(model%import_tensors)

    if (.not. model%has_import_bundle) return

    if (allocated(manifest%tensors)) then
      allocate(model%import_tensors(size(manifest%tensors)))
      model%import_tensors = import_tensor_state()
      preview_index = 0_i32
      do tensor_index = 1_i32, int(size(manifest%tensors), kind=i32)
        model%import_tensors(tensor_index)%dtype = manifest%tensors(tensor_index)%dtype
        model%import_tensors(tensor_index)%rank = manifest%tensors(tensor_index)%rank
        model%import_tensors(tensor_index)%shape = manifest%tensors(tensor_index)%shape
        model%import_tensors(tensor_index)%source_offset = manifest%tensors(tensor_index)%source_offset
        model%import_tensors(tensor_index)%tensor_name = manifest%tensors(tensor_index)%tensor_name
        model%import_tensors(tensor_index)%tensor_role = manifest%tensors(tensor_index)%tensor_role
        model%import_tensors(tensor_index)%layout_name = manifest%tensors(tensor_index)%layout_name
        model%import_tensors(tensor_index)%storage_type = manifest%tensors(tensor_index)%storage_type
        model%import_tensors(tensor_index)%source_path = manifest%tensors(tensor_index)%source_path

        lineage_entry = ""
        write(lineage_entry, '(A,"|",A,"|",A,"|storage=",A,"|source_offset=",I0)') &
          trim(manifest%tensors(tensor_index)%tensor_name), &
          trim(manifest%tensors(tensor_index)%tensor_role), trim(manifest%tensors(tensor_index)%source_path), &
          trim(manifest%tensors(tensor_index)%storage_type), manifest%tensors(tensor_index)%source_offset
        entry_hash = hash_text64(trim(lineage_entry))
        model%import_inventory_hash = ieor(model%import_inventory_hash, entry_hash)
        model%import_tensor_bytes = model%import_tensor_bytes + estimate_manifest_tensor_bytes( &
          manifest%tensors(tensor_index)%dtype, manifest%tensors(tensor_index)%rank, &
          manifest%tensors(tensor_index)%shape, manifest%tensors(tensor_index)%storage_type)

        if (preview_index >= int(size(model%import_tensor_paths), kind=i32)) cycle
        if (len_trim(manifest%tensors(tensor_index)%source_path) == 0) cycle

        preview_index = preview_index + 1_i32
        model%import_tensor_names(preview_index) = trim(manifest%tensors(tensor_index)%tensor_name)
        model%import_tensor_roles(preview_index) = trim(manifest%tensors(tensor_index)%tensor_role)
        model%import_tensor_paths(preview_index) = trim(manifest%tensors(tensor_index)%source_path)
      end do
      model%import_preview_count = preview_index
    end if

    if (manifest%projector%is_present .and. len_trim(manifest%projector%artifact_path) > 0) then
      model%import_projector_artifact_path = trim(manifest%projector%artifact_path)
      entry_hash = hash_text64(trim(manifest%projector%artifact_path))
      model%import_inventory_hash = ieor(model%import_inventory_hash, entry_hash)
      if (allocated(model%import_tensors)) then
        do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
          if (.not. import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle
          model%import_projector_bytes = model%import_projector_bytes + &
            estimate_import_tensor_bytes(model%import_tensors(tensor_index))
        end do
      end if
    end if

    call recompute_import_weight_pack_summary(model)

    if (model%import_inventory_hash == 0_i64) then
      model%import_inventory_hash = hash_text64(trim(model%source_model_id) // ":import_bundle")
    end if
  end subroutine copy_model_import_snapshot

  subroutine recompute_import_weight_pack_summary(model)
    type(model_state), intent(inout)    :: model
    character(len=MAX_PATH_LEN + 3 * MAX_NAME_LEN + 144) :: pack_entry
    integer(i32)                        :: tensor_index
    integer(i64)                        :: tensor_bytes
    integer(i64)                        :: pack_offset

    model%import_weight_pack_bytes = 0_i64
    model%import_weight_pack_hash = 0_i64
    model%import_weight_pack_count = 0_i32
    if (.not. allocated(model%import_tensors)) return

    pack_offset = 0_i64
    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle

      model%import_weight_pack_count = model%import_weight_pack_count + 1_i32
      pack_entry = ""
      write(pack_entry, '(A,"|",A,"|",A,"|offset=",I0,"|bytes=",I0,"|layout=",A,' // &
        '"|storage=",A,"|source_offset=",I0)') &
        trim(model%import_tensors(tensor_index)%tensor_name), trim(model%import_tensors(tensor_index)%tensor_role), &
        trim(model%import_tensors(tensor_index)%source_path), pack_offset, tensor_bytes, &
        trim(model%import_tensors(tensor_index)%layout_name), trim(model%import_tensors(tensor_index)%storage_type), &
        model%import_tensors(tensor_index)%source_offset
      model%import_weight_pack_hash = ieor(model%import_weight_pack_hash, hash_text64(trim(pack_entry)))
      pack_offset = align_import_bytes(pack_offset + tensor_bytes)
    end do

    model%import_weight_pack_bytes = pack_offset
    if (model%import_weight_pack_count > 0_i32 .and. model%import_weight_pack_hash == 0_i64) then
      model%import_weight_pack_hash = hash_text64(trim(model%source_model_id) // ":weight_pack")
    end if
  end subroutine recompute_import_weight_pack_summary

  subroutine append_import_lineage_payload(payload_text, payload_bytes, stage_kind, model)
    character(len=*), intent(inout) :: payload_text
    integer(i64), intent(out)       :: payload_bytes
    integer(i32), intent(in)        :: stage_kind
    type(model_state), intent(in)   :: model
    character(len=MAX_PATH_LEN + 64) :: field_text
    integer(i32)                     :: preview_index

    if (.not. model%has_import_bundle) then
      payload_bytes = int(len_trim(payload_text) + 1, kind=i64)
      return
    end if

    field_text = ""
    write(field_text, '(";source_id=",A)') trim(model%source_model_id)
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";import_hash=",Z16.16)') model%import_inventory_hash
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";tensor_bytes=",I0)') model%import_tensor_bytes
    call append_payload_fragment(payload_text, trim(field_text))

    if (stage_kind == MIZU_STAGE_MODEL_LOAD .or. stage_kind == MIZU_STAGE_PROJECTOR) then
      field_text = ""
      write(field_text, '(";weight_pack_bytes=",I0)') model%import_weight_pack_bytes
      call append_payload_fragment(payload_text, trim(field_text))

      field_text = ""
      write(field_text, '(";weight_pack_hash=",Z16.16)') model%import_weight_pack_hash
      call append_payload_fragment(payload_text, trim(field_text))

      field_text = ""
      write(field_text, '(";weight_pack_count=",I0)') model%import_weight_pack_count
      call append_payload_fragment(payload_text, trim(field_text))

      if (len_trim(model%import_projector_artifact_path) > 0) then
        field_text = ""
        write(field_text, '(";projector_artifact=",A)') trim(model%import_projector_artifact_path)
        call append_payload_fragment(payload_text, trim(field_text))

        field_text = ""
        write(field_text, '(";projector_bytes=",I0)') model%import_projector_bytes
        call append_payload_fragment(payload_text, trim(field_text))
      end if
    end if

    if (stage_kind == MIZU_STAGE_MODEL_LOAD) then
      call append_import_weight_pack_payload(payload_text, model)
    else if (stage_kind == MIZU_STAGE_PROJECTOR) then
      call append_import_weight_pack_dependency_payload(payload_text, model)
    end if

    if (stage_kind == MIZU_STAGE_MODEL_LOAD .or. stage_kind == MIZU_STAGE_PROJECTOR) then
      do preview_index = 1_i32, model%import_preview_count
        field_text = ""
        write(field_text, '(";tensor",I0,"=",A,"|",A,"|",A)') preview_index, &
          trim(model%import_tensor_names(preview_index)), trim(model%import_tensor_roles(preview_index)), &
          trim(model%import_tensor_paths(preview_index))
        call append_payload_fragment(payload_text, trim(field_text))
      end do
    end if

    if (model%tensor_count > model%import_preview_count) then
      field_text = ""
      write(field_text, '(";tensor_preview=",I0,"/",I0)') model%import_preview_count, model%tensor_count
      call append_payload_fragment(payload_text, trim(field_text))
    end if

    payload_bytes = int(len_trim(payload_text) + 1, kind=i64)
  end subroutine append_import_lineage_payload

  subroutine append_import_weight_pack_payload(payload_text, model)
    character(len=*), intent(inout) :: payload_text
    type(model_state), intent(in)   :: model
    character(len=MAX_PATH_LEN + MAX_NAME_LEN + 224) :: field_text
    integer(i32)                      :: tensor_index
    integer(i32)                      :: pack_index
    integer(i64)                      :: tensor_bytes
    integer(i64)                      :: pack_offset
    integer(i64)                      :: pack_total_bytes

    if (.not. allocated(model%import_tensors)) return

    pack_index = 0_i32
    pack_offset = 0_i64

    field_text = ""
    write(field_text, '(";pack_kind=cuda_import_weight_pack_v1")')
    call append_payload_fragment(payload_text, trim(field_text))

    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle

      pack_index = pack_index + 1_i32
      field_text = ""
      write(field_text, '(";pack",I0,"=",A,"|",A,"|",A,"|offset=",I0,"|bytes=",I0,' // &
        '"|layout=",A,"|storage=",A,"|source_offset=",I0)') &
        pack_index, trim(model%import_tensors(tensor_index)%tensor_name), &
        trim(model%import_tensors(tensor_index)%tensor_role), trim(model%import_tensors(tensor_index)%source_path), &
        pack_offset, tensor_bytes, trim(model%import_tensors(tensor_index)%layout_name), &
        trim(model%import_tensors(tensor_index)%storage_type), model%import_tensors(tensor_index)%source_offset
      call append_payload_fragment(payload_text, trim(field_text))

      pack_offset = align_import_bytes(pack_offset + tensor_bytes)
    end do

    pack_total_bytes = pack_offset
    field_text = ""
    write(field_text, '(";pack_count=",I0)') pack_index
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";pack_total_bytes=",I0)') pack_total_bytes
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";pack_hash=",Z16.16)') model%import_weight_pack_hash
    call append_payload_fragment(payload_text, trim(field_text))
  end subroutine append_import_weight_pack_payload

  subroutine append_import_weight_pack_dependency_payload(payload_text, model)
    character(len=*), intent(inout) :: payload_text
    type(model_state), intent(in)   :: model
    character(len=96)               :: field_text

    if (model%import_weight_pack_hash == 0_i64) return

    field_text = ""
    write(field_text, '(";pack_dependency=cuda_import_weight_pack_v1")')
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";pack_ref_hash=",Z16.16)') model%import_weight_pack_hash
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";pack_ref_count=",I0)') model%import_weight_pack_count
    call append_payload_fragment(payload_text, trim(field_text))

    field_text = ""
    write(field_text, '(";pack_ref_bytes=",I0)') model%import_weight_pack_bytes
    call append_payload_fragment(payload_text, trim(field_text))
  end subroutine append_import_weight_pack_dependency_payload

  function append_import_stage_usage_identity(base_key_text, stage_kind, model) result(key_text)
    character(len=*), intent(in) :: base_key_text
    integer(i32), intent(in)     :: stage_kind
    type(model_state), intent(in) :: model
    character(len=MAX_CACHE_KEY_LEN) :: key_text
    integer(i32)                    :: usage_count
    integer(i64)                    :: usage_bytes
    integer(i64)                    :: usage_hash

    key_text = ""
    if (len_trim(base_key_text) == 0) return

    call summarize_import_stage_usage(stage_kind, model, usage_hash, usage_count, usage_bytes)
    if (usage_count <= 0_i32) then
      key_text = trim(base_key_text)
      return
    end if

    write(key_text, '(A,":usehash=",Z16.16,":usecount=",I0,":usebytes=",I0)') trim(base_key_text), &
      usage_hash, usage_count, usage_bytes
  end function append_import_stage_usage_identity

  subroutine summarize_import_stage_usage(stage_kind, model, usage_hash, usage_count, usage_bytes)
    integer(i32), intent(in)      :: stage_kind
    type(model_state), intent(in) :: model
    integer(i64), intent(out)     :: usage_hash
    integer(i32), intent(out)     :: usage_count
    integer(i64), intent(out)     :: usage_bytes
    character(len=MAX_PATH_LEN + MAX_NAME_LEN + 224) :: usage_entry
    integer(i32)                    :: tensor_index
    integer(i64)                    :: tensor_bytes
    integer(i64)                    :: pack_offset

    usage_hash = 0_i64
    usage_count = 0_i32
    usage_bytes = 0_i64
    if (.not. allocated(model%import_tensors)) return

    pack_offset = 0_i64
    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle

      if (import_stage_uses_tensor(stage_kind, model%import_tensors(tensor_index))) then
        usage_count = usage_count + 1_i32
        usage_bytes = usage_bytes + tensor_bytes
        usage_entry = ""
        write(usage_entry, '(A,"|",A,"|offset=",I0,"|bytes=",I0,"|layout=",A,' // &
          '"|storage=",A,"|source_offset=",I0)') &
          trim(model%import_tensors(tensor_index)%tensor_name), &
          trim(model%import_tensors(tensor_index)%tensor_role), pack_offset, tensor_bytes, &
          trim(model%import_tensors(tensor_index)%layout_name), trim(model%import_tensors(tensor_index)%storage_type), &
          model%import_tensors(tensor_index)%source_offset
        usage_hash = ieor(usage_hash, hash_text64(trim(usage_entry)))
      end if

      pack_offset = align_import_bytes(pack_offset + tensor_bytes)
    end do

    if (usage_count > 0_i32 .and. usage_hash == 0_i64) then
      usage_hash = hash_text64(trim(stage_pack_usage_kind_token(stage_kind)) // ":" // trim(model%source_model_id))
    end if
  end subroutine summarize_import_stage_usage

  subroutine summarize_import_stage_dispatch(stage_kind, model, span_root, usage_hash, usage_count, usage_bytes, &
                                             first_pack_offset, last_pack_offset, last_pack_bytes, dispatch_count, &
                                             dispatch_pack_indices, dispatch_offsets, dispatch_bytes, &
                                             dispatch_source_offsets, dispatch_role_codes, dispatch_layout_codes, &
                                             dispatch_span_paths)
    integer(i32), intent(in)      :: stage_kind
    type(model_state), intent(in) :: model
    character(len=*), intent(out) :: span_root
    integer(i64), intent(out)     :: usage_hash
    integer(i32), intent(out)     :: usage_count
    integer(i64), intent(out)     :: usage_bytes
    integer(i64), intent(out)     :: first_pack_offset
    integer(i64), intent(out)     :: last_pack_offset
    integer(i64), intent(out)     :: last_pack_bytes
    integer(i32), intent(out)     :: dispatch_count
    integer(i32), intent(out)     :: dispatch_pack_indices(:)
    integer(i64), intent(out)     :: dispatch_offsets(:)
    integer(i64), intent(out)     :: dispatch_bytes(:)
    integer(i64), intent(out)     :: dispatch_source_offsets(:)
    integer(i32), intent(out)     :: dispatch_role_codes(:)
    integer(i32), intent(out)     :: dispatch_layout_codes(:)
    character(len=*), intent(out) :: dispatch_span_paths(:)
    character(len=MAX_PATH_LEN + MAX_NAME_LEN + 224) :: field_text
    character(len=MAX_NAME_LEN)       :: usage_kind
    integer(i32)                      :: tensor_index
    integer(i32)                      :: usage_index
    integer(i32)                      :: dispatch_index
    integer(i32)                      :: pack_index
    integer(i32)                      :: role_code
    integer(i32)                      :: layout_code
    integer(i64)                      :: tensor_bytes
    integer(i64)                      :: pack_offset

    span_root = ""
    usage_hash = 0_i64
    usage_count = 0_i32
    usage_bytes = 0_i64
    first_pack_offset = 0_i64
    last_pack_offset = 0_i64
    last_pack_bytes = 0_i64
    dispatch_count = 0_i32
    dispatch_pack_indices = 0_i32
    dispatch_offsets = 0_i64
    dispatch_bytes = 0_i64
    dispatch_source_offsets = -1_i64
    dispatch_role_codes = 0_i32
    dispatch_layout_codes = 0_i32
    dispatch_span_paths = ""
    if (.not. allocated(model%import_tensors)) return

    usage_kind = stage_pack_usage_kind_token(stage_kind)
    if (len_trim(usage_kind) == 0) return
    span_root = build_import_bundle_root_path(model)

    usage_index = 0_i32
    dispatch_index = 0_i32
    pack_index = 0_i32
    pack_offset = 0_i64

    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle
      pack_index = pack_index + 1_i32

      if (import_stage_uses_tensor(stage_kind, model%import_tensors(tensor_index))) then
        usage_index = usage_index + 1_i32
        usage_bytes = usage_bytes + tensor_bytes
        if (usage_index == 1_i32) first_pack_offset = pack_offset
        last_pack_offset = pack_offset
        last_pack_bytes = tensor_bytes

        field_text = ""
        write(field_text, '(";pack_use",I0,"=",A,"|",A,"|offset=",I0,"|bytes=",I0,' // &
          '"|layout=",A,"|storage=",A,"|source_offset=",I0)') &
          usage_index, trim(model%import_tensors(tensor_index)%tensor_name), &
          trim(model%import_tensors(tensor_index)%tensor_role), pack_offset, tensor_bytes, &
          trim(model%import_tensors(tensor_index)%layout_name), trim(model%import_tensors(tensor_index)%storage_type), &
          model%import_tensors(tensor_index)%source_offset
        usage_hash = ieor(usage_hash, hash_text64(trim(field_text)))

        if (dispatch_index < min(MAX_IMPORT_STAGE_PACK_DISPATCH, int(size(dispatch_pack_indices), kind=i32))) then
          dispatch_index = dispatch_index + 1_i32
          role_code = import_tensor_role_code(trim(model%import_tensors(tensor_index)%tensor_role))
          layout_code = import_tensor_layout_code(trim(model%import_tensors(tensor_index)%layout_name))
          dispatch_pack_indices(dispatch_index) = pack_index
          dispatch_offsets(dispatch_index) = pack_offset
          dispatch_bytes(dispatch_index) = tensor_bytes
          dispatch_source_offsets(dispatch_index) = model%import_tensors(tensor_index)%source_offset
          dispatch_role_codes(dispatch_index) = role_code
          dispatch_layout_codes(dispatch_index) = layout_code
          if (dispatch_index <= int(size(dispatch_span_paths), kind=i32)) then
            dispatch_span_paths(dispatch_index) = trim(model%import_tensors(tensor_index)%source_path)
          end if

          field_text = ""
          write(field_text, '(";pack_dispatch",I0,"=pack=",I0)') dispatch_index, pack_index
        end if
      end if

      pack_offset = align_import_bytes(pack_offset + tensor_bytes)
    end do

    usage_count = usage_index
    dispatch_count = dispatch_index
    if (usage_count > 0_i32 .and. usage_hash == 0_i64) then
      usage_hash = hash_text64(trim(usage_kind) // ":" // trim(model%source_model_id))
    end if
  end subroutine summarize_import_stage_dispatch

  function stage_pack_usage_kind_token(stage_kind) result(kind_token)
    integer(i32), intent(in)    :: stage_kind
    character(len=MAX_NAME_LEN) :: kind_token

    kind_token = ""
    select case (stage_kind)
    case (MIZU_STAGE_PREFILL)
      kind_token = "cuda_prefill_pack_usage_v1"
    case (MIZU_STAGE_DECODE)
      kind_token = "cuda_decode_pack_usage_v1"
    case default
      kind_token = ""
    end select
  end function stage_pack_usage_kind_token

  pure logical function import_stage_uses_tensor(stage_kind, import_tensor) result(is_used)
    integer(i32), intent(in)         :: stage_kind
    type(import_tensor_state), intent(in) :: import_tensor
    character(len=MAX_NAME_LEN)      :: tensor_role

    tensor_role = trim(import_tensor%tensor_role)
    is_used = .false.

    select case (trim(tensor_role))
    case ("embedding_table")
      is_used = (stage_kind == MIZU_STAGE_PREFILL .or. stage_kind == MIZU_STAGE_DECODE)
    case ("decoder_stack", "normalization")
      is_used = (stage_kind == MIZU_STAGE_PREFILL .or. stage_kind == MIZU_STAGE_DECODE)
    case ("token_projection")
      is_used = (stage_kind == MIZU_STAGE_DECODE)
    case default
      is_used = .false.
    end select
  end function import_stage_uses_tensor

  pure integer(i32) function import_tensor_role_code(role_text) result(role_code)
    character(len=*), intent(in) :: role_text

    select case (trim(role_text))
    case ("embedding_table")
      role_code = 1_i32
    case ("decoder_stack")
      role_code = 2_i32
    case ("normalization")
      role_code = 3_i32
    case ("token_projection")
      role_code = 4_i32
    case ("multimodal_projector")
      role_code = 5_i32
    case default
      role_code = 0_i32
    end select
  end function import_tensor_role_code

  pure integer(i32) function import_tensor_layout_code(layout_text) result(layout_code)
    character(len=*), intent(in) :: layout_text

    select case (trim(layout_text))
    case ("row_major")
      layout_code = 1_i32
    case ("packed")
      layout_code = 2_i32
    case ("vector")
      layout_code = 3_i32
    case default
      layout_code = 0_i32
    end select
  end function import_tensor_layout_code

  function build_import_bundle_root_path(model) result(root_path)
    type(model_state), intent(in)    :: model
    character(len=MAX_PATH_LEN)      :: root_path
    integer(i32)                     :: root_len

    root_path = ""
    if (.not. model%has_import_bundle) return
    if (len_trim(model%open_config%model_root) == 0) return

    root_len = len_trim(model%open_config%model_root)
    if (model%open_config%model_root(root_len:root_len) == "/") then
      root_path = trim(model%open_config%model_root) // "mizu_import"
    else
      root_path = trim(model%open_config%model_root) // "/mizu_import"
    end if
  end function build_import_bundle_root_path

  pure logical function import_tensor_belongs_to_projector(import_tensor, model) result(is_projector_tensor)
    type(import_tensor_state), intent(in) :: import_tensor
    type(model_state), intent(in)         :: model
    character(len=MAX_NAME_LEN)           :: tensor_role

    is_projector_tensor = .false.
    if (len_trim(model%import_projector_artifact_path) == 0) return
    tensor_role = trim(import_tensor%tensor_role)
    if (trim(tensor_role) == "multimodal_projector" .or. trim(tensor_role) == "vision_encoder") then
      is_projector_tensor = .true.
      return
    end if
    if (len_trim(import_tensor%source_path) == 0) return
    is_projector_tensor = (trim(import_tensor%source_path) == trim(model%import_projector_artifact_path))
  end function import_tensor_belongs_to_projector

  subroutine append_payload_fragment(payload_text, fragment)
    character(len=*), intent(inout) :: payload_text
    character(len=*), intent(in)    :: fragment
    integer(i32)                    :: start_index
    integer(i32)                    :: write_count

    if (len_trim(fragment) == 0) return

    start_index = len_trim(payload_text) + 1_i32
    if (start_index > len(payload_text)) return

    write_count = min(len_trim(fragment), len(payload_text) - start_index + 1_i32)
    if (write_count <= 0_i32) return
    payload_text(start_index:start_index + write_count - 1_i32) = fragment(1:write_count)
  end subroutine append_payload_fragment

  pure integer(i64) function estimate_import_tensor_bytes(import_tensor) result(byte_count)
    type(import_tensor_state), intent(in) :: import_tensor

    byte_count = estimate_manifest_tensor_bytes(import_tensor%dtype, import_tensor%rank, import_tensor%shape, &
      import_tensor%storage_type)
  end function estimate_import_tensor_bytes

  pure integer(i64) function estimate_manifest_tensor_bytes(dtype, rank, shape, storage_type) result(byte_count)
    integer(i32), intent(in) :: dtype
    integer(i32), intent(in) :: rank
    integer(i64), intent(in) :: shape(:)
    character(len=*), intent(in), optional :: storage_type
    integer(i32)             :: axis_index
    integer(i64)             :: storage_block_elements
    integer(i64)             :: storage_block_bytes
    integer(i64)             :: element_count

    byte_count = 0_i64
    if (rank <= 0_i32) return

    element_count = 1_i64
    do axis_index = 1_i32, min(rank, int(size(shape), kind=i32))
      if (shape(axis_index) <= 0_i64) return
      element_count = element_count * shape(axis_index)
    end do

    if (present(storage_type)) then
      call resolve_import_storage_block(storage_type, storage_block_elements, storage_block_bytes)
      if (storage_block_elements > 0_i64 .and. storage_block_bytes > 0_i64) then
        byte_count = ((element_count + storage_block_elements - 1_i64) / storage_block_elements) * storage_block_bytes
        return
      end if
    end if

    byte_count = element_count * dtype_storage_bytes(dtype)
  end function estimate_manifest_tensor_bytes

  pure subroutine resolve_import_storage_block(storage_type, block_elements, block_bytes)
    character(len=*), intent(in) :: storage_type
    integer(i64), intent(out)    :: block_elements
    integer(i64), intent(out)    :: block_bytes
    character(len=len(storage_type)) :: normalized_storage

    normalized_storage = lowercase_import_ascii(trim(storage_type))
    block_elements = 0_i64
    block_bytes = 0_i64

    select case (trim(normalized_storage))
    case ("u8", "i8")
      block_elements = 1_i64
      block_bytes = 1_i64
    case ("f16", "bf16", "i16")
      block_elements = 1_i64
      block_bytes = 2_i64
    case ("f32", "i32")
      block_elements = 1_i64
      block_bytes = 4_i64
    case ("f64", "i64")
      block_elements = 1_i64
      block_bytes = 8_i64
    case ("q4_0", "iq4_nl")
      block_elements = 32_i64
      block_bytes = 18_i64
    case ("q4_1")
      block_elements = 32_i64
      block_bytes = 20_i64
    case ("q5_0")
      block_elements = 32_i64
      block_bytes = 22_i64
    case ("q5_1")
      block_elements = 32_i64
      block_bytes = 24_i64
    case ("q8_0")
      block_elements = 32_i64
      block_bytes = 34_i64
    case ("q8_1")
      block_elements = 32_i64
      block_bytes = 36_i64
    case ("q2_k")
      block_elements = 256_i64
      block_bytes = 84_i64
    case ("q3_k")
      block_elements = 256_i64
      block_bytes = 110_i64
    case ("q4_k")
      block_elements = 256_i64
      block_bytes = 144_i64
    case ("q5_k")
      block_elements = 256_i64
      block_bytes = 176_i64
    case ("q6_k")
      block_elements = 256_i64
      block_bytes = 210_i64
    case ("q8_k")
      block_elements = 256_i64
      block_bytes = 292_i64
    case ("iq2_xxs", "tq2_0")
      block_elements = 256_i64
      block_bytes = 66_i64
    case ("iq2_xs")
      block_elements = 256_i64
      block_bytes = 74_i64
    case ("iq2_s")
      block_elements = 256_i64
      block_bytes = 82_i64
    case ("iq3_xxs")
      block_elements = 256_i64
      block_bytes = 98_i64
    case ("iq3_s")
      block_elements = 256_i64
      block_bytes = 110_i64
    case ("iq1_s")
      block_elements = 256_i64
      block_bytes = 50_i64
    case ("iq1_m")
      block_elements = 256_i64
      block_bytes = 56_i64
    case ("iq4_xs")
      block_elements = 256_i64
      block_bytes = 136_i64
    case ("tq1_0")
      block_elements = 256_i64
      block_bytes = 54_i64
    case default
      block_elements = 0_i64
      block_bytes = 0_i64
    end select
  end subroutine resolve_import_storage_block

  pure function lowercase_import_ascii(text) result(lowered)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered
    integer(i32)                 :: index_value
    integer(i32)                 :: code_point

    lowered = text
    do index_value = 1_i32, len(text)
      code_point = iachar(lowered(index_value:index_value), kind=i32)
      if (code_point >= iachar("A", kind=i32) .and. code_point <= iachar("Z", kind=i32)) then
        lowered(index_value:index_value) = achar(code_point + 32_i32)
      end if
    end do
  end function lowercase_import_ascii

  pure integer(i64) function dtype_storage_bytes(dtype) result(byte_count)
    integer(i32), intent(in) :: dtype

    select case (dtype)
    case (MIZU_DTYPE_U8)
      byte_count = 1_i64
    case (MIZU_DTYPE_F16, MIZU_DTYPE_BF16)
      byte_count = 2_i64
    case (MIZU_DTYPE_I32, MIZU_DTYPE_F32)
      byte_count = 4_i64
    case default
      byte_count = 4_i64
    end select
  end function dtype_storage_bytes

  pure integer(i64) function import_workspace_hint_bytes(stage_kind, model) result(workspace_bytes)
    integer(i32), intent(in)      :: stage_kind
    type(model_state), intent(in) :: model

    workspace_bytes = 0_i64
    if (.not. model%has_import_bundle) return

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      if (model%import_weight_pack_bytes > 0_i64) then
        workspace_bytes = align_import_bytes(model%import_weight_pack_bytes)
      else
        workspace_bytes = align_import_bytes(model%import_tensor_bytes)
      end if
    case (MIZU_STAGE_PROJECTOR)
      workspace_bytes = align_import_bytes(model%import_projector_bytes)
    case default
      workspace_bytes = 0_i64
    end select
  end function import_workspace_hint_bytes

  pure integer(i64) function align_import_bytes(byte_count) result(aligned_bytes)
    integer(i64), intent(in) :: byte_count

    if (byte_count <= 0_i64) then
      aligned_bytes = 0_i64
      return
    end if

    aligned_bytes = ((byte_count + 255_i64) / 256_i64) * 256_i64
  end function align_import_bytes

  integer(i64) function resolve_weight_cache_flags(runtime, runtime_cache, optimization_store, manifest, &
                                                   model, allowed_backend_mask, elapsed_us, plan_id, &
                                                   selection_mode, backend_family, execution_route) &
      result(cache_flags)
    type(runtime_state), intent(in)                  :: runtime
    type(runtime_cache_bundle), intent(inout)        :: runtime_cache
    type(runtime_optimization_store), intent(inout)  :: optimization_store
    type(model_manifest), intent(in)                 :: manifest
    type(model_state), intent(in)                    :: model
    integer(i64), intent(in)                         :: allowed_backend_mask
    integer(i64), intent(in)                         :: elapsed_us
    integer(i64), intent(out)                        :: plan_id
    integer(i32), intent(out)                        :: selection_mode
    integer(i32), intent(out)                        :: backend_family
    integer(i32), intent(out)                        :: execution_route
    type(plan_request)                               :: stage_request
    type(weight_cache_key)                           :: optimization_key
    type(weight_cache_key)                           :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                 :: optimization_key_text
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_text
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_texts(3)
    integer(i64)                                     :: candidate_plan_ids(3)
    integer(i32)                                     :: candidate_backend_families(3)
    integer(i32)                                     :: candidate_execution_routes(3)
    integer(i32)                                     :: optimization_backend_family
    integer(i32)                                     :: candidate_count
    integer(i32)                                     :: candidate_index
    character(len=MAX_NAME_LEN)                      :: device_key
    type(invalidation_version_fields)                :: key_versions
    logical                                          :: was_hit
    logical                                          :: reused_winner

    call enumerate_candidate_routes(allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call resolve_cache_key_identity(runtime, manifest, optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      device_key, key_versions)
    call build_weight_cache_key(manifest, trim(device_key), "logical", optimization_backend_family, &
      MIZU_EXEC_ROUTE_NONE, optimization_key, key_versions)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), allowed_backend_mask)
    optimization_key_text = append_import_pack_identity(trim(optimization_key_text), model)
    call initialize_plan_request(stage_request, MIZU_STAGE_MODEL_LOAD, OP_FAMILY_NONE, &
      manifest%model_family, allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1) = manifest%logical_model_hash
    stage_request%shape_signature(2) = manifest%projector%revision_identity
    stage_request%planner_version_hint = int(manifest%runtime_version%planner_version, kind=i64)

    do candidate_index = 1_i32, candidate_count
      call resolve_cache_key_identity(runtime, manifest, candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), device_key, key_versions)
      call build_weight_cache_key(manifest, trim(device_key), "logical", &
        candidate_backend_families(candidate_index), candidate_execution_routes(candidate_index), &
        candidate_key, key_versions)
      candidate_key_texts(candidate_index) = append_import_pack_identity(trim(candidate_key%key_text), model)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    call touch_weight_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_weight_artifact_metadata(runtime_cache, trim(candidate_key_text), &
      build_stage_artifact_metadata(MIZU_STAGE_MODEL_LOAD, backend_family, execution_route, &
        trim(candidate_key_text), stage_request, runtime%config%cache_root, model))
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_WEIGHT_HIT, was_hit, reused_winner)
  end function resolve_weight_cache_flags

  subroutine resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
                                         elapsed_us, plan_id, selection_mode, backend_family, &
                                         execution_route, cache_flags)
    type(runtime_state), intent(in)                 :: runtime
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    type(model_state), intent(in)                   :: model
    type(session_state), intent(in)                 :: session
    integer(i64), intent(in)                        :: elapsed_us
    integer(i64), intent(out)                       :: plan_id
    integer(i32), intent(out)                       :: selection_mode
    integer(i32), intent(out)                       :: backend_family
    integer(i32), intent(out)                       :: execution_route
    integer(i64), intent(out)                       :: cache_flags
    type(model_manifest)                            :: manifest
    type(session_cache_key)                         :: optimization_key
    type(session_cache_key)                         :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                :: optimization_key_text
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_text
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_texts(3)
    integer(i64)                                    :: candidate_plan_ids(3)
    integer(i32)                                    :: candidate_backend_families(3)
    integer(i32)                                    :: candidate_execution_routes(3)
    integer(i32)                                    :: optimization_backend_family
    integer(i32)                                    :: candidate_count
    integer(i32)                                    :: candidate_index
    character(len=MAX_NAME_LEN)                     :: device_key
    type(invalidation_version_fields)               :: key_versions
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call populate_manifest_identity(model, manifest)
    call enumerate_candidate_routes(model%info%allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call resolve_cache_key_identity(runtime, manifest, optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      device_key, key_versions)
    call build_session_cache_key(manifest, trim(device_key), optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      session%config%max_context_tokens, session%config%max_decode_tokens, optimization_key, key_versions)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), &
      model%info%allowed_backend_mask)

    do candidate_index = 1_i32, candidate_count
      call resolve_cache_key_identity(runtime, manifest, candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), device_key, key_versions)
      call build_session_cache_key(manifest, trim(device_key), candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), session%config%max_context_tokens, &
        session%config%max_decode_tokens, candidate_key, key_versions)
      candidate_key_texts(candidate_index) = trim(candidate_key%key_text)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    call touch_session_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_SESSION_HIT, was_hit, reused_winner)
  end subroutine resolve_session_stage_cache

  subroutine persist_session_checkpoint(runtime, runtime_cache, model, session, checkpoint_ready)
    type(runtime_state), intent(in)           :: runtime
    type(runtime_cache_bundle), intent(inout) :: runtime_cache
    type(model_state), intent(in)             :: model
    type(session_state), intent(in)           :: session
    logical, intent(out)                      :: checkpoint_ready
    type(artifact_metadata_record)            :: metadata
    character(len=MAX_CACHE_KEY_LEN)          :: checkpoint_key_text
    character(len=4 * MAX_LIVE_CONTEXT_BYTES + 256) :: payload_text
    integer(i64)                             :: payload_bytes

    checkpoint_ready = .false.
    if (len_trim(runtime%config%cache_root) == 0) return
    if (session%live_context_byte_count <= 0_i32) return
    if (.not. backend_context_bytes_are_valid(session%live_context_backend_family, &
        session%live_context_execution_route, session%live_context_bytes, session%live_context_byte_count)) return

    call build_session_checkpoint_key(model, session, checkpoint_key_text)
    if (len_trim(checkpoint_key_text) == 0) return

    metadata = build_stage_artifact_metadata(MIZU_STAGE_PARK, session%live_context_backend_family, &
      session%live_context_execution_route, trim(checkpoint_key_text))
    call build_session_checkpoint_payload_text(session, payload_text, payload_bytes)
    call materialize_artifact_payload(runtime%config%cache_root, metadata, trim(payload_text), payload_bytes)
    call record_session_artifact_metadata(runtime_cache, trim(checkpoint_key_text), metadata)
    checkpoint_ready = metadata%is_materialized
  end subroutine persist_session_checkpoint

  subroutine restore_session_checkpoint(cache_root, runtime_cache, model, session, restored_ok)
    character(len=*), intent(in)              :: cache_root
    type(runtime_cache_bundle), intent(in)    :: runtime_cache
    type(model_state), intent(in)             :: model
    type(session_state), intent(inout)        :: session
    logical, intent(out)                      :: restored_ok
    type(artifact_metadata_record)            :: metadata
    character(len=MAX_CACHE_KEY_LEN)          :: checkpoint_key_text
    character(len=MAX_PATH_LEN)               :: full_path
    integer(i64)                              :: kv_token_count
    integer(i64)                              :: live_context_hash
    integer(i8)                               :: context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64)                              :: context_artifact_hash
    integer(i32)                              :: backend_family
    integer(i32)                              :: execution_route
    integer(i32)                              :: context_byte_count
    integer(i32)                              :: context_producer_stage
    logical                                   :: found
    logical                                   :: lineage_known
    logical                                   :: loaded_ok

    restored_ok = .false.
    if (len_trim(cache_root) == 0) return
    if (session%live_context_backend_family == MIZU_BACKEND_FAMILY_NONE) return
    if (session%live_context_execution_route == MIZU_EXEC_ROUTE_NONE) return

    call build_session_checkpoint_key(model, session, checkpoint_key_text)
    if (len_trim(checkpoint_key_text) == 0) return

    call lookup_session_artifact_metadata(runtime_cache, trim(checkpoint_key_text), metadata, found)
    if (.not. found) return
    if (.not. metadata%is_materialized) return
    if (len_trim(metadata%payload_path) == 0) return

    full_path = join_cache_root_with_payload_path(cache_root, metadata%payload_path)
    if (len_trim(full_path) == 0) return

    call load_session_checkpoint_payload(trim(full_path), kv_token_count, live_context_hash, backend_family, &
      execution_route, context_bytes, context_byte_count, loaded_ok)
    if (.not. loaded_ok) return
    if (.not. backend_context_bytes_are_valid(backend_family, execution_route, context_bytes, context_byte_count)) return
    if (backend_family /= session%live_context_backend_family) return
    if (execution_route /= session%live_context_execution_route) return
    if (kv_token_count /= session%kv_token_count) return
    if (live_context_hash /= session%live_context_hash) return
    if (context_byte_count /= session%live_context_byte_count) return
    call extract_backend_context_lineage(backend_family, execution_route, context_bytes, context_byte_count, &
      context_producer_stage, context_artifact_hash, lineage_known)
    if (session%live_context_producer_stage /= MIZU_STAGE_NONE) then
      if (.not. lineage_known) return
      if (context_producer_stage /= session%live_context_producer_stage) return
    end if
    if (session%live_context_artifact_hash /= 0_i64) then
      if (.not. lineage_known) return
      if (context_artifact_hash /= session%live_context_artifact_hash) return
    end if

    session%kv_token_count = kv_token_count
    session%live_context_hash = live_context_hash
    session%has_live_context = .true.
    call store_live_context_record(session, backend_family, execution_route, context_bytes, context_byte_count, &
      producer_stage=context_producer_stage, artifact_hash=context_artifact_hash)
    restored_ok = .true.
  end subroutine restore_session_checkpoint

  subroutine build_session_checkpoint_key(model, session, checkpoint_key_text)
    type(model_state), intent(in)         :: model
    type(session_state), intent(in)       :: session
    character(len=*), intent(out)         :: checkpoint_key_text
    type(model_manifest)                  :: manifest
    type(session_cache_key)               :: checkpoint_key
    type(runtime_state), pointer          :: runtime
    type(invalidation_version_fields)     :: key_versions
    character(len=MAX_NAME_LEN)           :: device_key
    integer(i32)                          :: owner_status

    checkpoint_key_text = ""
    if (session%live_context_backend_family == MIZU_BACKEND_FAMILY_NONE) return
    if (session%live_context_execution_route == MIZU_EXEC_ROUTE_NONE) return

    call populate_manifest_identity(model, manifest)
    call resolve_model_owner_runtime(model, runtime, owner_status)
    if (owner_status == MIZU_STATUS_OK) then
      call resolve_cache_key_identity(runtime, manifest, session%live_context_backend_family, &
        session%live_context_execution_route, device_key, key_versions)
    else
      call initialize_cache_key_identity(manifest, session%live_context_backend_family, &
        session%live_context_execution_route, device_key, key_versions)
    end if
    call build_session_cache_key(manifest, trim(device_key), session%live_context_backend_family, &
      session%live_context_execution_route, session%config%max_context_tokens, &
      session%config%max_decode_tokens, checkpoint_key, key_versions)
    write(checkpoint_key_text, '(A,":ctx_hash=",I0,":kv=",I0,":ctx_bytes=",I0)') trim(checkpoint_key%key_text), &
      session%live_context_hash, session%kv_token_count, session%live_context_byte_count
  end subroutine build_session_checkpoint_key

  subroutine build_session_checkpoint_payload_text(session, payload_text, payload_bytes)
    type(session_state), intent(in)       :: session
    character(len=*), intent(out)         :: payload_text
    integer(i64), intent(out)             :: payload_bytes
    character(len=2 * MAX_LIVE_CONTEXT_BYTES) :: hex_text

    call encode_bytes_as_hex(session%live_context_bytes, session%live_context_byte_count, hex_text)
    payload_text = ""
    write(payload_text, '(I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,A)') session%live_context_backend_family, &
      session%live_context_execution_route, session%kv_token_count, session%live_context_hash, &
      session%live_context_byte_count, trim(hex_text)
    payload_bytes = int(len_trim(payload_text), kind=i64)
  end subroutine build_session_checkpoint_payload_text

  subroutine load_session_checkpoint_payload(file_path, kv_token_count, live_context_hash, backend_family, &
                                             execution_route, context_bytes, context_byte_count, loaded_ok)
    character(len=*), intent(in)          :: file_path
    integer(i64), intent(out)             :: kv_token_count
    integer(i64), intent(out)             :: live_context_hash
    integer(i32), intent(out)             :: backend_family
    integer(i32), intent(out)             :: execution_route
    integer(i8), intent(out)              :: context_bytes(:)
    integer(i32), intent(out)             :: context_byte_count
    logical, intent(out)                  :: loaded_ok
    character(len=4 * MAX_LIVE_CONTEXT_BYTES + 256) :: line
    character(len=2 * MAX_LIVE_CONTEXT_BYTES) :: hex_text
    integer(i32)                          :: unit_id
    integer(i32)                          :: ios
    logical                               :: exists

    kv_token_count = 0_i64
    live_context_hash = 0_i64
    backend_family = MIZU_BACKEND_FAMILY_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE
    context_bytes = 0_i8
    context_byte_count = 0_i32
    loaded_ok = .false.

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) return

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return
    read(unit_id, "(A)", iostat=ios) line
    close(unit_id)
    if (ios /= 0_i32) return

    hex_text = ""
    read(line, *, iostat=ios) backend_family, execution_route, kv_token_count, live_context_hash, &
      context_byte_count, hex_text
    if (ios /= 0_i32) return

    call decode_hex_to_bytes(trim(hex_text), context_byte_count, context_bytes, loaded_ok)
  end subroutine load_session_checkpoint_payload

  pure logical function backend_context_bytes_are_valid(backend_family, execution_route, context_bytes, &
                                                        context_byte_count) result(is_valid)
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: context_byte_count

    is_valid = .false.
    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      if (execution_route /= MIZU_EXEC_ROUTE_ANE .and. execution_route /= MIZU_EXEC_ROUTE_METAL) return
      is_valid = apple_context_bytes_are_valid(context_bytes, context_byte_count)
    case (MIZU_BACKEND_FAMILY_CUDA)
      if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
      is_valid = cuda_context_bytes_are_valid(context_bytes, context_byte_count)
    end select
  end function backend_context_bytes_are_valid

  pure subroutine extract_backend_context_lineage(backend_family, execution_route, context_bytes, &
                                                  context_byte_count, producer_stage, artifact_hash, &
                                                  lineage_known)
    integer(i32), intent(in)  :: backend_family
    integer(i32), intent(in)  :: execution_route
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i64), intent(out) :: artifact_hash
    logical, intent(out)      :: lineage_known
    integer(i32)              :: context_route

    producer_stage = MIZU_STAGE_NONE
    artifact_hash = 0_i64
    lineage_known = .false.
    context_route = MIZU_EXEC_ROUTE_NONE

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      call extract_apple_context_lineage(context_bytes, context_byte_count, producer_stage, context_route, &
        artifact_hash, lineage_known)
      if (lineage_known) lineage_known = (context_route == execution_route)
    case (MIZU_BACKEND_FAMILY_CUDA)
      call extract_cuda_context_lineage(context_bytes, context_byte_count, producer_stage, artifact_hash, &
        lineage_known)
      if (lineage_known) lineage_known = (execution_route == MIZU_EXEC_ROUTE_CUDA)
    end select
  end subroutine extract_backend_context_lineage

  subroutine encode_bytes_as_hex(byte_values, byte_count, hex_text)
    integer(i8), intent(in)               :: byte_values(:)
    integer(i32), intent(in)              :: byte_count
    character(len=*), intent(out)         :: hex_text
    character(len=16), parameter          :: HEX_DIGITS = "0123456789ABCDEF"
    integer(i32)                          :: byte_index
    integer(i32)                          :: encoded_count
    integer(i32)                          :: byte_value

    hex_text = ""
    encoded_count = max(0_i32, min(byte_count, min(int(size(byte_values), kind=i32), len(hex_text) / 2)))
    do byte_index = 1_i32, encoded_count
      byte_value = int(byte_values(byte_index), kind=i32)
      if (byte_value < 0_i32) byte_value = byte_value + 256_i32
      hex_text((2 * byte_index) - 1:(2 * byte_index) - 1) = HEX_DIGITS((byte_value / 16_i32) + 1:(byte_value / 16_i32) + 1)
      hex_text(2 * byte_index:2 * byte_index) = HEX_DIGITS(mod(byte_value, 16_i32) + 1:mod(byte_value, 16_i32) + 1)
    end do
  end subroutine encode_bytes_as_hex

  subroutine decode_hex_to_bytes(hex_text, byte_count, byte_values, decoded_ok)
    character(len=*), intent(in)          :: hex_text
    integer(i32), intent(in)              :: byte_count
    integer(i8), intent(out)              :: byte_values(:)
    logical, intent(out)                  :: decoded_ok
    integer(i32)                          :: byte_index
    integer(i32)                          :: decoded_count
    integer(i32)                          :: upper_nibble
    integer(i32)                          :: lower_nibble
    integer(i32)                          :: byte_value

    byte_values = 0_i8
    decoded_ok = .false.
    decoded_count = max(0_i32, min(byte_count, min(int(size(byte_values), kind=i32), len_trim(hex_text) / 2)))
    if (decoded_count <= 0_i32) then
      decoded_ok = (byte_count <= 0_i32)
      return
    end if
    if ((2 * decoded_count) > len_trim(hex_text)) return

    do byte_index = 1_i32, decoded_count
      upper_nibble = hex_digit_value(hex_text((2 * byte_index) - 1:(2 * byte_index) - 1))
      lower_nibble = hex_digit_value(hex_text(2 * byte_index:2 * byte_index))
      if (upper_nibble < 0_i32 .or. lower_nibble < 0_i32) return

      byte_value = (16_i32 * upper_nibble) + lower_nibble
      if (byte_value > 127_i32) then
        byte_values(byte_index) = int(byte_value - 256_i32, kind=i8)
      else
        byte_values(byte_index) = int(byte_value, kind=i8)
      end if
    end do

    decoded_ok = .true.
  end subroutine decode_hex_to_bytes

  pure integer(i32) function hex_digit_value(hex_char) result(digit_value)
    character(len=*), intent(in) :: hex_char
    integer(i32)                 :: ascii_code

    digit_value = -1_i32
    if (len_trim(hex_char) <= 0) return

    ascii_code = iachar(hex_char(1:1))
    select case (ascii_code)
    case (iachar("0"):iachar("9"))
      digit_value = ascii_code - iachar("0")
    case (iachar("A"):iachar("F"))
      digit_value = 10_i32 + ascii_code - iachar("A")
    case (iachar("a"):iachar("f"))
      digit_value = 10_i32 + ascii_code - iachar("a")
    case default
      digit_value = -1_i32
    end select
  end function hex_digit_value

  subroutine prepare_plan_stage_candidate(runtime, optimization_store, model, stage_kind, op_family, &
                                          shape, token_count, allowed_backend_mask, optimization_key_text, &
                                          candidate_key_text, plan_id, selection_mode, &
                                          backend_family, execution_route, fallback_reason, &
                                          artifact_metadata, status_code)
    type(runtime_state), intent(in)                  :: runtime
    type(runtime_optimization_store), intent(inout)  :: optimization_store
    type(model_state), intent(in)                    :: model
    integer(i32), intent(in)                         :: stage_kind
    integer(i32), intent(in)                         :: op_family
    integer(i64), intent(in)                         :: shape(3)
    integer(i64), intent(in)                         :: token_count
    integer(i64), intent(in)                         :: allowed_backend_mask
    character(len=*), intent(out)                    :: optimization_key_text
    character(len=*), intent(out)                    :: candidate_key_text
    integer(i64), intent(out)                        :: plan_id
    integer(i32), intent(out)                        :: selection_mode
    integer(i32), intent(out)                        :: backend_family
    integer(i32), intent(out)                        :: execution_route
    integer(i32), intent(out)                        :: fallback_reason
    type(artifact_metadata_record), intent(out)      :: artifact_metadata
    integer(i32), intent(out)                        :: status_code
    type(model_manifest)                             :: manifest
    type(plan_request)                               :: stage_request
    type(plan_cache_key)                             :: optimization_key
    type(plan_cache_key)                             :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_texts(3)
    integer(i64)                                     :: candidate_plan_ids(3)
    integer(i32)                                     :: candidate_backend_families(3)
    integer(i32)                                     :: candidate_execution_routes(3)
    integer(i32)                                     :: candidate_fallback_reasons(3)
    integer(i32)                                     :: optimization_backend_family
    integer(i32)                                     :: candidate_count
    integer(i32)                                     :: candidate_index
    integer(i32)                                     :: selected_candidate_index
    character(len=MAX_NAME_LEN)                      :: device_key
    type(invalidation_version_fields)                :: key_versions

    call populate_manifest_identity(model, manifest)
    optimization_key_text = ""
    candidate_key_text = ""
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    backend_family = MIZU_BACKEND_FAMILY_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE
    fallback_reason = MIZU_FALLBACK_REASON_NONE
    artifact_metadata = artifact_metadata_record()
    status_code = MIZU_STATUS_OK

    call initialize_plan_request(stage_request, stage_kind, op_family, model%info%model_family, &
      allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1:3) = shape
    stage_request%token_count = max(0_i64, token_count)
    stage_request%planner_version_hint = 1_i64
    call collect_plannable_stage_candidates(stage_request, candidate_backend_families, candidate_execution_routes, &
      candidate_fallback_reasons, candidate_count, status_code)
    if (status_code /= MIZU_STATUS_OK) return
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call resolve_cache_key_identity(runtime, manifest, optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      device_key, key_versions)
    call build_plan_cache_key(manifest, trim(device_key), "logical", stage_kind, optimization_backend_family, &
      MIZU_EXEC_ROUTE_NONE, MIZU_DTYPE_BF16, 3_i32, shape, optimization_key, key_versions)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), &
      allowed_backend_mask)
    optimization_key_text = append_import_pack_identity(trim(optimization_key_text), model)
    optimization_key_text = append_import_stage_usage_identity(trim(optimization_key_text), stage_kind, model)

    do candidate_index = 1_i32, candidate_count
      call resolve_cache_key_identity(runtime, manifest, candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), device_key, key_versions)
      call build_plan_cache_key(manifest, trim(device_key), "logical", stage_kind, &
        candidate_backend_families(candidate_index), candidate_execution_routes(candidate_index), &
        MIZU_DTYPE_BF16, 3_i32, shape, candidate_key, key_versions)
      candidate_key_text = append_import_pack_identity(trim(candidate_key%key_text), model)
      candidate_key_text = append_import_stage_usage_identity(trim(candidate_key_text), stage_kind, model)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_text))
      candidate_key_texts(candidate_index) = trim(candidate_key_text)
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    selected_candidate_index = candidate_index_for_route(candidate_backend_families, candidate_execution_routes, &
      candidate_count, backend_family, execution_route)
    if (selected_candidate_index > 0_i32) then
      fallback_reason = candidate_fallback_reasons(selected_candidate_index)
    end if
    artifact_metadata = build_stage_artifact_metadata(stage_kind, backend_family, execution_route, &
      trim(candidate_key_text), stage_request, runtime%config%cache_root, model)
  end subroutine prepare_plan_stage_candidate

  subroutine finalize_plan_stage_cache(runtime_cache, optimization_store, optimization_key_text, &
                                       candidate_key_text, plan_id, selection_mode, elapsed_us, &
                                       artifact_metadata, cache_flags)
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=*), intent(in)                    :: optimization_key_text
    character(len=*), intent(in)                    :: candidate_key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i32), intent(in)                        :: selection_mode
    integer(i64), intent(in)                        :: elapsed_us
    type(artifact_metadata_record), intent(in)      :: artifact_metadata
    integer(i64), intent(out)                       :: cache_flags
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call touch_plan_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_plan_artifact_metadata(runtime_cache, trim(candidate_key_text), artifact_metadata)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_PLAN_HIT, was_hit, reused_winner)
  end subroutine finalize_plan_stage_cache

  subroutine reserve_stage_workspace(runtime, artifact_metadata, was_reserved, status_code)
    type(runtime_state), intent(inout)             :: runtime
    type(artifact_metadata_record), intent(in)     :: artifact_metadata
    logical, intent(out)                           :: was_reserved
    integer(i32), intent(out)                      :: status_code

    was_reserved = .false.
    status_code = MIZU_STATUS_OK
    if (artifact_metadata%workspace_bytes <= 0_i64) return

    call reserve_workspace_bytes(runtime%workspace, artifact_metadata%workspace_bytes, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      call set_runtime_error(runtime, status_code, "workspace reservation failed")
      return
    end if

    was_reserved = .true.
  end subroutine reserve_stage_workspace

  subroutine release_stage_workspace(runtime, was_reserved)
    type(runtime_state), intent(inout) :: runtime
    logical, intent(in)                :: was_reserved

    if (.not. was_reserved) return
    call release_workspace_bytes(runtime%workspace)
  end subroutine release_stage_workspace

  subroutine prepare_projector_stage_candidate(runtime, optimization_store, model, staged_modal_byte_count, &
                                               staged_modal_kind, staged_modal_dtype, staged_modal_slot_name, &
                                               optimization_key_text, candidate_key_text, plan_id, &
                                               selection_mode, backend_family, execution_route, &
                                               fallback_reason, artifact_metadata, placeholder_count, status_code)
    type(runtime_state), intent(in)                 :: runtime
    type(runtime_optimization_store), intent(inout) :: optimization_store
    type(model_state), intent(in)                   :: model
    integer(i64), intent(in)                        :: staged_modal_byte_count
    integer(i32), intent(in)                        :: staged_modal_kind
    integer(i32), intent(in)                        :: staged_modal_dtype
    character(len=*), intent(in)                    :: staged_modal_slot_name
    character(len=*), intent(out)                   :: optimization_key_text
    character(len=*), intent(out)                   :: candidate_key_text
    integer(i64), intent(out)                       :: plan_id
    integer(i32), intent(out)                       :: selection_mode
    integer(i32), intent(out)                       :: backend_family
    integer(i32), intent(out)                       :: execution_route
    integer(i32), intent(out)                       :: fallback_reason
    type(artifact_metadata_record), intent(out)     :: artifact_metadata
    integer(i32), intent(out)                       :: placeholder_count
    integer(i32), intent(out)                       :: status_code
    type(plan_request)                              :: stage_request
    type(model_manifest)                            :: manifest
    type(multimodal_cache_key)                      :: key
    integer(i32)                                    :: modality_kind
    integer(i32)                                    :: modality_dtype
    character(len=MAX_PATH_LEN)                     :: slot_name
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_texts(3)
    integer(i64)                                    :: candidate_plan_ids(3)
    integer(i32)                                    :: candidate_backend_families(3)
    integer(i32)                                    :: candidate_execution_routes(3)
    integer(i32)                                    :: candidate_fallback_reasons(3)
    integer(i32)                                    :: optimization_backend_family
    integer(i32)                                    :: candidate_count
    integer(i32)                                    :: candidate_index
    integer(i32)                                    :: selected_candidate_index
    character(len=MAX_NAME_LEN)                     :: device_key
    type(invalidation_version_fields)               :: key_versions

    call populate_manifest_identity(model, manifest)
    slot_name = trim(staged_modal_slot_name)
    if (len_trim(slot_name) == 0) slot_name = "image"
    modality_kind = staged_modal_kind
    if (modality_kind <= 0_i32) modality_kind = MIZU_MODALITY_KIND_IMAGE
    modality_dtype = staged_modal_dtype
    if (modality_dtype <= 0_i32) modality_dtype = MIZU_DTYPE_U8
    call initialize_plan_request(stage_request, MIZU_STAGE_PROJECTOR, OP_FAMILY_PROJECTOR, &
      model%info%model_family, model%info%allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1) = max(0_i64, staged_modal_byte_count)
    stage_request%shape_signature(2) = int(modality_kind, kind=i64)
    stage_request%shape_signature(3) = int(modality_dtype, kind=i64)
    stage_request%planner_version_hint = 1_i64
    stage_request%projector%is_present = manifest%projector%is_present
    stage_request%projector%placeholder_count = manifest%projector%placeholder_count
    stage_request%projector%input_dtype = manifest%projector%input_dtype
    stage_request%projector%embedding_dtype = manifest%projector%embedding_dtype
    stage_request%projector%slot_name = manifest%projector%slot_name

    optimization_key_text = ""
    candidate_key_text = ""
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    fallback_reason = MIZU_FALLBACK_REASON_NONE
    artifact_metadata = artifact_metadata_record()
    status_code = MIZU_STATUS_OK
    call collect_plannable_stage_candidates(stage_request, candidate_backend_families, candidate_execution_routes, &
      candidate_fallback_reasons, candidate_count, status_code)
    if (status_code /= MIZU_STATUS_OK) return
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)
    call resolve_cache_key_identity(runtime, manifest, optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      device_key, key_versions)
    call build_multimodal_cache_key(manifest, trim(device_key), trim(slot_name), modality_kind, &
      modality_dtype, max(0_i64, staged_modal_byte_count), key, key_versions)
    optimization_key_text = append_allowed_mask_identity(trim(key%key_text), model%info%allowed_backend_mask)
    optimization_key_text = append_import_pack_identity(trim(optimization_key_text), model)

    do candidate_index = 1_i32, candidate_count
      call resolve_cache_key_identity(runtime, manifest, candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), device_key, key_versions)
      call build_multimodal_cache_key(manifest, trim(device_key), trim(slot_name), modality_kind, &
        modality_dtype, max(0_i64, staged_modal_byte_count), key, key_versions)
      candidate_key_texts(candidate_index) = append_route_identity(trim(key%key_text), &
        candidate_backend_families(candidate_index), candidate_execution_routes(candidate_index))
      candidate_key_texts(candidate_index) = append_import_pack_identity(trim(candidate_key_texts(candidate_index)), model)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    selected_candidate_index = candidate_index_for_route(candidate_backend_families, candidate_execution_routes, &
      candidate_count, backend_family, execution_route)
    if (selected_candidate_index > 0_i32) then
      fallback_reason = candidate_fallback_reasons(selected_candidate_index)
    end if
    artifact_metadata = build_stage_artifact_metadata(MIZU_STAGE_PROJECTOR, backend_family, execution_route, &
      trim(candidate_key_text), stage_request, runtime%config%cache_root, model)
    placeholder_count = max(1_i32, stage_request%projector%placeholder_count)
  end subroutine prepare_projector_stage_candidate

  subroutine collect_plannable_stage_candidates(stage_request, candidate_backend_families, &
                                                candidate_execution_routes, candidate_fallback_reasons, &
                                                candidate_count, status_code)
    type(plan_request), intent(in)  :: stage_request
    integer(i32), intent(out)       :: candidate_backend_families(:)
    integer(i32), intent(out)       :: candidate_execution_routes(:)
    integer(i32), intent(out)       :: candidate_fallback_reasons(:)
    integer(i32), intent(out)       :: candidate_count
    integer(i32), intent(out)       :: status_code
    type(planner_result)            :: planning_result
    integer(i32)                    :: raw_backend_families(3)
    integer(i32)                    :: raw_execution_routes(3)
    integer(i32)                    :: raw_candidate_count
    integer(i32)                    :: raw_candidate_index
    integer(i32)                    :: plan_status
    integer(i32)                    :: selected_candidate_index

    candidate_backend_families = MIZU_BACKEND_FAMILY_NONE
    candidate_execution_routes = MIZU_EXEC_ROUTE_NONE
    candidate_fallback_reasons = MIZU_FALLBACK_REASON_NONE
    candidate_count = 0_i32
    status_code = MIZU_STATUS_OK
    raw_backend_families = MIZU_BACKEND_FAMILY_NONE
    raw_execution_routes = MIZU_EXEC_ROUTE_NONE

    call enumerate_candidate_routes(stage_request%allowed_backend_mask, raw_backend_families, &
      raw_execution_routes, raw_candidate_count)

    do raw_candidate_index = 1_i32, raw_candidate_count
      call plan_stage_route_candidate(stage_request, raw_backend_families(raw_candidate_index), &
        raw_execution_routes(raw_candidate_index), planning_result, plan_status)
      if (plan_status /= MIZU_STATUS_OK) cycle
      if (.not. planner_result_is_success(planning_result)) cycle

      selected_candidate_index = candidate_index_for_route(candidate_backend_families, candidate_execution_routes, &
        candidate_count, planning_result%chosen_plan%backend_family, planning_result%chosen_plan%execution_route)
      if (selected_candidate_index > 0_i32) cycle
      if (candidate_count >= size(candidate_backend_families)) exit

      candidate_count = candidate_count + 1_i32
      candidate_backend_families(candidate_count) = planning_result%chosen_plan%backend_family
      candidate_execution_routes(candidate_count) = planning_result%chosen_plan%execution_route
      candidate_fallback_reasons(candidate_count) = planning_result%fallback_reason
    end do

    if (candidate_count <= 0_i32) then
      status_code = MIZU_STATUS_NO_VALID_PLAN
    else
      status_code = MIZU_STATUS_OK
    end if
  end subroutine collect_plannable_stage_candidates

  subroutine plan_stage_route_candidate(stage_request, preferred_backend_family, preferred_execution_route, &
                                        planning_result, status_code)
    type(plan_request), intent(in)    :: stage_request
    integer(i32), intent(in)          :: preferred_backend_family
    integer(i32), intent(in)          :: preferred_execution_route
    type(planner_result), intent(out) :: planning_result
    integer(i32), intent(out)         :: status_code
    type(plan_request)                :: candidate_request

    candidate_request = stage_request
    candidate_request%preferred_backend_mask = execution_route_backend_mask(preferred_execution_route)
    planning_result = planner_result()

    select case (preferred_backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      call plan_apple_stage(candidate_request, planning_result, status_code)
    case (MIZU_BACKEND_FAMILY_CUDA)
      call plan_cuda_stage(candidate_request, planning_result, status_code)
    case default
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      planning_result%status_code = status_code
    end select
  end subroutine plan_stage_route_candidate

  subroutine finalize_projector_stage_cache(runtime_cache, optimization_store, optimization_key_text, &
                                            candidate_key_text, plan_id, selection_mode, elapsed_us, &
                                            artifact_metadata, cache_flags)
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=*), intent(in)                    :: optimization_key_text
    character(len=*), intent(in)                    :: candidate_key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i32), intent(in)                        :: selection_mode
    integer(i64), intent(in)                        :: elapsed_us
    type(artifact_metadata_record), intent(in)      :: artifact_metadata
    integer(i64), intent(out)                       :: cache_flags
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call touch_multimodal_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_multimodal_artifact_metadata(runtime_cache, trim(candidate_key_text), artifact_metadata)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_MM_HIT, was_hit, reused_winner)
  end subroutine finalize_projector_stage_cache

  subroutine resolve_stage_candidate(runtime, optimization_store, optimization_key_text, &
                                     candidate_count, candidate_backend_families, &
                                     candidate_execution_routes, candidate_plan_ids, &
                                     candidate_key_texts, candidate_key_text, plan_id, &
                                     selection_mode, backend_family, execution_route)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=*), intent(in)                   :: optimization_key_text
    integer(i32), intent(in)                       :: candidate_count
    integer(i32), intent(in)                       :: candidate_backend_families(:)
    integer(i32), intent(in)                       :: candidate_execution_routes(:)
    integer(i64), intent(in)                       :: candidate_plan_ids(:)
    character(len=*), intent(in)                   :: candidate_key_texts(:)
    character(len=*), intent(out)                  :: candidate_key_text
    integer(i64), intent(out)                      :: plan_id
    integer(i32), intent(out)                      :: selection_mode
    integer(i32), intent(out)                      :: backend_family
    integer(i32), intent(out)                      :: execution_route
    character(len=MAX_CACHE_KEY_LEN)               :: winner_candidate_key_text
    integer(i64)                                   :: total_samples
    integer(i64)                                   :: winner_plan_id
    integer(i32)                                   :: candidate_budget
    integer(i32)                                   :: candidate_index
    integer(i32)                                   :: observed_candidate_count
    integer(i32)                                   :: stale_candidate_count
    integer(i32)                                   :: winner_index
    logical                                        :: has_winner

    candidate_key_text = ""
    plan_id = 0_i64
    selection_mode = MIZU_SELECTION_MODE_NONE
    backend_family = MIZU_BACKEND_FAMILY_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE

    if (candidate_count <= 0_i32) return
    call assign_stage_candidate(1_i32, candidate_backend_families, candidate_execution_routes, &
      candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
    selection_mode = MIZU_SELECTION_MODE_DIRECT
    if (len_trim(optimization_key_text) == 0) return

    if (runtime%config%optimization_mode == MIZU_OPTIMIZATION_MODE_DISABLED) return

    candidate_budget = min(max(1_i32, runtime%config%exploration_budget), candidate_count)
    call invalidate_stale_optimization_candidates(optimization_store, trim(optimization_key_text), &
      candidate_key_texts, candidate_count, stale_candidate_count)
    call lookup_optimization_entry_stats(optimization_store, trim(optimization_key_text), &
      total_samples, observed_candidate_count)
    call lookup_winner_candidate(optimization_store, trim(optimization_key_text), winner_plan_id, &
      winner_candidate_key_text, has_winner)
    winner_index = find_candidate_index(candidate_key_texts, candidate_plan_ids, candidate_count, &
      winner_candidate_key_text, winner_plan_id)

    if (candidate_budget <= 1_i32 .or. candidate_count <= 1_i32) then
      if (has_winner .and. winner_index > 0_i32) then
        call assign_stage_candidate(winner_index, candidate_backend_families, candidate_execution_routes, &
          candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
        selection_mode = MIZU_SELECTION_MODE_REUSE
      end if
      return
    end if

    if (has_winner .and. winner_index > 0_i32 .and. total_samples >= int(candidate_budget, kind=i64) .and. &
        observed_candidate_count >= candidate_budget) then
      call assign_stage_candidate(winner_index, candidate_backend_families, candidate_execution_routes, &
        candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
      selection_mode = MIZU_SELECTION_MODE_REUSE
      return
    end if

    candidate_index = 1_i32 + mod(int(total_samples, kind=i32), candidate_budget)
    call assign_stage_candidate(candidate_index, candidate_backend_families, candidate_execution_routes, &
      candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
    selection_mode = MIZU_SELECTION_MODE_EXPLORATORY
  end subroutine resolve_stage_candidate

  pure integer(i64) function compose_cache_flags(hit_flag, was_hit, reused_winner) result(cache_flags)
    integer(i64), intent(in) :: hit_flag
    logical, intent(in)      :: was_hit
    logical, intent(in)      :: reused_winner

    cache_flags = MIZU_CACHE_FLAG_NONE
    if (was_hit) cache_flags = ior(cache_flags, hit_flag)
    if (reused_winner) cache_flags = ior(cache_flags, MIZU_CACHE_FLAG_WINNER_REUSED)
  end function compose_cache_flags

  pure integer(i32) function resolve_stage_cold_state(default_cold_state, selection_mode, cache_flags) &
      result(cold_state)
    integer(i32), intent(in) :: default_cold_state
    integer(i32), intent(in) :: selection_mode
    integer(i64), intent(in) :: cache_flags
    integer(i64)             :: hit_flags

    cold_state = default_cold_state
    if (cold_state == MIZU_COLD_STATE_WARM) return

    hit_flags = ior(ior(MIZU_CACHE_FLAG_WEIGHT_HIT, MIZU_CACHE_FLAG_PLAN_HIT), &
      ior(MIZU_CACHE_FLAG_SESSION_HIT, MIZU_CACHE_FLAG_MM_HIT))
    if (selection_mode == MIZU_SELECTION_MODE_REUSE .or. iand(cache_flags, hit_flags) /= 0_i64) then
      cold_state = MIZU_COLD_STATE_WARM
    end if
  end function resolve_stage_cold_state

  pure subroutine enumerate_candidate_routes(backend_mask, backend_families, execution_routes, candidate_count)
    integer(i64), intent(in)  :: backend_mask
    integer(i32), intent(out) :: backend_families(:)
    integer(i32), intent(out) :: execution_routes(:)
    integer(i32), intent(out) :: candidate_count

    backend_families = MIZU_BACKEND_FAMILY_NONE
    execution_routes = MIZU_EXEC_ROUTE_NONE
    candidate_count = 0_i32

    if (iand(backend_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_APPLE
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_ANE
    end if
    if (iand(backend_mask, MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_APPLE
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_METAL
    end if
    if (iand(backend_mask, MIZU_BACKEND_MASK_CUDA) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_CUDA
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_CUDA
    end if
  end subroutine enumerate_candidate_routes

  pure integer(i64) function execution_route_backend_mask(execution_route) result(route_mask)
    integer(i32), intent(in) :: execution_route

    route_mask = MIZU_BACKEND_MASK_NONE
    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      route_mask = MIZU_BACKEND_MASK_APPLE_ANE
    case (MIZU_EXEC_ROUTE_METAL)
      route_mask = MIZU_BACKEND_MASK_APPLE_METAL
    case (MIZU_EXEC_ROUTE_CUDA)
      route_mask = MIZU_BACKEND_MASK_CUDA
    end select
  end function execution_route_backend_mask

  pure integer(i32) function candidate_index_for_route(candidate_backend_families, candidate_execution_routes, &
                                                       candidate_count, backend_family, execution_route) &
      result(candidate_index)
    integer(i32), intent(in) :: candidate_backend_families(:)
    integer(i32), intent(in) :: candidate_execution_routes(:)
    integer(i32), intent(in) :: candidate_count
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    integer(i32)             :: index

    candidate_index = 0_i32
    do index = 1_i32, candidate_count
      if (candidate_backend_families(index) == backend_family .and. &
          candidate_execution_routes(index) == execution_route) then
        candidate_index = index
        return
      end if
    end do
  end function candidate_index_for_route

  pure integer(i32) function derive_optimization_backend_family(candidate_backend_families, candidate_count) &
      result(backend_family)
    integer(i32), intent(in) :: candidate_backend_families(:)
    integer(i32), intent(in) :: candidate_count
    integer(i32)             :: candidate_index

    backend_family = MIZU_BACKEND_FAMILY_NONE
    if (candidate_count <= 0_i32) return

    backend_family = candidate_backend_families(1)
    do candidate_index = 2_i32, candidate_count
      if (candidate_backend_families(candidate_index) /= backend_family) then
        backend_family = MIZU_BACKEND_FAMILY_NONE
        return
      end if
    end do
  end function derive_optimization_backend_family

  function append_allowed_mask_identity(base_key_text, allowed_backend_mask) result(optimization_key_text)
    character(len=*), intent(in) :: base_key_text
    integer(i64), intent(in)     :: allowed_backend_mask
    character(len=MAX_CACHE_KEY_LEN) :: optimization_key_text

    optimization_key_text = ""
    if (len_trim(base_key_text) == 0) return

    write(optimization_key_text, '(A,":allowmask=",I0)') trim(base_key_text), allowed_backend_mask
  end function append_allowed_mask_identity

  function append_route_identity(base_key_text, backend_family, execution_route) result(candidate_key_text)
    character(len=*), intent(in) :: base_key_text
    integer(i32), intent(in)     :: backend_family
    integer(i32), intent(in)     :: execution_route
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text

    candidate_key_text = ""
    if (len_trim(base_key_text) == 0) return

    write(candidate_key_text, '(A,":candidate_backend=",I0,":candidate_route=",I0)') &
      trim(base_key_text), backend_family, execution_route
  end function append_route_identity

  function append_import_pack_identity(base_key_text, model) result(key_text)
    character(len=*), intent(in) :: base_key_text
    type(model_state), intent(in) :: model
    character(len=MAX_CACHE_KEY_LEN) :: key_text

    key_text = ""
    if (len_trim(base_key_text) == 0) return

    if (model%import_weight_pack_hash == 0_i64) then
      key_text = trim(base_key_text)
      return
    end if

    write(key_text, '(A,":packhash=",Z16.16,":packbytes=",I0)') trim(base_key_text), &
      model%import_weight_pack_hash, model%import_weight_pack_bytes
  end function append_import_pack_identity

  function build_stage_artifact_metadata(stage_kind, backend_family, execution_route, candidate_key_text, &
                                         request, cache_root, model) result(metadata)
    integer(i32), intent(in)      :: stage_kind
    integer(i32), intent(in)      :: backend_family
    integer(i32), intent(in)      :: execution_route
    character(len=*), intent(in)  :: candidate_key_text
    type(plan_request), intent(in), optional :: request
    character(len=*), intent(in), optional   :: cache_root
    type(model_state), intent(in), optional  :: model
    type(artifact_metadata_record) :: metadata
    type(planner_result)           :: planning_result
    type(plan_request)             :: planning_request
    character(len=MAX_NAME_LEN)    :: fingerprint_token
    character(len=:), allocatable  :: payload_text
    integer(i64)                   :: payload_bytes
    integer(i32)                   :: payload_capacity
    integer(i32)                   :: status_code

    metadata = artifact_metadata_record()
    metadata%backend_family = backend_family
    metadata%execution_route = execution_route
    metadata%stage_kind = stage_kind
    metadata%is_materialized = .false.
    metadata%payload_bytes = 0_i64
    metadata%workspace_bytes = 0_i64
    metadata%artifact_format = build_artifact_format_label(stage_kind, backend_family, execution_route)
    fingerprint_token = build_artifact_fingerprint_token(trim(candidate_key_text))
    metadata%payload_fingerprint = trim(fingerprint_token)
    metadata%payload_path = build_artifact_payload_path(stage_kind, backend_family, execution_route, &
      trim(fingerprint_token))
    payload_capacity = max(APPLE_ARTIFACT_PAYLOAD_LEN, CUDA_ARTIFACT_PAYLOAD_LEN) + 65536_i32
    if (present(model)) then
      if (allocated(model%import_tensors)) then
        payload_capacity = payload_capacity + int(size(model%import_tensors), kind=i32) * 256_i32
      end if
    end if
    allocate(character(len=payload_capacity) :: payload_text)

    if (.not. present(request)) return
    planning_request = request

    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_APPLE_ANE)
    case (MIZU_EXEC_ROUTE_METAL)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_APPLE_METAL)
    case (MIZU_EXEC_ROUTE_CUDA)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_CUDA)
    end select

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      if (execution_route /= MIZU_EXEC_ROUTE_ANE .and. execution_route /= MIZU_EXEC_ROUTE_METAL) return
      call plan_apple_stage(planning_request, planning_result, status_code)
      if (status_code /= MIZU_STATUS_OK) return
      if (.not. planner_result_is_success(planning_result)) return

      metadata%artifact_format = trim(planning_result%chosen_plan%pack_format)
      metadata%workspace_bytes = max(0_i64, planning_result%chosen_plan%workspace_bytes)
      if (present(model)) metadata%workspace_bytes = max(metadata%workspace_bytes, &
        import_workspace_hint_bytes(stage_kind, model))
      call build_apple_artifact_payload_text(planning_request, planning_result%chosen_plan, trim(candidate_key_text), &
        payload_text, payload_bytes)
      if (present(model)) call append_import_lineage_payload(payload_text, payload_bytes, stage_kind, model)
      if (present(cache_root)) then
        call materialize_artifact_payload(trim(cache_root), metadata, trim(payload_text), payload_bytes)
      end if
    case (MIZU_BACKEND_FAMILY_CUDA)
      if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
      call plan_cuda_stage(planning_request, planning_result, status_code)
      if (status_code /= MIZU_STATUS_OK) return
      if (.not. planner_result_is_success(planning_result)) return

      metadata%artifact_format = trim(planning_result%chosen_plan%pack_format)
      metadata%workspace_bytes = max(0_i64, planning_result%chosen_plan%workspace_bytes)
      if (present(model)) metadata%workspace_bytes = max(metadata%workspace_bytes, &
        import_workspace_hint_bytes(stage_kind, model))
      call build_cuda_artifact_payload_text(planning_request, planning_result%chosen_plan, trim(candidate_key_text), &
        payload_text, payload_bytes)
      if (present(model)) call append_import_lineage_payload(payload_text, payload_bytes, stage_kind, model)
      if (present(model)) then
        call append_cuda_weight_pack_reference_payload(payload_text, payload_bytes, stage_kind, execution_route, &
          model, metadata%payload_path)
      end if
      if (present(cache_root)) then
        if (present(model)) then
          call materialize_cuda_weight_pack_tile_cache(trim(cache_root), metadata, model)
          call append_cuda_pack_span_cache_payload(trim(cache_root), metadata, payload_text, payload_bytes, model)
        else
          call append_cuda_pack_span_cache_payload(trim(cache_root), metadata, payload_text, payload_bytes)
        end if
        call materialize_artifact_payload(trim(cache_root), metadata, trim(payload_text), payload_bytes)
      end if
    end select
  end function build_stage_artifact_metadata

  function build_artifact_format_label(stage_kind, backend_family, execution_route) result(format_label)
    integer(i32), intent(in)    :: stage_kind
    integer(i32), intent(in)    :: backend_family
    integer(i32), intent(in)    :: execution_route
    character(len=MAX_NAME_LEN) :: format_label
    character(len=MAX_NAME_LEN) :: family_token
    character(len=MAX_NAME_LEN) :: route_token
    character(len=MAX_NAME_LEN) :: stage_token

    format_label = ""
    family_token = artifact_family_token(backend_family)
    route_token = artifact_route_token(execution_route)
    stage_token = artifact_stage_token(stage_kind)
    write(format_label, '(A,"_",A,"_",A,"_v1")') trim(family_token), trim(route_token), trim(stage_token)
  end function build_artifact_format_label

  function build_artifact_payload_path(stage_kind, backend_family, execution_route, fingerprint_token) &
      result(payload_path)
    integer(i32), intent(in)    :: stage_kind
    integer(i32), intent(in)    :: backend_family
    integer(i32), intent(in)    :: execution_route
    character(len=*), intent(in) :: fingerprint_token
    character(len=MAX_PATH_LEN) :: payload_path
    character(len=MAX_NAME_LEN) :: family_token
    character(len=MAX_NAME_LEN) :: route_token

    payload_path = ""
    family_token = artifact_family_token(backend_family)
    route_token = artifact_route_token(execution_route)

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      write(payload_path, '(A,"/",A,"/",A,"/weights/",A,".pack")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PROJECTOR)
      write(payload_path, '(A,"/",A,"/",A,"/projector/",A,".mm")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PREFILL)
      write(payload_path, '(A,"/",A,"/",A,"/plans/prefill/",A,".plan")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_DECODE)
      write(payload_path, '(A,"/",A,"/",A,"/plans/decode/",A,".plan")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PARK, MIZU_STAGE_RESUME)
      write(payload_path, '(A,"/",A,"/",A,"/sessions/",A,".session")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case default
      write(payload_path, '(A,"/",A,"/",A,"/misc/",A,".artifact")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    end select
  end function build_artifact_payload_path

  function build_artifact_fingerprint_token(candidate_key_text) result(fingerprint_token)
    character(len=*), intent(in) :: candidate_key_text
    character(len=MAX_NAME_LEN)  :: fingerprint_token
    integer(i64)                 :: key_hash

    fingerprint_token = ""
    if (len_trim(candidate_key_text) == 0) return

    key_hash = hash_text64(trim(candidate_key_text))
    write(fingerprint_token, '(Z16.16)') key_hash
  end function build_artifact_fingerprint_token

  subroutine materialize_artifact_payload(cache_root, metadata, payload_text, payload_bytes)
    character(len=*), intent(in)              :: cache_root
    type(artifact_metadata_record), intent(inout) :: metadata
    character(len=*), intent(in)              :: payload_text
    integer(i64), intent(in)                  :: payload_bytes
    character(len=MAX_PATH_LEN)               :: full_path
    character(len=MAX_PATH_LEN)               :: parent_dir
    integer(i64)                              :: existing_size
    integer(i32)                              :: unit_id
    integer(i32)                              :: ios
    logical                                   :: exists

    if (len_trim(cache_root) == 0) return
    if (len_trim(metadata%payload_path) == 0) return

    full_path = join_cache_root_with_payload_path(cache_root, metadata%payload_path)
    inquire(file=trim(full_path), exist=exists, size=existing_size)
    if (exists .and. existing_size > 0_i64) then
      metadata%is_materialized = .true.
      metadata%payload_bytes = max(1_i64, existing_size)
      return
    end if

    parent_dir = parent_directory_path(full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)

    open(newunit=unit_id, file=trim(full_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return
    write(unit_id, "(A)", iostat=ios) trim(payload_text)
    close(unit_id)
    if (ios /= 0_i32) return

    metadata%is_materialized = .true.
    metadata%payload_bytes = max(1_i64, payload_bytes)
  end subroutine materialize_artifact_payload

  subroutine append_cuda_pack_span_cache_payload_from_model(cache_root, metadata, payload_text, payload_bytes, model)
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_MAGIC = int(z'58455A4D', kind=i32)
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_VERSION = 3_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_HEADER_BYTES = 72_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_ENTRY_BYTES = 104_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_PAYLOAD_BYTES_PER_ENTRY = 64_i32 + &
                                                  (MAX_CUDA_PACK_PAGE_WORDS * 4_i32) + MAX_CUDA_PACK_TILE_BYTES
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_CAPACITY = CUDA_EXEC_BUFFER_HEADER_BYTES + MAX_PATH_LEN + &
                                                  (MAX_IMPORT_STAGE_PACK_DISPATCH * (CUDA_EXEC_BUFFER_ENTRY_BYTES + &
                                                   CUDA_EXEC_BUFFER_PAYLOAD_BYTES_PER_ENTRY))
    character(len=*), intent(in)               :: cache_root
    type(artifact_metadata_record), intent(in) :: metadata
    character(len=*), intent(in)               :: payload_text
    integer(i64), intent(inout)                :: payload_bytes
    type(model_state), intent(in)              :: model
    character(len=MAX_PATH_LEN)                :: exec_buffer_path
    character(len=MAX_PATH_LEN)                :: pack_tile_path
    character(len=MAX_PATH_LEN)                :: pack_tile_buffer_path
    character(len=MAX_PATH_LEN)                :: weight_payload_path
    character(len=MAX_PATH_LEN)                :: exec_buffer_full_path
    character(len=MAX_PATH_LEN)                :: parent_dir
    character(len=MAX_PATH_LEN)                :: span_root
    character(len=MAX_PATH_LEN)                :: dispatch_span_paths(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i32)                               :: entry_index
    integer(i32)                               :: unit_id
    integer(i32)                               :: ios
    integer(i32)                               :: role_code
    integer(i32)                               :: layout_code
    integer(i32)                               :: pack_index
    integer(i32)                               :: usage_count
    integer(i32)                               :: dispatch_count
    integer(i32)                               :: page_word_count
    integer(i32)                               :: tile_byte_count
    integer(i32)                               :: exec_path_bytes
    integer(i32)                               :: exec_path_offset
    integer(i32)                               :: exec_path_index
    integer(i32)                               :: exec_entry_count
    integer(i32)                               :: exec_buffer_offset
    integer(i32)                               :: exec_record_offset
    integer(i32)                               :: exec_sample_count
    integer(i32)                               :: exec_sample_offset
    integer(i32)                               :: exec_page_byte_count
    integer(i32)                               :: exec_page_data_offset
    integer(i32)                               :: exec_tile_data_offset
    integer(i32)                               :: dispatch_pack_indices(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i32)                               :: dispatch_role_codes(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i32)                               :: dispatch_layout_codes(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i64)                               :: usage_hash
    integer(i64)                               :: usage_total_bytes
    integer(i64)                               :: first_pack_offset
    integer(i64)                               :: last_pack_offset
    integer(i64)                               :: last_pack_bytes
    integer(i64)                               :: pack_materialized_hash
    integer(i64)                               :: page_hash
    integer(i64)                               :: tile_hash
    integer(i64)                               :: span_hash
    integer(i64)                               :: actual_sample_bytes
    integer(i64)                               :: dispatch_offsets(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i64)                               :: dispatch_bytes(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i64)                               :: dispatch_source_offsets(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i8)                                :: sample_bytes(64)
    integer(i8)                                :: page_bytes(MAX_CUDA_PACK_PAGE_WORDS * 4_i32)
    integer(i32)                               :: page_words(MAX_CUDA_PACK_PAGE_WORDS)
    integer(i8)                                :: tile_bytes(MAX_CUDA_PACK_TILE_BYTES)
    integer(i8)                                :: exec_buffer(CUDA_EXEC_BUFFER_CAPACITY)
    logical                                    :: exists
    logical                                    :: has_entries
    payload_bytes = int(len_trim(payload_text) + 1_i64, kind=i64)
    if (.not. model%has_import_bundle) return

    call summarize_import_stage_dispatch(metadata%stage_kind, model, span_root, usage_hash, usage_count, &
      usage_total_bytes, first_pack_offset, last_pack_offset, last_pack_bytes, dispatch_count, &
      dispatch_pack_indices, dispatch_offsets, dispatch_bytes, dispatch_source_offsets, dispatch_role_codes, &
      dispatch_layout_codes, dispatch_span_paths)
    if (dispatch_count <= 0_i32) return
    if (len_trim(span_root) == 0) return

    exec_buffer_path = build_cuda_pack_execution_buffer_path(metadata%payload_path)
    if (len_trim(exec_buffer_path) == 0) return
    exec_buffer_full_path = join_cache_root_with_payload_path(cache_root, exec_buffer_path)
    if (len_trim(exec_buffer_full_path) == 0) return
    inquire(file=trim(exec_buffer_full_path), exist=exists)
    if (exists) return

    weight_payload_path = build_cuda_weight_pack_payload_path(model, metadata%execution_route)
    pack_tile_path = ""
    pack_tile_buffer_path = ""
    if (len_trim(weight_payload_path) > 0) then
      pack_tile_path = build_cuda_weight_pack_tile_cache_path(trim(weight_payload_path))
      pack_tile_buffer_path = build_cuda_weight_pack_tile_buffer_path(trim(weight_payload_path))
    end if

    exec_buffer = 0_i8
    has_entries = .false.
    exec_entry_count = 0_i32
    exec_buffer_offset = CUDA_EXEC_BUFFER_HEADER_BYTES + &
      (MAX_IMPORT_STAGE_PACK_DISPATCH * CUDA_EXEC_BUFFER_ENTRY_BYTES)
    exec_path_bytes = 0_i32
    exec_path_offset = 0_i32

    do entry_index = 1_i32, dispatch_count
      if (len_trim(dispatch_span_paths(entry_index)) == 0) cycle

      call resolve_import_span_record_cache(trim(span_root), trim(dispatch_span_paths(entry_index)), &
        IMPORT_STAGE_SPAN_SAMPLE_BYTES, span_hash, actual_sample_bytes, sample_bytes, &
        dispatch_source_offsets(entry_index), dispatch_bytes(entry_index))
      if (span_hash <= 0_i64) cycle

      pack_index = dispatch_pack_indices(entry_index)
      role_code = dispatch_role_codes(entry_index)
      layout_code = dispatch_layout_codes(entry_index)

      call build_cuda_pack_page_record_cache(dispatch_offsets(entry_index), dispatch_bytes(entry_index), role_code, &
        layout_code, span_hash, sample_bytes, actual_sample_bytes, page_hash, page_word_count, page_words)
      page_bytes = 0_i8
      exec_page_byte_count = 0_i32
      if (page_hash > 0_i64 .and. page_word_count > 0_i32) then
        call pack_i32_words_to_le_bytes_cache(page_words, page_word_count, page_bytes, exec_page_byte_count)
      end if

      call build_cuda_pack_tile_record_cache(dispatch_offsets(entry_index), dispatch_bytes(entry_index), role_code, &
        layout_code, sample_bytes, actual_sample_bytes, tile_hash, tile_byte_count, tile_bytes)

      exec_entry_count = exec_entry_count + 1_i32
      exec_record_offset = CUDA_EXEC_BUFFER_HEADER_BYTES + &
        ((exec_entry_count - 1_i32) * CUDA_EXEC_BUFFER_ENTRY_BYTES)
      exec_sample_count = int(min(actual_sample_bytes, 64_i64), kind=i32)
      exec_sample_offset = 0_i32
      if (exec_sample_count > 0_i32) then
        call append_pack_buffer_bytes_cache(sample_bytes, exec_sample_count, exec_buffer, exec_buffer_offset, &
          exec_sample_offset)
      end if
      exec_page_data_offset = 0_i32
      if (exec_page_byte_count > 0_i32) then
        call append_pack_buffer_bytes_cache(page_bytes, exec_page_byte_count, exec_buffer, exec_buffer_offset, &
          exec_page_data_offset)
      end if
      exec_tile_data_offset = 0_i32
      if (tile_byte_count > 0_i32) then
        call append_pack_buffer_bytes_cache(tile_bytes, tile_byte_count, exec_buffer, exec_buffer_offset, &
          exec_tile_data_offset)
      end if
      pack_materialized_hash = 0_i64
      if (pack_index > 0_i32 .and. model%import_weight_pack_hash > 0_i64) then
        pack_materialized_hash = build_cuda_materialized_pack_seed_cache("cuda_weight_pack_entry", pack_index, &
          dispatch_offsets(entry_index), dispatch_bytes(entry_index), role_code, layout_code, &
          model%import_weight_pack_hash, model%import_weight_pack_bytes, span_hash)
        call build_cuda_materialized_pack_page_record_cache(pack_index, dispatch_offsets(entry_index), &
          dispatch_bytes(entry_index), role_code, layout_code, model%import_weight_pack_hash, &
          model%import_weight_pack_bytes, span_hash, pack_materialized_hash, page_hash, page_word_count, page_words)
        page_bytes = 0_i8
        exec_page_byte_count = 0_i32
        if (page_hash > 0_i64 .and. page_word_count > 0_i32) then
          call pack_i32_words_to_le_bytes_cache(page_words, page_word_count, page_bytes, exec_page_byte_count)
        end if
        exec_page_data_offset = 0_i32
        if (exec_page_byte_count > 0_i32) then
          call append_pack_buffer_bytes_cache(page_bytes, exec_page_byte_count, exec_buffer, exec_buffer_offset, &
            exec_page_data_offset)
        end if
        call build_cuda_materialized_pack_tile_record_cache(pack_index, dispatch_offsets(entry_index), &
          dispatch_bytes(entry_index), role_code, layout_code, model%import_weight_pack_hash, &
          model%import_weight_pack_bytes, span_hash, pack_materialized_hash, tile_hash, tile_byte_count, tile_bytes)
        exec_tile_data_offset = 0_i32
        if (tile_byte_count > 0_i32) then
          call append_pack_buffer_bytes_cache(tile_bytes, tile_byte_count, exec_buffer, exec_buffer_offset, &
            exec_tile_data_offset)
        end if
      end if
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 0_i32, pack_index)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 4_i32, role_code)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 8_i32, layout_code)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 12_i32, exec_sample_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 16_i32, exec_sample_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 20_i32, page_word_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 24_i32, exec_page_data_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 28_i32, exec_page_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 32_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 36_i32, exec_tile_data_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 40_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 44_i32, entry_index)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 48_i32, dispatch_offsets(entry_index))
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 56_i32, dispatch_bytes(entry_index))
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 64_i32, span_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 72_i32, actual_sample_bytes)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 80_i32, page_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 88_i32, tile_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 96_i32, pack_materialized_hash)

      has_entries = .true.
    end do

    if (.not. has_entries) return

    call store_pack_buffer_i32_cache(exec_buffer, 0_i32, CUDA_EXEC_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(exec_buffer, 4_i32, CUDA_EXEC_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(exec_buffer, 8_i32, CUDA_EXEC_BUFFER_HEADER_BYTES)
    call store_pack_buffer_i32_cache(exec_buffer, 12_i32, CUDA_EXEC_BUFFER_ENTRY_BYTES)
    call store_pack_buffer_i32_cache(exec_buffer, 16_i32, exec_entry_count)
    call store_pack_buffer_i32_cache(exec_buffer, 20_i32, usage_count)
    call store_pack_buffer_i64_cache(exec_buffer, 24_i32, usage_hash)
    call store_pack_buffer_i64_cache(exec_buffer, 32_i32, usage_total_bytes)
    call store_pack_buffer_i64_cache(exec_buffer, 40_i32, first_pack_offset)
    call store_pack_buffer_i64_cache(exec_buffer, 48_i32, last_pack_offset)
    call store_pack_buffer_i64_cache(exec_buffer, 56_i32, last_pack_bytes)
    if (len_trim(pack_tile_buffer_path) > 0) then
      exec_path_bytes = min(len_trim(pack_tile_buffer_path), MAX_PATH_LEN)
      exec_path_offset = exec_buffer_offset
      call store_pack_buffer_i32_cache(exec_buffer, 64_i32, exec_path_bytes)
      call store_pack_buffer_i32_cache(exec_buffer, 68_i32, exec_path_offset)
      do exec_path_index = 1_i32, exec_path_bytes
        exec_buffer(exec_path_offset + exec_path_index) = int(iachar(pack_tile_buffer_path(exec_path_index:exec_path_index)), &
          kind=i8)
      end do
      exec_buffer_offset = exec_path_offset + exec_path_bytes
    end if

    parent_dir = parent_directory_path(exec_buffer_full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
    open(newunit=unit_id, file=trim(exec_buffer_full_path), status="replace", access="stream", &
      form="unformatted", action="write", iostat=ios)
    if (ios /= 0_i32) return
    write(unit_id, iostat=ios) exec_buffer(1:max(CUDA_EXEC_BUFFER_HEADER_BYTES, exec_buffer_offset))
    close(unit_id)
  end subroutine append_cuda_pack_span_cache_payload_from_model

  subroutine append_cuda_pack_span_cache_payload(cache_root, metadata, payload_text, payload_bytes, model)
    integer(i32), parameter                    :: CUDA_DISPATCH_BUFFER_MAGIC = int(z'53445A4D', kind=i32)
    integer(i32), parameter                    :: CUDA_DISPATCH_BUFFER_VERSION = 1_i32
    integer(i32), parameter                    :: CUDA_DISPATCH_BUFFER_HEADER_BYTES = 32_i32
    integer(i32), parameter                    :: CUDA_DISPATCH_BUFFER_ENTRY_BYTES = 16_i32
    integer(i32), parameter                    :: CUDA_USAGE_BUFFER_MAGIC = int(z'42555A4D', kind=i32)
    integer(i32), parameter                    :: CUDA_USAGE_BUFFER_VERSION = 2_i32
    integer(i32), parameter                    :: CUDA_USAGE_BUFFER_HEADER_BYTES = 72_i32
    integer(i32), parameter                    :: CUDA_SPAN_BUFFER_MAGIC = int(z'42535A4D', kind=i32)
    integer(i32), parameter                    :: CUDA_SPAN_BUFFER_VERSION = 1_i32
    integer(i32), parameter                    :: CUDA_SPAN_BUFFER_HEADER_BYTES = 32_i32
    integer(i32), parameter                    :: CUDA_SPAN_BUFFER_ENTRY_BYTES = 32_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_MAGIC = int(z'58455A4D', kind=i32)
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_VERSION = 3_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_HEADER_BYTES = 72_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_ENTRY_BYTES = 104_i32
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_PAYLOAD_BYTES_PER_ENTRY = 64_i32 + &
                                                  (MAX_CUDA_PACK_PAGE_WORDS * 4_i32) + MAX_CUDA_PACK_TILE_BYTES
    integer(i32), parameter                    :: CUDA_EXEC_BUFFER_CAPACITY = CUDA_EXEC_BUFFER_HEADER_BYTES + MAX_PATH_LEN + &
                                                  (MAX_IMPORT_STAGE_PACK_DISPATCH * (CUDA_EXEC_BUFFER_ENTRY_BYTES + &
                                                   CUDA_EXEC_BUFFER_PAYLOAD_BYTES_PER_ENTRY))
    character(len=*), intent(in)              :: cache_root
    type(artifact_metadata_record), intent(in) :: metadata
    character(len=*), intent(inout)           :: payload_text
    integer(i64), intent(inout)               :: payload_bytes
    type(model_state), intent(in), optional   :: model
    character(len=(MAX_PATH_LEN * 6) + 4096)  :: sidecar_payload
    character(len=(MAX_PATH_LEN * 2) + 4096)  :: tile_payload
    character(len=MAX_PATH_LEN)               :: sidecar_path
    character(len=MAX_PATH_LEN)               :: tile_path
    character(len=MAX_PATH_LEN)               :: dispatch_buffer_path
    character(len=MAX_PATH_LEN)               :: usage_buffer_path
    character(len=MAX_PATH_LEN)               :: span_buffer_path
    character(len=MAX_PATH_LEN)               :: exec_buffer_path
    character(len=MAX_PATH_LEN)               :: pack_tile_path
    character(len=MAX_PATH_LEN)               :: pack_tile_buffer_path
    character(len=MAX_PATH_LEN)               :: weight_payload_path
    character(len=MAX_PATH_LEN)               :: full_path
    character(len=MAX_PATH_LEN)               :: tile_full_path
    character(len=MAX_PATH_LEN)               :: dispatch_buffer_full_path
    character(len=MAX_PATH_LEN)               :: usage_buffer_full_path
    character(len=MAX_PATH_LEN)               :: span_buffer_full_path
    character(len=MAX_PATH_LEN)               :: exec_buffer_full_path
    character(len=MAX_PATH_LEN)               :: parent_dir
    character(len=MAX_PATH_LEN)               :: span_root
    character(len=MAX_PATH_LEN)               :: path_text
    character(len=MAX_PATH_LEN + 96)          :: entry_text
    character(len=128)                        :: hex_text
    character(len=320)                        :: field_text
    integer(i32)                              :: entry_index
    integer(i32)                              :: entry_limit
    integer(i32)                              :: role_code
    integer(i32)                              :: layout_code
    integer(i32)                              :: pack_index
    integer(i32)                              :: unit_id
    integer(i32)                              :: ios
    integer(i64)                              :: parsed_i64
    integer(i64)                              :: span_hash
    integer(i64)                              :: actual_sample_bytes
    integer(i64)                              :: entry_offset
    integer(i64)                              :: entry_bytes
    integer(i64)                              :: page_hash
    integer(i64)                              :: tile_hash
    integer(i8)                               :: sample_bytes(64)
    integer(i8)                               :: page_bytes(MAX_CUDA_PACK_PAGE_WORDS * 4_i32)
    integer(i32)                              :: page_word_count
    integer(i32)                              :: page_words(MAX_CUDA_PACK_PAGE_WORDS)
    integer(i32)                              :: tile_byte_count
    integer(i8)                               :: tile_bytes(MAX_CUDA_PACK_TILE_BYTES)
    integer(i8)                               :: dispatch_buffer(CUDA_DISPATCH_BUFFER_HEADER_BYTES + &
                                                     (MAX_IMPORT_STAGE_PACK_DISPATCH * CUDA_DISPATCH_BUFFER_ENTRY_BYTES))
    integer(i8)                               :: usage_buffer(CUDA_USAGE_BUFFER_HEADER_BYTES + MAX_PATH_LEN)
    integer(i8)                               :: span_buffer(CUDA_SPAN_BUFFER_HEADER_BYTES + &
                                                     (MAX_IMPORT_STAGE_PACK_DISPATCH * CUDA_SPAN_BUFFER_ENTRY_BYTES) + &
                                                     (MAX_IMPORT_STAGE_PACK_DISPATCH * 64_i32))
    integer(i8)                               :: exec_buffer(CUDA_EXEC_BUFFER_CAPACITY)
    integer(i32)                              :: dispatch_pack_indices(MAX_IMPORT_STAGE_PACK_DISPATCH)
    integer(i32)                              :: dispatch_buffer_offset
    integer(i32)                              :: span_buffer_offset
    integer(i32)                              :: dispatch_entry_count
    integer(i32)                              :: span_entry_count
    integer(i32)                              :: exec_entry_count
    integer(i32)                              :: span_record_offset
    integer(i32)                              :: span_data_offset
    integer(i32)                              :: exec_buffer_offset
    integer(i32)                              :: exec_record_offset
    integer(i32)                              :: exec_sample_count
    integer(i32)                              :: exec_sample_offset
    integer(i32)                              :: exec_page_byte_count
    integer(i32)                              :: exec_page_data_offset
    integer(i32)                              :: exec_tile_data_offset
    integer(i64)                              :: usage_hash
    integer(i64)                              :: usage_total_bytes
    integer(i64)                              :: first_pack_offset
    integer(i64)                              :: last_pack_offset
    integer(i64)                              :: last_pack_bytes
    integer(i64)                              :: pack_materialized_hash
    integer(i32)                              :: usage_count
    integer(i32)                              :: usage_path_bytes
    integer(i32)                              :: usage_path_offset
    integer(i32)                              :: usage_path_index
    integer(i32)                              :: exec_path_bytes
    integer(i32)                              :: exec_path_offset
    integer(i32)                              :: exec_path_index
    logical                                   :: exists
    logical                                   :: found_count
    logical                                   :: found_usage_hash
    logical                                   :: found_usage_count
    logical                                   :: found_usage_bytes
    logical                                   :: found_first_offset
    logical                                   :: found_last_offset
    logical                                   :: found_last_bytes
    logical                                   :: found_root
    logical                                   :: found_entry
    logical                                   :: found_path
    logical                                   :: found_sample_bytes
    logical                                   :: found_pack_tile
    logical                                   :: found_pack_tile_buffer
    logical                                   :: has_entries
    logical                                   :: has_tiles
    logical                                   :: parsed_dispatch

    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_CUDA) return
    if (len_trim(cache_root) == 0) return
    if (len_trim(metadata%payload_path) == 0) return
    if (present(model)) then
      if (metadata%stage_kind == MIZU_STAGE_PREFILL .or. metadata%stage_kind == MIZU_STAGE_DECODE) then
        call append_cuda_pack_span_cache_payload_from_model(cache_root, metadata, payload_text, payload_bytes, model)
        return
      end if
    end if

    span_root = ""
    call extract_payload_field_text_cache(payload_text, "pack_span_root=", span_root, found_root)
    if (.not. found_root) return

    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_dispatch_count=", field_text, found_count)
    if (.not. found_count) return
    if (.not. parse_i64_text_cache(field_text, parsed_i64)) return
    entry_limit = max(0_i32, min(MAX_IMPORT_STAGE_PACK_DISPATCH, int(parsed_i64, kind=i32)))
    if (entry_limit <= 0_i32) return

    sidecar_path = build_cuda_pack_span_cache_path(metadata%payload_path)
    if (len_trim(sidecar_path) == 0) return
    tile_path = build_cuda_pack_tile_cache_path(metadata%payload_path)
    dispatch_buffer_path = build_cuda_pack_dispatch_buffer_path(metadata%payload_path)
    usage_buffer_path = build_cuda_pack_usage_buffer_path(metadata%payload_path)
    span_buffer_path = build_cuda_pack_span_buffer_path(metadata%payload_path)

    payload_bytes = int(len_trim(payload_text) + 1_i64, kind=i64)

    full_path = join_cache_root_with_payload_path(cache_root, sidecar_path)
    if (len_trim(full_path) == 0) return
    inquire(file=trim(full_path), exist=exists)
    if (exists) return

    sidecar_payload = "kind=cuda_pack_span_cache_v4"
    has_entries = .false.
    tile_payload = "kind=cuda_pack_tile_cache_v1"
    has_tiles = .false.
    dispatch_pack_indices = 0_i32
    dispatch_buffer = 0_i8
    dispatch_entry_count = 0_i32
    usage_buffer = 0_i8
    span_buffer = 0_i8
    exec_buffer = 0_i8
    span_buffer_offset = CUDA_SPAN_BUFFER_HEADER_BYTES + &
      (MAX_IMPORT_STAGE_PACK_DISPATCH * CUDA_SPAN_BUFFER_ENTRY_BYTES)
    span_entry_count = 0_i32
    exec_buffer_offset = CUDA_EXEC_BUFFER_HEADER_BYTES + &
      (MAX_IMPORT_STAGE_PACK_DISPATCH * CUDA_EXEC_BUFFER_ENTRY_BYTES)
    exec_entry_count = 0_i32
    usage_hash = 0_i64
    usage_count = entry_limit
    usage_total_bytes = 0_i64
    usage_path_bytes = 0_i32
    usage_path_offset = CUDA_USAGE_BUFFER_HEADER_BYTES
    exec_path_bytes = 0_i32
    exec_path_offset = 0_i32
    first_pack_offset = 0_i64
    last_pack_offset = 0_i64
    last_pack_bytes = 0_i64
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_hash=", field_text, found_usage_hash)
    if (found_usage_hash) then
      usage_hash = positive_hash64_cache(trim(field_text))
    end if
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_count=", field_text, found_usage_count)
    if (found_usage_count) then
      if (parse_i64_text_cache(field_text, parsed_i64)) usage_count = max(0_i32, int(parsed_i64, kind=i32))
    end if
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_bytes=", field_text, found_usage_bytes)
    if (found_usage_bytes) then
      if (parse_i64_text_cache(field_text, parsed_i64)) usage_total_bytes = max(0_i64, parsed_i64)
    end if
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_first_offset=", field_text, found_first_offset)
    if (found_first_offset) then
      if (parse_i64_text_cache(field_text, parsed_i64)) first_pack_offset = max(0_i64, parsed_i64)
    end if
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_last_offset=", field_text, found_last_offset)
    if (found_last_offset) then
      if (parse_i64_text_cache(field_text, parsed_i64)) last_pack_offset = max(0_i64, parsed_i64)
    end if
    field_text = ""
    call extract_payload_field_text_cache(payload_text, "pack_use_last_bytes=", field_text, found_last_bytes)
    if (found_last_bytes) then
      if (parse_i64_text_cache(field_text, parsed_i64)) last_pack_bytes = max(0_i64, parsed_i64)
    end if
    pack_tile_path = ""
    pack_tile_buffer_path = ""
    call extract_payload_field_text_cache(payload_text, "pack_ref_tile_cache=", pack_tile_path, found_pack_tile)
    if (.not. found_pack_tile) then
      call extract_payload_field_text_cache(payload_text, "pack_tile_cache=", pack_tile_path, found_pack_tile)
    end if
    call extract_payload_field_text_cache(payload_text, "pack_ref_tile_buffer=", pack_tile_buffer_path, &
      found_pack_tile_buffer)
    if (.not. found_pack_tile_buffer) then
      call extract_payload_field_text_cache(payload_text, "pack_buffer=", pack_tile_buffer_path, &
        found_pack_tile_buffer)
    end if
    if (.not. found_pack_tile .and. present(model)) then
      weight_payload_path = build_cuda_weight_pack_payload_path(model, metadata%execution_route)
      if (len_trim(weight_payload_path) > 0) then
        pack_tile_path = build_cuda_weight_pack_tile_cache_path(trim(weight_payload_path))
        found_pack_tile = (len_trim(pack_tile_path) > 0)
        if (.not. found_pack_tile_buffer) then
          pack_tile_buffer_path = build_cuda_weight_pack_tile_buffer_path(trim(weight_payload_path))
          found_pack_tile_buffer = (len_trim(pack_tile_buffer_path) > 0)
        end if
      end if
    end if
    if (.not. found_pack_tile_buffer .and. metadata%stage_kind == MIZU_STAGE_MODEL_LOAD) then
      pack_tile_buffer_path = build_cuda_weight_pack_tile_buffer_path(metadata%payload_path)
      found_pack_tile_buffer = (len_trim(pack_tile_buffer_path) > 0)
    end if
    if (found_pack_tile .and. len_trim(pack_tile_path) > 0) then
      field_text = ""
      write(field_text, '(";pack_tile_cache=",A)') trim(pack_tile_path)
      call append_payload_fragment(sidecar_payload, trim(field_text))
    end if
    if (len_trim(span_buffer_path) > 0) then
      field_text = ""
      write(field_text, '(";span_buffer=",A)') trim(span_buffer_path)
      call append_payload_fragment(sidecar_payload, trim(field_text))
    end if
    if (len_trim(exec_buffer_path) > 0) then
      field_text = ""
      write(field_text, '(";exec_buffer=",A)') trim(exec_buffer_path)
      call append_payload_fragment(sidecar_payload, trim(field_text))
    end if
    if (len_trim(usage_buffer_path) > 0) then
      field_text = ""
      write(field_text, '(";usage_buffer=",A)') trim(usage_buffer_path)
      call append_payload_fragment(sidecar_payload, trim(field_text))
    end if
    if (len_trim(tile_path) > 0) then
      field_text = ""
      write(field_text, '(";tile_cache=",A)') trim(tile_path)
      call append_payload_fragment(sidecar_payload, trim(field_text))
    end if

    do entry_index = 1_i32, entry_limit
      write(field_text, '("pack_span",I0,"=")') entry_index
      entry_text = ""
      call extract_payload_field_text_cache(payload_text, trim(field_text), entry_text, found_entry)
      if (.not. found_entry) cycle

      path_text = ""
      call extract_pipe_field_cache(trim(entry_text), 1_i32, path_text, found_path)
      if (.not. found_path) cycle

      field_text = ""
      call extract_inline_numeric_field_cache(trim(entry_text), "sample_bytes=", field_text, found_sample_bytes)
      parsed_i64 = 0_i64
      if (found_sample_bytes) then
        if (.not. parse_i64_text_cache(field_text, parsed_i64)) parsed_i64 = 0_i64
      end if

      call resolve_import_span_record_cache(trim(span_root), trim(path_text), parsed_i64, span_hash, &
        actual_sample_bytes, sample_bytes)
      if (span_hash <= 0_i64) cycle

      role_code = 0_i32
      layout_code = 0_i32
      pack_index = 0_i32
      entry_offset = 0_i64
      entry_bytes = 0_i64
      parsed_dispatch = .false.
      write(field_text, '("pack_dispatch",I0,"=")') entry_index
      entry_text = ""
      call extract_payload_field_text_cache(payload_text, trim(field_text), entry_text, found_entry)
      if (found_entry) then
        call extract_payload_dispatch_entry_cache(trim(entry_text), pack_index, entry_offset, entry_bytes, role_code, &
          layout_code, parsed_dispatch)
      end if
      if (parsed_dispatch .and. (entry_bytes <= 0_i64 .or. role_code <= 0_i32 .or. layout_code <= 0_i32)) then
        write(field_text, '("pack_use",I0,"=")') entry_index
        entry_text = ""
        call extract_payload_field_text_cache(payload_text, trim(field_text), entry_text, found_entry)
        if (found_entry) then
          call extract_payload_usage_entry_cache(trim(entry_text), entry_offset, entry_bytes, role_code, layout_code, &
            parsed_dispatch)
        else
          parsed_dispatch = .false.
        end if
      end if

      field_text = ""
      write(field_text, '(";entry",I0,"_hash=",I0)') entry_index, span_hash
      call append_payload_fragment(sidecar_payload, trim(field_text))
      field_text = ""
      write(field_text, '(";entry",I0,"_bytes=",I0)') entry_index, actual_sample_bytes
      call append_payload_fragment(sidecar_payload, trim(field_text))
      if (actual_sample_bytes > 0_i64) then
        hex_text = ""
        call encode_bytes_as_hex(sample_bytes, int(min(actual_sample_bytes, int(size(sample_bytes), kind=i64)), kind=i32), &
          hex_text)
        field_text = ""
        write(field_text, '(";entry",I0,"_sample_hex=",A)') entry_index, trim(hex_text)
        call append_payload_fragment(sidecar_payload, trim(field_text))

        span_entry_count = span_entry_count + 1_i32
        span_record_offset = CUDA_SPAN_BUFFER_HEADER_BYTES + &
          ((span_entry_count - 1_i32) * CUDA_SPAN_BUFFER_ENTRY_BYTES)
        call append_pack_buffer_bytes_cache(sample_bytes, int(min(actual_sample_bytes, 64_i64), kind=i32), &
          span_buffer, span_buffer_offset, span_data_offset)
        call store_pack_buffer_i32_cache(span_buffer, span_record_offset + 0_i32, entry_index)
        call store_pack_buffer_i32_cache(span_buffer, span_record_offset + 4_i32, pack_index)
        call store_pack_buffer_i32_cache(span_buffer, span_record_offset + 8_i32, &
          int(min(actual_sample_bytes, 64_i64), kind=i32))
        call store_pack_buffer_i32_cache(span_buffer, span_record_offset + 12_i32, span_data_offset)
        call store_pack_buffer_i64_cache(span_buffer, span_record_offset + 16_i32, span_hash)
        call store_pack_buffer_i64_cache(span_buffer, span_record_offset + 24_i32, actual_sample_bytes)
      end if
      if (parsed_dispatch) then
        if (pack_index > 0_i32 .and. entry_index <= MAX_IMPORT_STAGE_PACK_DISPATCH) then
          dispatch_pack_indices(entry_index) = pack_index
        end if
        if (pack_index > 0_i32) then
          field_text = ""
          write(field_text, '(";entry",I0,"_pack=",I0)') entry_index, pack_index
          call append_payload_fragment(sidecar_payload, trim(field_text))
        end if
        call build_cuda_pack_page_record_cache(entry_offset, entry_bytes, role_code, layout_code, span_hash, &
          sample_bytes, actual_sample_bytes, page_hash, page_word_count, page_words)
        page_bytes = 0_i8
        exec_page_byte_count = 0_i32
        if (page_hash > 0_i64 .and. page_word_count > 0_i32) then
          field_text = ""
          write(field_text, '(";entry",I0,"_page_hash=",I0)') entry_index, page_hash
          call append_payload_fragment(sidecar_payload, trim(field_text))
          field_text = ""
          write(field_text, '(";entry",I0,"_page_words=",I0)') entry_index, page_word_count
          call append_payload_fragment(sidecar_payload, trim(field_text))
          hex_text = ""
          call encode_i32_words_as_hex_cache(page_words, page_word_count, hex_text)
          field_text = ""
          write(field_text, '(";entry",I0,"_page_hex=",A)') entry_index, trim(hex_text)
          call append_payload_fragment(sidecar_payload, trim(field_text))
          call pack_i32_words_to_le_bytes_cache(page_words, page_word_count, page_bytes, exec_page_byte_count)
        end if
        call build_cuda_pack_tile_record_cache(entry_offset, entry_bytes, role_code, layout_code, sample_bytes, &
          actual_sample_bytes, tile_hash, tile_byte_count, tile_bytes)
        if (tile_hash > 0_i64 .and. tile_byte_count > 0_i32) then
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_hash=",I0)') entry_index, tile_hash
          call append_payload_fragment(sidecar_payload, trim(field_text))
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_bytes=",I0)') entry_index, tile_byte_count
          call append_payload_fragment(sidecar_payload, trim(field_text))
          hex_text = ""
          call encode_bytes_as_hex(tile_bytes, tile_byte_count, hex_text)
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_hex=",A)') entry_index, trim(hex_text)
          call append_payload_fragment(sidecar_payload, trim(field_text))
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_hash=",I0)') entry_index, tile_hash
          call append_payload_fragment(tile_payload, trim(field_text))
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_bytes=",I0)') entry_index, tile_byte_count
          call append_payload_fragment(tile_payload, trim(field_text))
          field_text = ""
          write(field_text, '(";entry",I0,"_tile_hex=",A)') entry_index, trim(hex_text)
          call append_payload_fragment(tile_payload, trim(field_text))
          has_tiles = .true.
        end if
      end if
      exec_entry_count = exec_entry_count + 1_i32
      exec_record_offset = CUDA_EXEC_BUFFER_HEADER_BYTES + &
        ((exec_entry_count - 1_i32) * CUDA_EXEC_BUFFER_ENTRY_BYTES)
      exec_sample_count = int(min(actual_sample_bytes, 64_i64), kind=i32)
      exec_sample_offset = 0_i32
      if (exec_sample_count > 0_i32) then
        call append_pack_buffer_bytes_cache(sample_bytes, exec_sample_count, exec_buffer, exec_buffer_offset, &
          exec_sample_offset)
      end if
      exec_page_data_offset = 0_i32
      if (exec_page_byte_count > 0_i32) then
        call append_pack_buffer_bytes_cache(page_bytes, exec_page_byte_count, exec_buffer, exec_buffer_offset, &
          exec_page_data_offset)
      end if
      exec_tile_data_offset = 0_i32
      if (tile_byte_count > 0_i32) then
        call append_pack_buffer_bytes_cache(tile_bytes, tile_byte_count, exec_buffer, exec_buffer_offset, &
          exec_tile_data_offset)
      end if
      pack_materialized_hash = 0_i64
      if (pack_index > 0_i32 .and. present(model)) then
        if (model%import_weight_pack_hash > 0_i64) then
          pack_materialized_hash = build_cuda_materialized_pack_seed_cache("cuda_weight_pack_entry", pack_index, &
            entry_offset, entry_bytes, role_code, layout_code, model%import_weight_pack_hash, &
            model%import_weight_pack_bytes, span_hash)
          call build_cuda_materialized_pack_page_record_cache(pack_index, entry_offset, entry_bytes, role_code, &
            layout_code, model%import_weight_pack_hash, model%import_weight_pack_bytes, span_hash, &
            pack_materialized_hash, page_hash, page_word_count, page_words)
          page_bytes = 0_i8
          exec_page_byte_count = 0_i32
          if (page_hash > 0_i64 .and. page_word_count > 0_i32) then
            call pack_i32_words_to_le_bytes_cache(page_words, page_word_count, page_bytes, exec_page_byte_count)
          end if
          exec_page_data_offset = 0_i32
          if (exec_page_byte_count > 0_i32) then
            call append_pack_buffer_bytes_cache(page_bytes, exec_page_byte_count, exec_buffer, exec_buffer_offset, &
              exec_page_data_offset)
          end if
          call build_cuda_materialized_pack_tile_record_cache(pack_index, entry_offset, entry_bytes, role_code, &
            layout_code, model%import_weight_pack_hash, model%import_weight_pack_bytes, span_hash, &
            pack_materialized_hash, tile_hash, tile_byte_count, tile_bytes)
          exec_tile_data_offset = 0_i32
          if (tile_byte_count > 0_i32) then
            call append_pack_buffer_bytes_cache(tile_bytes, tile_byte_count, exec_buffer, exec_buffer_offset, &
              exec_tile_data_offset)
          end if
        end if
      end if
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 0_i32, pack_index)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 4_i32, role_code)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 8_i32, layout_code)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 12_i32, exec_sample_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 16_i32, exec_sample_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 20_i32, page_word_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 24_i32, exec_page_data_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 28_i32, exec_page_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 32_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 36_i32, exec_tile_data_offset)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 40_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(exec_buffer, exec_record_offset + 44_i32, entry_index)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 48_i32, entry_offset)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 56_i32, entry_bytes)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 64_i32, span_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 72_i32, actual_sample_bytes)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 80_i32, page_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 88_i32, tile_hash)
      call store_pack_buffer_i64_cache(exec_buffer, exec_record_offset + 96_i32, pack_materialized_hash)
      has_entries = .true.
    end do

    if (.not. has_entries) return

    dispatch_buffer_offset = CUDA_DISPATCH_BUFFER_HEADER_BYTES
    do entry_index = 1_i32, entry_limit
      if (dispatch_pack_indices(entry_index) <= 0_i32) cycle
      dispatch_entry_count = dispatch_entry_count + 1_i32
      call store_pack_buffer_i32_cache(dispatch_buffer, dispatch_buffer_offset + 0_i32, dispatch_pack_indices(entry_index))
      call store_pack_buffer_i32_cache(dispatch_buffer, dispatch_buffer_offset + 4_i32, entry_index)
      call store_pack_buffer_i64_cache(dispatch_buffer, dispatch_buffer_offset + 8_i32, 0_i64)
      dispatch_buffer_offset = dispatch_buffer_offset + CUDA_DISPATCH_BUFFER_ENTRY_BYTES
    end do
    call store_pack_buffer_i32_cache(dispatch_buffer, 0_i32, CUDA_DISPATCH_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(dispatch_buffer, 4_i32, CUDA_DISPATCH_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(dispatch_buffer, 8_i32, CUDA_DISPATCH_BUFFER_HEADER_BYTES)
    call store_pack_buffer_i32_cache(dispatch_buffer, 12_i32, CUDA_DISPATCH_BUFFER_ENTRY_BYTES)
    call store_pack_buffer_i32_cache(dispatch_buffer, 16_i32, dispatch_entry_count)
    call store_pack_buffer_i32_cache(dispatch_buffer, 20_i32, dispatch_entry_count)
    call store_pack_buffer_i64_cache(dispatch_buffer, 24_i32, usage_hash)
    call store_pack_buffer_i32_cache(usage_buffer, 0_i32, CUDA_USAGE_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(usage_buffer, 4_i32, CUDA_USAGE_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(usage_buffer, 8_i32, CUDA_USAGE_BUFFER_HEADER_BYTES)
    call store_pack_buffer_i32_cache(usage_buffer, 12_i32, usage_count)
    call store_pack_buffer_i32_cache(usage_buffer, 16_i32, dispatch_entry_count)
    call store_pack_buffer_i32_cache(usage_buffer, 20_i32, entry_limit)
    call store_pack_buffer_i64_cache(usage_buffer, 24_i32, usage_total_bytes)
    call store_pack_buffer_i64_cache(usage_buffer, 32_i32, first_pack_offset)
    call store_pack_buffer_i64_cache(usage_buffer, 40_i32, last_pack_offset)
    call store_pack_buffer_i64_cache(usage_buffer, 48_i32, last_pack_bytes)
    call store_pack_buffer_i64_cache(usage_buffer, 56_i32, usage_hash)
    if (found_pack_tile_buffer .and. len_trim(pack_tile_buffer_path) > 0) then
      usage_path_bytes = min(len_trim(pack_tile_buffer_path), MAX_PATH_LEN)
      call store_pack_buffer_i32_cache(usage_buffer, 64_i32, usage_path_bytes)
      call store_pack_buffer_i32_cache(usage_buffer, 68_i32, usage_path_offset)
      do usage_path_index = 1_i32, usage_path_bytes
        usage_buffer(usage_path_offset + usage_path_index) = int(iachar(pack_tile_buffer_path(usage_path_index:usage_path_index)), &
          kind=i8)
      end do
    end if
    call store_pack_buffer_i32_cache(span_buffer, 0_i32, CUDA_SPAN_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(span_buffer, 4_i32, CUDA_SPAN_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(span_buffer, 8_i32, CUDA_SPAN_BUFFER_HEADER_BYTES)
    call store_pack_buffer_i32_cache(span_buffer, 12_i32, CUDA_SPAN_BUFFER_ENTRY_BYTES)
    call store_pack_buffer_i32_cache(span_buffer, 16_i32, span_entry_count)
    call store_pack_buffer_i32_cache(span_buffer, 20_i32, entry_limit)
    call store_pack_buffer_i64_cache(span_buffer, 24_i32, usage_hash)
    call store_pack_buffer_i32_cache(exec_buffer, 0_i32, CUDA_EXEC_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(exec_buffer, 4_i32, CUDA_EXEC_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(exec_buffer, 8_i32, CUDA_EXEC_BUFFER_HEADER_BYTES)
    call store_pack_buffer_i32_cache(exec_buffer, 12_i32, CUDA_EXEC_BUFFER_ENTRY_BYTES)
    call store_pack_buffer_i32_cache(exec_buffer, 16_i32, exec_entry_count)
    call store_pack_buffer_i32_cache(exec_buffer, 20_i32, usage_count)
    call store_pack_buffer_i64_cache(exec_buffer, 24_i32, usage_hash)
    call store_pack_buffer_i64_cache(exec_buffer, 32_i32, usage_total_bytes)
    call store_pack_buffer_i64_cache(exec_buffer, 40_i32, first_pack_offset)
    call store_pack_buffer_i64_cache(exec_buffer, 48_i32, last_pack_offset)
    call store_pack_buffer_i64_cache(exec_buffer, 56_i32, last_pack_bytes)
    if (len_trim(pack_tile_buffer_path) > 0) then
      exec_path_bytes = min(len_trim(pack_tile_buffer_path), MAX_PATH_LEN)
      exec_path_offset = exec_buffer_offset
      call store_pack_buffer_i32_cache(exec_buffer, 64_i32, exec_path_bytes)
      call store_pack_buffer_i32_cache(exec_buffer, 68_i32, exec_path_offset)
      do exec_path_index = 1_i32, exec_path_bytes
        exec_buffer(exec_path_offset + exec_path_index) = int(iachar(pack_tile_buffer_path(exec_path_index:exec_path_index)), &
          kind=i8)
      end do
      exec_buffer_offset = exec_path_offset + exec_path_bytes
    end if

    field_text = ""
    write(field_text, '(";entry_count=",I0)') entry_limit
    call append_payload_fragment(sidecar_payload, trim(field_text))
    if (has_tiles) then
      field_text = ""
      write(field_text, '(";entry_count=",I0)') entry_limit
      call append_payload_fragment(tile_payload, trim(field_text))
    end if

    parent_dir = parent_directory_path(full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)

    open(newunit=unit_id, file=trim(full_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return
    write(unit_id, "(A)", iostat=ios) trim(sidecar_payload)
    close(unit_id)

    if (len_trim(dispatch_buffer_path) > 0) then
      dispatch_buffer_full_path = join_cache_root_with_payload_path(cache_root, dispatch_buffer_path)
      if (len_trim(dispatch_buffer_full_path) > 0) then
        parent_dir = parent_directory_path(dispatch_buffer_full_path)
        if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
        open(newunit=unit_id, file=trim(dispatch_buffer_full_path), status="replace", access="stream", &
          form="unformatted", action="write", iostat=ios)
        if (ios == 0_i32) then
          write(unit_id, iostat=ios) dispatch_buffer(1:dispatch_buffer_offset)
          close(unit_id)
        end if
      end if
    end if

    if (len_trim(usage_buffer_path) > 0) then
      usage_buffer_full_path = join_cache_root_with_payload_path(cache_root, usage_buffer_path)
      if (len_trim(usage_buffer_full_path) > 0) then
        parent_dir = parent_directory_path(usage_buffer_full_path)
        if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
        open(newunit=unit_id, file=trim(usage_buffer_full_path), status="replace", access="stream", &
          form="unformatted", action="write", iostat=ios)
        if (ios == 0_i32) then
          write(unit_id, iostat=ios) usage_buffer(1:max(CUDA_USAGE_BUFFER_HEADER_BYTES, usage_path_offset + usage_path_bytes))
          close(unit_id)
        end if
      end if
    end if

    if (len_trim(span_buffer_path) > 0 .and. span_entry_count > 0_i32) then
      span_buffer_full_path = join_cache_root_with_payload_path(cache_root, span_buffer_path)
      if (len_trim(span_buffer_full_path) > 0) then
        parent_dir = parent_directory_path(span_buffer_full_path)
        if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
        open(newunit=unit_id, file=trim(span_buffer_full_path), status="replace", access="stream", &
          form="unformatted", action="write", iostat=ios)
        if (ios == 0_i32) then
          write(unit_id, iostat=ios) span_buffer(1:span_buffer_offset)
          close(unit_id)
        end if
      end if
    end if

    if (len_trim(exec_buffer_path) > 0 .and. exec_entry_count > 0_i32) then
      exec_buffer_full_path = join_cache_root_with_payload_path(cache_root, exec_buffer_path)
      if (len_trim(exec_buffer_full_path) > 0) then
        parent_dir = parent_directory_path(exec_buffer_full_path)
        if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
        open(newunit=unit_id, file=trim(exec_buffer_full_path), status="replace", access="stream", &
          form="unformatted", action="write", iostat=ios)
        if (ios == 0_i32) then
          write(unit_id, iostat=ios) exec_buffer(1:max(CUDA_EXEC_BUFFER_HEADER_BYTES, exec_buffer_offset))
          close(unit_id)
        end if
      end if
    end if

    if (.not. has_tiles) return
    tile_full_path = join_cache_root_with_payload_path(cache_root, tile_path)
    if (len_trim(tile_full_path) == 0) return
    parent_dir = parent_directory_path(tile_full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
    open(newunit=unit_id, file=trim(tile_full_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return
    write(unit_id, "(A)", iostat=ios) trim(tile_payload)
    close(unit_id)
  end subroutine append_cuda_pack_span_cache_payload

  function build_cuda_pack_span_cache_path(payload_path) result(sidecar_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: sidecar_path

    sidecar_path = ""
    if (len_trim(payload_path) == 0) return
    write(sidecar_path, '(A,".spancache")') trim(payload_path)
  end function build_cuda_pack_span_cache_path

  function build_cuda_pack_tile_cache_path(payload_path) result(tile_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: tile_path

    tile_path = ""
    if (len_trim(payload_path) == 0) return
    write(tile_path, '(A,".tilecache")') trim(payload_path)
  end function build_cuda_pack_tile_cache_path

  function build_cuda_pack_dispatch_buffer_path(payload_path) result(buffer_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: buffer_path

    buffer_path = ""
    if (len_trim(payload_path) == 0) return
    write(buffer_path, '(A,".dispatchbuffer")') trim(payload_path)
  end function build_cuda_pack_dispatch_buffer_path

  function build_cuda_pack_usage_buffer_path(payload_path) result(buffer_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: buffer_path

    buffer_path = ""
    if (len_trim(payload_path) == 0) return
    write(buffer_path, '(A,".usagebuffer")') trim(payload_path)
  end function build_cuda_pack_usage_buffer_path

  function build_cuda_pack_span_buffer_path(payload_path) result(buffer_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: buffer_path

    buffer_path = ""
    if (len_trim(payload_path) == 0) return
    write(buffer_path, '(A,".spanbuffer")') trim(payload_path)
  end function build_cuda_pack_span_buffer_path

  function build_cuda_pack_execution_buffer_path(payload_path) result(buffer_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: buffer_path

    buffer_path = ""
    if (len_trim(payload_path) == 0) return
    write(buffer_path, '(A,".execbuffer")') trim(payload_path)
  end function build_cuda_pack_execution_buffer_path

  function build_cuda_weight_pack_tile_cache_path(payload_path) result(tile_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: tile_path

    tile_path = ""
    if (len_trim(payload_path) == 0) return
    write(tile_path, '(A,".packtiles")') trim(payload_path)
  end function build_cuda_weight_pack_tile_cache_path

  function build_cuda_weight_pack_tile_buffer_path(payload_path) result(tile_path)
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: tile_path

    tile_path = ""
    if (len_trim(payload_path) == 0) return
    write(tile_path, '(A,".packbuffer")') trim(payload_path)
  end function build_cuda_weight_pack_tile_buffer_path

  function build_cuda_weight_pack_payload_path(model, execution_route) result(payload_path)
    type(model_state), intent(in) :: model
    integer(i32), intent(in)      :: execution_route
    type(model_manifest)          :: manifest
    type(weight_cache_key)        :: key
    type(runtime_state), pointer  :: runtime
    type(invalidation_version_fields) :: key_versions
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text
    character(len=MAX_NAME_LEN)      :: device_key
    character(len=MAX_NAME_LEN)      :: fingerprint_token
    character(len=MAX_PATH_LEN)      :: payload_path
    integer(i32)                     :: owner_status

    payload_path = ""
    if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
    if (model%import_weight_pack_hash == 0_i64) return

    call populate_manifest_identity(model, manifest)
    call resolve_model_owner_runtime(model, runtime, owner_status)
    if (owner_status == MIZU_STATUS_OK) then
      call resolve_cache_key_identity(runtime, manifest, MIZU_BACKEND_FAMILY_CUDA, execution_route, &
        device_key, key_versions)
    else
      call initialize_cache_key_identity(manifest, MIZU_BACKEND_FAMILY_CUDA, execution_route, &
        device_key, key_versions)
    end if
    call build_weight_cache_key(manifest, trim(device_key), "logical", MIZU_BACKEND_FAMILY_CUDA, &
      execution_route, key, key_versions)
    candidate_key_text = append_import_pack_identity(trim(key%key_text), model)
    fingerprint_token = build_artifact_fingerprint_token(trim(candidate_key_text))
    payload_path = build_artifact_payload_path(MIZU_STAGE_MODEL_LOAD, MIZU_BACKEND_FAMILY_CUDA, execution_route, &
      trim(fingerprint_token))
  end function build_cuda_weight_pack_payload_path

  subroutine append_cuda_weight_pack_reference_payload(payload_text, payload_bytes, stage_kind, execution_route, &
                                                       model, current_payload_path)
    character(len=*), intent(inout) :: payload_text
    integer(i64), intent(inout)     :: payload_bytes
    integer(i32), intent(in)        :: stage_kind
    integer(i32), intent(in)        :: execution_route
    type(model_state), intent(in)   :: model
    character(len=*), intent(in)    :: current_payload_path
    character(len=MAX_PATH_LEN)     :: weight_payload_path
    character(len=MAX_PATH_LEN)     :: weight_tile_buffer_path
    character(len=MAX_PATH_LEN + 96) :: field_text

    if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
    if (model%import_weight_pack_hash == 0_i64) return

    if (stage_kind == MIZU_STAGE_MODEL_LOAD .and. len_trim(current_payload_path) > 0) then
      weight_payload_path = trim(current_payload_path)
    else
      weight_payload_path = build_cuda_weight_pack_payload_path(model, execution_route)
    end if
    if (len_trim(weight_payload_path) == 0) return

    weight_tile_buffer_path = build_cuda_weight_pack_tile_buffer_path(trim(weight_payload_path))

    if (stage_kind == MIZU_STAGE_MODEL_LOAD) then
      if (len_trim(weight_tile_buffer_path) > 0) then
        field_text = ""
        write(field_text, '(";pack_buffer=",A)') trim(weight_tile_buffer_path)
        call append_payload_fragment(payload_text, trim(field_text))
      end if
    else if (stage_kind == MIZU_STAGE_PROJECTOR) then
      field_text = ""
      write(field_text, '(";pack_ref_artifact=",A)') trim(weight_payload_path)
      call append_payload_fragment(payload_text, trim(field_text))
      if (len_trim(weight_tile_buffer_path) > 0) then
        field_text = ""
        write(field_text, '(";pack_ref_tile_buffer=",A)') trim(weight_tile_buffer_path)
        call append_payload_fragment(payload_text, trim(field_text))
      end if
    end if

    payload_bytes = int(len_trim(payload_text) + 1_i64, kind=i64)
  end subroutine append_cuda_weight_pack_reference_payload

  subroutine materialize_cuda_weight_pack_tile_cache(cache_root, metadata, model)
    character(len=*), intent(in)                 :: cache_root
    type(artifact_metadata_record), intent(in)   :: metadata
    type(model_state), intent(in)                :: model
    integer(i32), parameter                      :: CUDA_PACK_BUFFER_MAGIC = int(z'42505A4D', kind=i32)
    integer(i32), parameter                      :: CUDA_PACK_BUFFER_VERSION = 2_i32
    integer(i32), parameter                      :: CUDA_PACK_BUFFER_HEADER_BYTES = 32_i32
    integer(i32), parameter                      :: CUDA_PACK_BUFFER_ENTRY_BYTES = 104_i32
    character(len=MAX_PATH_LEN)                  :: pack_buffer_path
    character(len=MAX_PATH_LEN)                  :: pack_buffer_full_path
    character(len=MAX_PATH_LEN)                  :: parent_dir
    character(len=MAX_PATH_LEN + 96)             :: field_text
    integer(i32)                                 :: tensor_index
    integer(i32)                                 :: pack_index
    integer(i32)                                 :: non_projector_count
    integer(i32)                                 :: role_code
    integer(i32)                                 :: layout_code
    integer(i32)                                 :: page_word_count
    integer(i32)                                 :: page_byte_count
    integer(i32)                                 :: tile_byte_count
    integer(i32)                                 :: page_data_offset
    integer(i32)                                 :: tile_data_offset
    integer(i32)                                 :: pack_record_offset
    integer(i32)                                 :: buffer_offset
    integer(i32)                                 :: buffer_payload_bytes
    integer(i32)                                 :: unit_id
    integer(i32)                                 :: ios
    integer(i64)                                 :: tensor_bytes
    integer(i64)                                 :: pack_offset
    integer(i64)                                 :: pack_total_bytes
    integer(i64)                                 :: pack_materialized_hash
    integer(i64)                                 :: span_hash
    integer(i64)                                 :: actual_sample_bytes
    integer(i64)                                 :: page_hash
    integer(i64)                                 :: tile_hash
    integer(i8)                                  :: sample_bytes(64)
    integer(i32)                                 :: page_words(MAX_CUDA_PACK_PAGE_WORDS)
    integer(i8)                                  :: page_bytes(MAX_CUDA_PACK_PAGE_WORDS * 4_i32)
    integer(i8)                                  :: tile_bytes(MAX_CUDA_PACK_TILE_BYTES)
    integer(i8), allocatable                     :: pack_buffer(:)
    logical                                      :: exists

    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_CUDA) return
    if (metadata%stage_kind /= MIZU_STAGE_MODEL_LOAD) return
    if (len_trim(cache_root) == 0) return
    if (len_trim(metadata%payload_path) == 0) return
    if (.not. allocated(model%import_tensors)) return

    pack_buffer_path = build_cuda_weight_pack_tile_buffer_path(metadata%payload_path)
    if (len_trim(pack_buffer_path) == 0) return
    pack_buffer_full_path = join_cache_root_with_payload_path(cache_root, pack_buffer_path)
    if (len_trim(pack_buffer_full_path) == 0) return
    inquire(file=trim(pack_buffer_full_path), exist=exists)
    if (exists) return

    pack_index = 0_i32
    non_projector_count = 0_i32
    pack_offset = 0_i64
    pack_total_bytes = 0_i64
    buffer_offset = 0_i32

    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle
      non_projector_count = non_projector_count + 1_i32
      pack_total_bytes = align_import_bytes(pack_total_bytes + tensor_bytes)
    end do

    if (pack_total_bytes <= 0_i64) return
    buffer_offset = CUDA_PACK_BUFFER_HEADER_BYTES + (non_projector_count * CUDA_PACK_BUFFER_ENTRY_BYTES)
    allocate(pack_buffer(max(1_i32, buffer_offset + &
      (non_projector_count * ((MAX_CUDA_PACK_PAGE_WORDS * 4_i32) + MAX_CUDA_PACK_TILE_BYTES)))))
    pack_buffer = 0_i8

    do tensor_index = 1_i32, int(size(model%import_tensors), kind=i32)
      if (import_tensor_belongs_to_projector(model%import_tensors(tensor_index), model)) cycle

      tensor_bytes = estimate_import_tensor_bytes(model%import_tensors(tensor_index))
      if (tensor_bytes <= 0_i64) cycle

      pack_index = pack_index + 1_i32
      call resolve_import_span_record_cache(trim(build_import_bundle_root_path(model)), &
        trim(model%import_tensors(tensor_index)%source_path), IMPORT_STAGE_SPAN_SAMPLE_BYTES, span_hash, &
        actual_sample_bytes, sample_bytes, model%import_tensors(tensor_index)%source_offset, tensor_bytes)

      field_text = ""
      write(field_text, '(";pack",I0,"_offset=",I0)') pack_index, pack_offset
      field_text = ""
      write(field_text, '(";pack",I0,"_bytes=",I0)') pack_index, tensor_bytes
      if (span_hash > 0_i64) then
        field_text = ""
        write(field_text, '(";pack",I0,"_span_hash=",I0)') pack_index, span_hash
      end if
      if (actual_sample_bytes > 0_i64) then
        field_text = ""
        write(field_text, '(";pack",I0,"_span_bytes=",I0)') pack_index, actual_sample_bytes
      end if

      role_code = import_tensor_role_code(trim(model%import_tensors(tensor_index)%tensor_role))
      layout_code = import_tensor_layout_code(trim(model%import_tensors(tensor_index)%layout_name))
      pack_record_offset = CUDA_PACK_BUFFER_HEADER_BYTES + ((pack_index - 1_i32) * CUDA_PACK_BUFFER_ENTRY_BYTES)
      page_hash = 0_i64
      page_word_count = 0_i32
      page_byte_count = 0_i32
      page_data_offset = 0_i32
      tile_hash = 0_i64
      tile_byte_count = 0_i32
      tile_data_offset = 0_i32
      pack_materialized_hash = build_cuda_materialized_pack_seed_cache("cuda_weight_pack_entry", pack_index, &
        pack_offset, tensor_bytes, role_code, layout_code, max(1_i64, model%import_weight_pack_hash), &
        pack_total_bytes, span_hash)
      field_text = ""
      write(field_text, '(";pack",I0,"_role=",I0)') pack_index, role_code
      field_text = ""
      write(field_text, '(";pack",I0,"_layout=",I0)') pack_index, layout_code
      field_text = ""
      write(field_text, '(";pack",I0,"_materialized_hash=",I0)') pack_index, pack_materialized_hash
      field_text = ""
      write(field_text, '(";pack",I0,"_page_source=pack_materialized_v2")') pack_index
      field_text = ""
      write(field_text, '(";pack",I0,"_tile_source=pack_materialized_v2")') pack_index

      if (span_hash > 0_i64) then
        call build_cuda_materialized_pack_page_record_cache(pack_index, pack_offset, tensor_bytes, role_code, &
          layout_code, max(1_i64, model%import_weight_pack_hash), pack_total_bytes, span_hash, &
          pack_materialized_hash, page_hash, page_word_count, page_words)
        if (page_hash > 0_i64 .and. page_word_count > 0_i32) then
          call pack_i32_words_to_le_bytes_cache(page_words, page_word_count, page_bytes, page_byte_count)
          call append_pack_buffer_bytes_cache(page_bytes, page_byte_count, pack_buffer, buffer_offset, page_data_offset)
        end if

        call build_cuda_materialized_pack_tile_record_cache(pack_index, pack_offset, tensor_bytes, role_code, &
          layout_code, max(1_i64, model%import_weight_pack_hash), pack_total_bytes, span_hash, &
          pack_materialized_hash, tile_hash, tile_byte_count, tile_bytes)
        if (tile_hash > 0_i64 .and. tile_byte_count > 0_i32) then
          call append_pack_buffer_bytes_cache(tile_bytes, tile_byte_count, pack_buffer, buffer_offset, tile_data_offset)
        end if
      end if

      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 0_i32, pack_index)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 4_i32, role_code)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 8_i32, layout_code)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 12_i32, page_word_count)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 16_i32, page_data_offset)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 20_i32, page_byte_count)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 24_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 28_i32, tile_data_offset)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 32_i32, tile_byte_count)
      call store_pack_buffer_i32_cache(pack_buffer, pack_record_offset + 36_i32, 0_i32)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 40_i32, pack_offset)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 48_i32, tensor_bytes)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 56_i32, span_hash)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 64_i32, actual_sample_bytes)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 72_i32, page_hash)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 80_i32, tile_hash)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 88_i32, pack_materialized_hash)
      call store_pack_buffer_i64_cache(pack_buffer, pack_record_offset + 96_i32, &
        model%import_tensors(tensor_index)%source_offset)

      pack_offset = align_import_bytes(pack_offset + tensor_bytes)
    end do

    if (pack_index <= 0_i32) return
    buffer_payload_bytes = max(0_i32, buffer_offset - (CUDA_PACK_BUFFER_HEADER_BYTES + (pack_index * CUDA_PACK_BUFFER_ENTRY_BYTES)))
    call store_pack_buffer_i32_cache(pack_buffer, 0_i32, CUDA_PACK_BUFFER_MAGIC)
    call store_pack_buffer_i32_cache(pack_buffer, 4_i32, CUDA_PACK_BUFFER_VERSION)
    call store_pack_buffer_i32_cache(pack_buffer, 8_i32, &
      CUDA_PACK_BUFFER_HEADER_BYTES + (pack_index * CUDA_PACK_BUFFER_ENTRY_BYTES))
    call store_pack_buffer_i32_cache(pack_buffer, 12_i32, CUDA_PACK_BUFFER_ENTRY_BYTES)
    call store_pack_buffer_i32_cache(pack_buffer, 16_i32, pack_index)
    call store_pack_buffer_i32_cache(pack_buffer, 20_i32, buffer_payload_bytes)
    call store_pack_buffer_i32_cache(pack_buffer, 24_i32, 0_i32)
    call store_pack_buffer_i32_cache(pack_buffer, 28_i32, 0_i32)

    parent_dir = parent_directory_path(pack_buffer_full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)
    open(newunit=unit_id, file=trim(pack_buffer_full_path), status="replace", access="stream", &
      form="unformatted", action="write", iostat=ios)
    if (ios /= 0_i32) return
    if (buffer_offset > 0_i32) then
      write(unit_id, iostat=ios) pack_buffer(1:buffer_offset)
    end if
    close(unit_id)
  end subroutine materialize_cuda_weight_pack_tile_cache

  subroutine store_pack_buffer_i32_cache(pack_buffer, byte_offset, value)
    integer(i8), intent(inout) :: pack_buffer(:)
    integer(i32), intent(in)   :: byte_offset
    integer(i32), intent(in)   :: value
    integer(i32)               :: byte_index
    integer(i32)               :: target_index

    do byte_index = 0_i32, 3_i32
      target_index = byte_offset + byte_index + 1_i32
      if (target_index > int(size(pack_buffer), kind=i32)) return
      pack_buffer(target_index) = int(iand(shiftr(value, 8 * byte_index), int(z'FF', kind=i32)), kind=i8)
    end do
  end subroutine store_pack_buffer_i32_cache

  subroutine store_pack_buffer_i64_cache(pack_buffer, byte_offset, value)
    integer(i8), intent(inout) :: pack_buffer(:)
    integer(i32), intent(in)   :: byte_offset
    integer(i64), intent(in)   :: value
    integer(i32)               :: byte_index
    integer(i32)               :: target_index

    do byte_index = 0_i32, 7_i32
      target_index = byte_offset + byte_index + 1_i32
      if (target_index > int(size(pack_buffer), kind=i32)) return
      pack_buffer(target_index) = int(iand(shiftr(value, 8 * byte_index), int(z'FF', kind=i64)), kind=i8)
    end do
  end subroutine store_pack_buffer_i64_cache

  subroutine pack_i32_words_to_le_bytes_cache(word_values, word_count, byte_values, byte_count)
    integer(i32), intent(in)    :: word_values(:)
    integer(i32), intent(in)    :: word_count
    integer(i8), intent(out)    :: byte_values(:)
    integer(i32), intent(out)   :: byte_count
    integer(i32)                :: encode_word_count
    integer(i32)                :: word_index
    integer(i32)                :: byte_index
    integer(i32)                :: word_value

    byte_values = 0_i8
    encode_word_count = max(0_i32, min(word_count, min(int(size(word_values), kind=i32), int(size(byte_values), kind=i32) / 4_i32)))
    byte_count = encode_word_count * 4_i32
    do word_index = 1_i32, encode_word_count
      word_value = word_values(word_index)
      do byte_index = 0_i32, 3_i32
        byte_values(((word_index - 1_i32) * 4_i32) + byte_index + 1_i32) = &
          int(iand(shiftr(word_value, 8 * byte_index), int(z'FF', kind=i32)), kind=i8)
      end do
    end do
  end subroutine pack_i32_words_to_le_bytes_cache

  subroutine append_pack_buffer_bytes_cache(source_bytes, source_count, pack_buffer, buffer_offset, data_offset)
    integer(i8), intent(in)     :: source_bytes(:)
    integer(i32), intent(in)    :: source_count
    integer(i8), intent(inout)  :: pack_buffer(:)
    integer(i32), intent(inout) :: buffer_offset
    integer(i32), intent(out)   :: data_offset
    integer(i32)                :: write_count

    data_offset = 0_i32
    if (source_count <= 0_i32) return
    write_count = max(0_i32, min(source_count, int(size(source_bytes), kind=i32)))
    if (write_count <= 0_i32) return
    if ((buffer_offset + write_count) > int(size(pack_buffer), kind=i32)) return
    data_offset = buffer_offset
    pack_buffer(buffer_offset + 1_i32:buffer_offset + write_count) = source_bytes(1:write_count)
    buffer_offset = buffer_offset + write_count
  end subroutine append_pack_buffer_bytes_cache

  subroutine extract_payload_dispatch_entry_cache(dispatch_entry, pack_index, pack_offset, pack_bytes, role_code, &
                                                  layout_code, parsed_ok)
    character(len=*), intent(in) :: dispatch_entry
    integer(i32), intent(out)    :: pack_index
    integer(i64), intent(out)    :: pack_offset
    integer(i64), intent(out)    :: pack_bytes
    integer(i32), intent(out)    :: role_code
    integer(i32), intent(out)    :: layout_code
    logical, intent(out)         :: parsed_ok
    character(len=64)            :: pack_index_text
    character(len=64)            :: offset_text
    character(len=64)            :: bytes_text
    character(len=64)            :: role_text
    character(len=64)            :: layout_text
    integer(i64)                 :: parsed_i64
    logical                      :: found_pack_index
    logical                      :: found_offset
    logical                      :: found_bytes
    logical                      :: found_role
    logical                      :: found_layout

    pack_index = 0_i32
    pack_offset = 0_i64
    pack_bytes = 0_i64
    role_code = 0_i32
    layout_code = 0_i32
    parsed_ok = .false.
    if (len_trim(dispatch_entry) == 0) return

    call extract_inline_numeric_field_cache(dispatch_entry, "pack=", pack_index_text, found_pack_index)
    call extract_inline_numeric_field_cache(dispatch_entry, "offset=", offset_text, found_offset)
    call extract_inline_numeric_field_cache(dispatch_entry, "bytes=", bytes_text, found_bytes)
    call extract_inline_numeric_field_cache(dispatch_entry, "role=", role_text, found_role)
    call extract_inline_numeric_field_cache(dispatch_entry, "layout=", layout_text, found_layout)
    if (found_pack_index) then
      if (parse_i64_text_cache(pack_index_text, parsed_i64)) then
        pack_index = max(0_i32, int(parsed_i64, kind=i32))
      end if
    end if
    if (.not. found_offset .or. .not. found_bytes .or. .not. found_role .or. .not. found_layout) then
      parsed_ok = (pack_index > 0_i32)
      return
    end if
    if (.not. parse_i64_text_cache(offset_text, pack_offset)) return
    if (.not. parse_i64_text_cache(bytes_text, pack_bytes)) return
    if (.not. parse_i64_text_cache(role_text, parsed_i64)) return
    role_code = int(parsed_i64, kind=i32)
    if (.not. parse_i64_text_cache(layout_text, parsed_i64)) return
    layout_code = int(parsed_i64, kind=i32)
    parsed_ok = (pack_bytes > 0_i64 .and. pack_offset >= 0_i64)
  end subroutine extract_payload_dispatch_entry_cache

  subroutine extract_payload_usage_entry_cache(usage_entry, pack_offset, pack_bytes, role_code, layout_code, parsed_ok)
    character(len=*), intent(in) :: usage_entry
    integer(i64), intent(out)    :: pack_offset
    integer(i64), intent(out)    :: pack_bytes
    integer(i32), intent(out)    :: role_code
    integer(i32), intent(out)    :: layout_code
    logical, intent(out)         :: parsed_ok
    character(len=64)            :: role_text
    character(len=64)            :: offset_text
    character(len=64)            :: bytes_text
    character(len=64)            :: layout_text
    logical                      :: found_role
    logical                      :: found_offset
    logical                      :: found_bytes
    logical                      :: found_layout

    pack_offset = 0_i64
    pack_bytes = 0_i64
    role_code = 0_i32
    layout_code = 0_i32
    parsed_ok = .false.
    if (len_trim(usage_entry) == 0) return

    call extract_pipe_field_cache(usage_entry, 2_i32, role_text, found_role)
    call extract_inline_numeric_field_cache(usage_entry, "offset=", offset_text, found_offset)
    call extract_inline_numeric_field_cache(usage_entry, "bytes=", bytes_text, found_bytes)
    call extract_inline_numeric_field_cache(usage_entry, "layout=", layout_text, found_layout)
    if (.not. found_offset .or. .not. found_bytes) return
    if (.not. parse_i64_text_cache(offset_text, pack_offset)) return
    if (.not. parse_i64_text_cache(bytes_text, pack_bytes)) return
    if (found_role) role_code = import_tensor_role_code(trim(role_text))
    if (found_layout) layout_code = import_tensor_layout_code(trim(layout_text))
    parsed_ok = (pack_bytes > 0_i64 .and. pack_offset >= 0_i64)
  end subroutine extract_payload_usage_entry_cache

  subroutine build_cuda_pack_page_record_cache(pack_offset, pack_bytes, role_code, layout_code, span_hash, &
                                               sample_bytes, actual_sample_bytes, page_hash, page_word_count, &
                                               page_words)
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i64), intent(in) :: span_hash
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i64), intent(in) :: actual_sample_bytes
    integer(i64), intent(out) :: page_hash
    integer(i32), intent(out) :: page_word_count
    integer(i32), intent(out) :: page_words(:)
    integer(i32)              :: stored_sample_bytes
    integer(i32)              :: preview_word_1
    integer(i32)              :: preview_word_2
    integer(i32)              :: preview_word_3
    integer(i32)              :: preview_word_4
    integer(i32)              :: control_word
    integer(i32)              :: word_index

    page_hash = 0_i64
    page_word_count = 0_i32
    page_words = 0_i32
    if (span_hash <= 0_i64 .or. pack_bytes <= 0_i64) return

    stored_sample_bytes = int(max(0_i64, min(actual_sample_bytes, int(size(sample_bytes), kind=i64))), kind=i32)
    preview_word_1 = pack_sample_word_cache(sample_bytes, stored_sample_bytes, 1_i32)
    preview_word_2 = pack_sample_word_cache(sample_bytes, stored_sample_bytes, 2_i32)
    preview_word_3 = pack_sample_word_cache(sample_bytes, stored_sample_bytes, 3_i32)
    preview_word_4 = pack_sample_word_cache(sample_bytes, stored_sample_bytes, 4_i32)
    control_word = ior(iand(role_code, int(z'000000FF', kind=i32)), &
      ishft(iand(layout_code, int(z'000000FF', kind=i32)), 8))
    control_word = ior(control_word, ishft(iand(stored_sample_bytes, int(z'000000FF', kind=i32)), 16))
    control_word = ior(control_word, ishft(4_i32, 24))

    page_word_count = min(MAX_CUDA_PACK_PAGE_WORDS, int(size(page_words), kind=i32))
    if (page_word_count < 8_i32) return

    page_words(1) = int(iand(pack_offset, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(2) = int(iand(pack_bytes, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(3) = control_word
    page_words(4) = int(iand(span_hash, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(5) = int(iand(shiftr(span_hash, 32), int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(6) = preview_word_1
    page_words(7) = preview_word_2
    page_words(8) = ieor(preview_word_3, ishft(preview_word_4, 1))

    page_hash = max(1_i64, span_hash)
    do word_index = 1_i32, 8_i32
      page_hash = combine_positive_hash64_cache(max(1_i64, page_hash), &
        iand(int(page_words(word_index), kind=i64), int(z'FFFFFFFF', kind=i64)))
    end do
  end subroutine build_cuda_pack_page_record_cache

  subroutine build_cuda_pack_tile_record_cache(pack_offset, pack_bytes, role_code, layout_code, sample_bytes, &
                                               actual_sample_bytes, tile_hash, tile_byte_count, tile_bytes)
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i64), intent(in) :: actual_sample_bytes
    integer(i64), intent(out) :: tile_hash
    integer(i32), intent(out) :: tile_byte_count
    integer(i8), intent(out)  :: tile_bytes(:)
    integer(i32)              :: stored_sample_bytes
    integer(i32)              :: byte_index
    integer(i32)              :: source_index
    integer(i32)              :: start_index

    tile_hash = 0_i64
    tile_byte_count = 0_i32
    tile_bytes = 0_i8
    if (pack_bytes <= 0_i64) return

    stored_sample_bytes = int(max(0_i64, min(actual_sample_bytes, int(size(sample_bytes), kind=i64))), kind=i32)
    if (stored_sample_bytes <= 0_i32) return

    tile_byte_count = min(MAX_CUDA_PACK_TILE_BYTES, min(stored_sample_bytes, int(size(tile_bytes), kind=i32)))
    if (tile_byte_count <= 0_i32) return

    start_index = 1_i32 + mod(int(modulo(pack_offset, int(max(1_i32, stored_sample_bytes), kind=i64)), kind=i32), &
      max(1_i32, stored_sample_bytes))
    do byte_index = 1_i32, tile_byte_count
      source_index = select_cuda_pack_tile_source_index_cache(layout_code, byte_index, stored_sample_bytes, start_index)
      tile_bytes(byte_index) = sample_bytes(source_index)
    end do

    tile_hash = positive_hash64_cache("cuda_pack_tile")
    tile_hash = combine_positive_hash64_cache(tile_hash, iand(pack_offset, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    tile_hash = combine_positive_hash64_cache(tile_hash, iand(pack_bytes, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    tile_hash = combine_positive_hash64_cache(tile_hash, int(role_code, kind=i64) + 257_i64)
    tile_hash = combine_positive_hash64_cache(tile_hash, int(layout_code, kind=i64) + 1025_i64)
    tile_hash = combine_positive_hash64_cache(tile_hash, hash_i8_buffer64_cache(tile_bytes, int(tile_byte_count, kind=i64)))
  end subroutine build_cuda_pack_tile_record_cache

  integer(i64) function build_cuda_materialized_pack_seed_cache(seed_label, pack_index, pack_offset, pack_bytes, &
                                                                 role_code, layout_code, pack_hash, pack_total_bytes, &
                                                                 span_hash) result(seed_hash)
    character(len=*), intent(in) :: seed_label
    integer(i32), intent(in)     :: pack_index
    integer(i64), intent(in)     :: pack_offset
    integer(i64), intent(in)     :: pack_bytes
    integer(i32), intent(in)     :: role_code
    integer(i32), intent(in)     :: layout_code
    integer(i64), intent(in)     :: pack_hash
    integer(i64), intent(in)     :: pack_total_bytes
    integer(i64), intent(in)     :: span_hash

    seed_hash = positive_hash64_cache(seed_label)
    seed_hash = combine_positive_hash64_cache(seed_hash, int(pack_index, kind=i64) + 17_i64)
    seed_hash = combine_positive_hash64_cache(seed_hash, iand(pack_offset, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    seed_hash = combine_positive_hash64_cache(seed_hash, iand(pack_bytes, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    seed_hash = combine_positive_hash64_cache(seed_hash, int(role_code, kind=i64) + 257_i64)
    seed_hash = combine_positive_hash64_cache(seed_hash, int(layout_code, kind=i64) + 1025_i64)
    seed_hash = combine_positive_hash64_cache(seed_hash, max(1_i64, pack_hash))
    seed_hash = combine_positive_hash64_cache(seed_hash, iand(pack_total_bytes, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    if (span_hash > 0_i64) then
      seed_hash = combine_positive_hash64_cache(seed_hash, max(1_i64, span_hash))
    end if
  end function build_cuda_materialized_pack_seed_cache

  subroutine build_cuda_materialized_pack_page_record_cache(pack_index, pack_offset, pack_bytes, role_code, &
                                                            layout_code, pack_hash, pack_total_bytes, span_hash, &
                                                            materialized_hash, page_hash, page_word_count, page_words)
    integer(i32), intent(in) :: pack_index
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i64), intent(in) :: pack_hash
    integer(i64), intent(in) :: pack_total_bytes
    integer(i64), intent(in) :: span_hash
    integer(i64), intent(in) :: materialized_hash
    integer(i64), intent(out) :: page_hash
    integer(i32), intent(out) :: page_word_count
    integer(i32), intent(out) :: page_words(:)
    integer(i32)              :: control_word
    integer(i32)              :: word_index

    page_hash = 0_i64
    page_word_count = 0_i32
    page_words = 0_i32
    if (pack_bytes <= 0_i64) return
    if (pack_hash <= 0_i64) return

    page_word_count = min(MAX_CUDA_PACK_PAGE_WORDS, int(size(page_words), kind=i32))
    if (page_word_count < 8_i32) return

    control_word = ior(iand(role_code, int(z'000000FF', kind=i32)), &
      ishft(iand(layout_code, int(z'000000FF', kind=i32)), 8))
    control_word = ior(control_word, ishft(iand(min(pack_index, 255_i32), int(z'000000FF', kind=i32)), 16))
    control_word = ior(control_word, ishft(8_i32, 24))

    page_words(1) = int(iand(pack_offset, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(2) = int(iand(pack_bytes, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(3) = control_word
    page_words(4) = int(iand(pack_hash, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(5) = int(iand(shiftr(pack_hash, 32), int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(6) = int(iand(materialized_hash, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(7) = int(iand(shiftr(materialized_hash, 32), int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(8) = ieor(int(iand(pack_total_bytes, int(z'FFFFFFFF', kind=i64)), kind=i32), &
      int(iand(span_hash, int(z'FFFFFFFF', kind=i64)), kind=i32))

    page_hash = build_cuda_materialized_pack_seed_cache("cuda_weight_pack_page_v2", pack_index, pack_offset, &
      pack_bytes, role_code, layout_code, pack_hash, pack_total_bytes, span_hash)
    do word_index = 1_i32, 8_i32
      page_hash = combine_positive_hash64_cache(page_hash, &
        iand(int(page_words(word_index), kind=i64), int(z'FFFFFFFF', kind=i64)))
    end do
  end subroutine build_cuda_materialized_pack_page_record_cache

  subroutine build_cuda_materialized_pack_tile_record_cache(pack_index, pack_offset, pack_bytes, role_code, &
                                                            layout_code, pack_hash, pack_total_bytes, span_hash, &
                                                            materialized_hash, tile_hash, tile_byte_count, tile_bytes)
    integer(i32), intent(in) :: pack_index
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i64), intent(in) :: pack_hash
    integer(i64), intent(in) :: pack_total_bytes
    integer(i64), intent(in) :: span_hash
    integer(i64), intent(in) :: materialized_hash
    integer(i64), intent(out) :: tile_hash
    integer(i32), intent(out) :: tile_byte_count
    integer(i8), intent(out)  :: tile_bytes(:)
    integer(i64)              :: state_hash
    integer(i32)              :: byte_index
    integer(i32)              :: shift_bits

    tile_hash = 0_i64
    tile_byte_count = 0_i32
    tile_bytes = 0_i8
    if (pack_bytes <= 0_i64) return
    if (pack_hash <= 0_i64) return

    tile_byte_count = min(MAX_CUDA_PACK_TILE_BYTES, int(size(tile_bytes), kind=i32))
    if (tile_byte_count <= 0_i32) return

    state_hash = build_cuda_materialized_pack_seed_cache("cuda_weight_pack_tile_v2", pack_index, pack_offset, &
      pack_bytes, role_code, layout_code, pack_hash, pack_total_bytes, span_hash)
    state_hash = combine_positive_hash64_cache(state_hash, materialized_hash)
    do byte_index = 1_i32, tile_byte_count
      state_hash = combine_positive_hash64_cache(state_hash, int(byte_index, kind=i64) + int(pack_index, kind=i64))
      shift_bits = 8_i32 * mod(byte_index - 1_i32, 8_i32)
      tile_bytes(byte_index) = int(iand(shiftr(state_hash, shift_bits), int(z'FF', kind=i64)), kind=i8)
    end do

    tile_hash = positive_hash64_cache("cuda_weight_pack_tile_hash_v2")
    tile_hash = combine_positive_hash64_cache(tile_hash, materialized_hash)
    tile_hash = combine_positive_hash64_cache(tile_hash, hash_i8_buffer64_cache(tile_bytes, int(tile_byte_count, kind=i64)))
  end subroutine build_cuda_materialized_pack_tile_record_cache

  pure integer(i32) function select_cuda_pack_tile_source_index_cache(layout_code, byte_index, stored_sample_bytes, &
                                                                      start_index) result(source_index)
    integer(i32), intent(in) :: layout_code
    integer(i32), intent(in) :: byte_index
    integer(i32), intent(in) :: stored_sample_bytes
    integer(i32), intent(in) :: start_index
    integer(i32)             :: half_count
    integer(i32)             :: local_index

    source_index = 1_i32
    if (stored_sample_bytes <= 0_i32) return

    select case (layout_code)
    case (2_i32)
      half_count = max(1_i32, (stored_sample_bytes + 1_i32) / 2_i32)
      if (mod(byte_index, 2_i32) == 1_i32) then
        local_index = (byte_index + 1_i32) / 2_i32
      else
        local_index = half_count + (byte_index / 2_i32)
      end if
    case (3_i32)
      local_index = 1_i32 + mod((byte_index - 1_i32) * 2_i32, stored_sample_bytes)
    case default
      local_index = byte_index
    end select

    source_index = 1_i32 + mod((start_index - 1_i32) + (local_index - 1_i32), stored_sample_bytes)
  end function select_cuda_pack_tile_source_index_cache

  pure integer(i32) function pack_sample_word_cache(sample_bytes, stored_sample_bytes, word_index) result(word_value)
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i32), intent(in) :: stored_sample_bytes
    integer(i32), intent(in) :: word_index
    integer(i32)             :: byte_offset
    integer(i32)             :: byte_index
    integer(i32)             :: byte_value

    word_value = 0_i32
    if (word_index <= 0_i32) return

    byte_offset = (word_index - 1_i32) * 4_i32
    do byte_index = 0_i32, 3_i32
      if ((byte_offset + byte_index + 1_i32) > stored_sample_bytes) exit
      if ((byte_offset + byte_index + 1_i32) > int(size(sample_bytes), kind=i32)) exit
      byte_value = int(sample_bytes(byte_offset + byte_index + 1_i32), kind=i32)
      if (byte_value < 0_i32) byte_value = byte_value + 256_i32
      word_value = ior(word_value, ishft(byte_value, 8 * byte_index))
    end do
  end function pack_sample_word_cache

  subroutine encode_i32_words_as_hex_cache(word_values, word_count, hex_text)
    integer(i32), intent(in)       :: word_values(:)
    integer(i32), intent(in)       :: word_count
    character(len=*), intent(out)  :: hex_text
    integer(i8)                    :: packed_bytes(MAX_CUDA_PACK_PAGE_WORDS * 4)
    integer(i32)                   :: encode_word_count
    integer(i32)                   :: byte_index

    packed_bytes = 0_i8
    encode_word_count = max(0_i32, min(word_count, min(int(size(word_values), kind=i32), MAX_CUDA_PACK_PAGE_WORDS)))
    do byte_index = 1_i32, encode_word_count
      call store_i32_as_le_bytes_cache(word_values(byte_index), packed_bytes(((byte_index - 1_i32) * 4_i32) + 1: &
        ((byte_index - 1_i32) * 4_i32) + 4_i32))
    end do
    call encode_bytes_as_hex(packed_bytes, encode_word_count * 4_i32, hex_text)
  end subroutine encode_i32_words_as_hex_cache

  subroutine store_i32_as_le_bytes_cache(word_value, byte_values)
    integer(i32), intent(in)      :: word_value
    integer(i8), intent(out)      :: byte_values(:)
    integer(i64)                  :: unsigned_word
    integer(i32)                  :: byte_index
    integer(i32)                  :: byte_value

    byte_values = 0_i8
    unsigned_word = iand(int(word_value, kind=i64), int(z'FFFFFFFF', kind=i64))
    do byte_index = 1_i32, min(4_i32, int(size(byte_values), kind=i32))
      byte_value = int(iand(shiftr(unsigned_word, 8 * (byte_index - 1_i32)), int(z'FF', kind=i64)), kind=i32)
      if (byte_value > 127_i32) then
        byte_values(byte_index) = int(byte_value - 256_i32, kind=i8)
      else
        byte_values(byte_index) = int(byte_value, kind=i8)
      end if
    end do
  end subroutine store_i32_as_le_bytes_cache

  function join_cache_root_with_payload_path(cache_root, payload_path) result(full_path)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: full_path
    integer                      :: root_len

    full_path = ""
    if (len_trim(cache_root) == 0 .or. len_trim(payload_path) == 0) return

    root_len = len_trim(cache_root)
    if (cache_root(root_len:root_len) == "/") then
      full_path = trim(cache_root) // trim(payload_path)
    else
      full_path = trim(cache_root) // "/" // trim(payload_path)
    end if
  end function join_cache_root_with_payload_path

  function parent_directory_path(file_path) result(parent_path)
    character(len=*), intent(in) :: file_path
    character(len=MAX_PATH_LEN)  :: parent_path
    integer                      :: index_char

    parent_path = ""
    do index_char = len_trim(file_path), 1, -1
      if (file_path(index_char:index_char) == "/") then
        if (index_char > 1) then
          parent_path = file_path(1:index_char-1)
        end if
        return
      end if
    end do
  end function parent_directory_path

  pure function artifact_family_token(backend_family) result(family_token)
    integer(i32), intent(in)    :: backend_family
    character(len=MAX_NAME_LEN) :: family_token

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      family_token = "apple"
    case (MIZU_BACKEND_FAMILY_CUDA)
      family_token = "cuda"
    case default
      family_token = "generic"
    end select
  end function artifact_family_token

  pure function artifact_route_token(execution_route) result(route_token)
    integer(i32), intent(in)    :: execution_route
    character(len=MAX_NAME_LEN) :: route_token

    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      route_token = "ane"
    case (MIZU_EXEC_ROUTE_METAL)
      route_token = "metal"
    case (MIZU_EXEC_ROUTE_CUDA)
      route_token = "cuda"
    case default
      route_token = "generic"
    end select
  end function artifact_route_token

  pure function artifact_stage_token(stage_kind) result(stage_token)
    integer(i32), intent(in)    :: stage_kind
    character(len=MAX_NAME_LEN) :: stage_token

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      stage_token = "weight_pack"
    case (MIZU_STAGE_PROJECTOR)
      stage_token = "projector_cache"
    case (MIZU_STAGE_PREFILL)
      stage_token = "prefill_plan"
    case (MIZU_STAGE_DECODE)
      stage_token = "decode_plan"
    case (MIZU_STAGE_PARK, MIZU_STAGE_RESUME)
      stage_token = "session_checkpoint"
    case default
      stage_token = "artifact"
    end select
  end function artifact_stage_token

  subroutine assign_stage_candidate(candidate_index, candidate_backend_families, candidate_execution_routes, &
                                    candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, &
                                    backend_family, execution_route)
    integer(i32), intent(in)      :: candidate_index
    integer(i32), intent(in)      :: candidate_backend_families(:)
    integer(i32), intent(in)      :: candidate_execution_routes(:)
    integer(i64), intent(in)      :: candidate_plan_ids(:)
    character(len=*), intent(in)  :: candidate_key_texts(:)
    character(len=*), intent(out) :: candidate_key_text
    integer(i64), intent(out)     :: plan_id
    integer(i32), intent(out)     :: backend_family
    integer(i32), intent(out)     :: execution_route

    candidate_key_text = trim(candidate_key_texts(candidate_index))
    plan_id = candidate_plan_ids(candidate_index)
    backend_family = candidate_backend_families(candidate_index)
    execution_route = candidate_execution_routes(candidate_index)
  end subroutine assign_stage_candidate

  pure integer(i32) function find_candidate_index(candidate_key_texts, candidate_plan_ids, candidate_count, &
                                                  winner_candidate_key_text, winner_plan_id) result(candidate_index)
    character(len=*), intent(in) :: candidate_key_texts(:)
    integer(i64), intent(in)     :: candidate_plan_ids(:)
    integer(i32), intent(in)     :: candidate_count
    character(len=*), intent(in) :: winner_candidate_key_text
    integer(i64), intent(in)     :: winner_plan_id
    integer(i32)                 :: index

    candidate_index = 0_i32
    if (len_trim(winner_candidate_key_text) > 0) then
      do index = 1_i32, candidate_count
        if (trim(candidate_key_texts(index)) == trim(winner_candidate_key_text)) then
          candidate_index = index
          return
        end if
      end do
    end if

    if (winner_plan_id /= 0_i64) then
      do index = 1_i32, candidate_count
        if (candidate_plan_ids(index) == winner_plan_id) then
          candidate_index = index
          return
        end if
      end do
    end if
  end function find_candidate_index

  integer(i64) function monotonic_timestamp_us() result(timestamp_us)
    integer(i64) :: clock_count
    integer(i64) :: clock_rate

    call system_clock(clock_count, clock_rate)
    if (clock_rate <= 0_i64) then
      timestamp_us = 0_i64
      return
    end if

    timestamp_us = (clock_count * 1000000_i64) / clock_rate
  end function monotonic_timestamp_us

  integer(i64) function elapsed_since_us(started_us) result(elapsed_us)
    integer(i64), intent(in) :: started_us
    integer(i64)             :: finished_us

    if (started_us <= 0_i64) then
      elapsed_us = 1_i64
      return
    end if

    finished_us = monotonic_timestamp_us()
    elapsed_us = max(1_i64, finished_us - started_us)
  end function elapsed_since_us

  logical function force_session_eviction_requested() result(is_forced)
    logical :: has_override

    call read_boolean_env_override("MIZU_FORCE_SESSION_EVICTION", has_override, is_forced)
    if (.not. has_override) is_forced = .false.
  end function force_session_eviction_requested

  subroutine hydrate_runtime_cache_state(runtime, runtime_cache)
    type(runtime_state), intent(in)          :: runtime
    type(runtime_cache_bundle), intent(inout) :: runtime_cache
    character(len=MAX_PATH_LEN)              :: store_path
    logical                                  :: loaded_ok

    store_path = build_runtime_artifact_cache_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call load_runtime_cache_bundle(runtime_cache, trim(store_path), loaded_ok)
  end subroutine hydrate_runtime_cache_state

  subroutine persist_runtime_cache_state(runtime, runtime_cache)
    type(runtime_state), intent(in)         :: runtime
    type(runtime_cache_bundle), intent(in)  :: runtime_cache
    character(len=MAX_PATH_LEN)             :: store_path
    logical                                 :: saved_ok

    store_path = build_runtime_artifact_cache_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call ensure_directory_exists(runtime%config%cache_root)
    call save_runtime_cache_bundle(runtime_cache, trim(store_path), saved_ok)
  end subroutine persist_runtime_cache_state

  subroutine hydrate_runtime_optimization_state(runtime, optimization_store)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=MAX_PATH_LEN)                    :: store_path
    logical                                        :: loaded_ok

    store_path = build_runtime_optimization_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call load_runtime_optimization_store(optimization_store, trim(store_path), loaded_ok)
  end subroutine hydrate_runtime_optimization_state

  subroutine persist_runtime_optimization_state(runtime, optimization_store)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(in)   :: optimization_store
    character(len=MAX_PATH_LEN)                    :: store_path
    logical                                        :: saved_ok

    store_path = build_runtime_optimization_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call ensure_directory_exists(runtime%config%cache_root)
    call save_runtime_optimization_store(optimization_store, trim(store_path), saved_ok)
  end subroutine persist_runtime_optimization_state

  function build_runtime_artifact_cache_store_path(cache_root) result(store_path)
    character(len=*), intent(in) :: cache_root
    character(len=MAX_PATH_LEN)  :: store_path
    integer                      :: root_len

    store_path = ""
    root_len = len_trim(cache_root)
    if (root_len == 0) return

    if (cache_root(root_len:root_len) == "/") then
      store_path = trim(cache_root) // "artifact_cache_v1.txt"
    else
      store_path = trim(cache_root) // "/artifact_cache_v1.txt"
    end if
  end function build_runtime_artifact_cache_store_path

  function build_runtime_optimization_store_path(cache_root) result(store_path)
    character(len=*), intent(in) :: cache_root
    character(len=MAX_PATH_LEN)  :: store_path
    integer                      :: root_len

    store_path = ""
    root_len = len_trim(cache_root)
    if (root_len == 0) return

    if (cache_root(root_len:root_len) == "/") then
      store_path = trim(cache_root) // "optimization_store_v1.txt"
    else
      store_path = trim(cache_root) // "/optimization_store_v1.txt"
    end if
  end function build_runtime_optimization_store_path

  subroutine ensure_directory_exists(directory_path)
    character(len=*), intent(in) :: directory_path
    character(len=(MAX_PATH_LEN * 2) + 16) :: command_text
    integer :: exit_status

    if (len_trim(directory_path) == 0) return

    command_text = "mkdir -p " // trim(shell_quote_text(trim(directory_path)))
    call execute_command_line(trim(command_text), exitstat=exit_status)
  end subroutine ensure_directory_exists

  subroutine resolve_import_span_record_cache(span_root, span_path, requested_sample_bytes, span_hash, &
                                              actual_sample_bytes, sample_bytes, source_offset, source_byte_count)
    character(len=*), intent(in) :: span_root
    character(len=*), intent(in) :: span_path
    integer(i64), intent(in)     :: requested_sample_bytes
    integer(i64), intent(out)    :: span_hash
    integer(i64), intent(out)    :: actual_sample_bytes
    integer(i8), intent(out)     :: sample_bytes(:)
    integer(i64), intent(in), optional :: source_offset
    integer(i64), intent(in), optional :: source_byte_count
    character(len=MAX_PATH_LEN)  :: full_path
    integer(i32)                 :: unit_id
    integer(i32)                 :: ios
    integer(i64)                 :: sample_count
    integer(i64)                 :: file_size
    integer(i64)                 :: span_start
    integer(i64)                 :: span_available
    logical                      :: exists
    integer(i64)                 :: stored_count
    integer(i8), allocatable     :: sample_buffer(:)

    span_hash = 0_i64
    actual_sample_bytes = 0_i64
    sample_bytes = 0_i8
    if (len_trim(span_root) == 0 .or. len_trim(span_path) == 0) return

    full_path = join_import_span_path_cache(span_root, span_path)
    span_hash = positive_hash64_cache(trim(full_path))
    span_start = 0_i64
    if (present(source_offset)) then
      if (source_offset >= 0_i64) then
        span_start = source_offset
        span_hash = combine_positive_hash64_cache(max(1_i64, span_hash), source_offset + 4099_i64)
        if (present(source_byte_count)) then
          span_hash = combine_positive_hash64_cache(max(1_i64, span_hash), max(0_i64, source_byte_count) + 8191_i64)
        end if
      end if
    end if
    inquire(file=trim(full_path), exist=exists, size=file_size)
    if (.not. exists) return

    span_available = max(0_i64, file_size - span_start)
    if (present(source_byte_count)) then
      if (source_byte_count > 0_i64) span_available = min(span_available, source_byte_count)
    end if

    sample_count = max(0_i64, min(max(1_i64, requested_sample_bytes), span_available))
    if (sample_count <= 0_i64) return

    allocate(sample_buffer(sample_count))
    sample_buffer = 0_i8
    open(newunit=unit_id, file=trim(full_path), status="old", access="stream", form="unformatted", &
      action="read", iostat=ios)
    if (ios /= 0_i32) then
      deallocate(sample_buffer)
      return
    end if
    read(unit_id, pos=span_start + 1_i64, iostat=ios) sample_buffer
    close(unit_id)
    if (ios /= 0_i32) then
      deallocate(sample_buffer)
      return
    end if

    actual_sample_bytes = sample_count
    span_hash = combine_positive_hash64_cache(max(1_i64, span_hash), &
      hash_i8_buffer64_cache(sample_buffer, sample_count))
    stored_count = min(sample_count, int(size(sample_bytes), kind=i64))
    if (stored_count > 0_i64) sample_bytes(1:stored_count) = sample_buffer(1:stored_count)
    deallocate(sample_buffer)
  end subroutine resolve_import_span_record_cache

  pure function join_import_span_path_cache(span_root, span_path) result(full_path)
    character(len=*), intent(in) :: span_root
    character(len=*), intent(in) :: span_path
    character(len=MAX_PATH_LEN)  :: full_path
    integer(i32)                 :: root_len

    full_path = ""
    if (len_trim(span_root) == 0 .or. len_trim(span_path) == 0) return
    if (span_path(1:1) == "/") then
      full_path = trim(span_path)
      return
    end if

    root_len = len_trim(span_root)
    if (span_root(root_len:root_len) == "/") then
      full_path = trim(span_root) // trim(span_path)
    else
      full_path = trim(span_root) // "/" // trim(span_path)
    end if
  end function join_import_span_path_cache

  integer(i64) function positive_hash64_cache(text) result(hash_value)
    character(len=*), intent(in) :: text

    hash_value = iand(hash_text64(text), int(z'7FFFFFFFFFFFFFFF', kind=i64))
    if (hash_value == 0_i64) hash_value = 1_i64
  end function positive_hash64_cache

  integer(i64) function combine_positive_hash64_cache(base_hash, content_hash) result(hash_value)
    integer(i64), intent(in) :: base_hash
    integer(i64), intent(in) :: content_hash
    integer(i64)             :: mixed_hash

    mixed_hash = ieor(max(1_i64, base_hash), content_hash + int(z'9E3779B97F4A7C15', kind=i64))
    mixed_hash = ieor(mixed_hash, shiftr(mixed_hash, 30))
    mixed_hash = mixed_hash * int(z'BF58476D1CE4E5B9', kind=i64)
    mixed_hash = ieor(mixed_hash, shiftr(mixed_hash, 27))
    mixed_hash = mixed_hash * int(z'94D049BB133111EB', kind=i64)
    hash_value = iand(ieor(mixed_hash, shiftr(mixed_hash, 31)), int(z'7FFFFFFFFFFFFFFF', kind=i64))
    if (hash_value == 0_i64) hash_value = 1_i64
  end function combine_positive_hash64_cache

  integer(i64) function hash_i8_buffer64_cache(buffer, buffer_count) result(hash_value)
    integer(i8), intent(in)  :: buffer(:)
    integer(i64), intent(in) :: buffer_count
    integer(i64)             :: index_byte

    hash_value = positive_hash64_cache("cuda_import_span")
    if (buffer_count <= 0_i64) return
    do index_byte = 1_i64, min(buffer_count, int(size(buffer), kind=i64))
      hash_value = combine_positive_hash64_cache(max(1_i64, hash_value), int(buffer(index_byte), kind=i64) + 257_i64)
    end do
  end function hash_i8_buffer64_cache

  subroutine extract_inline_numeric_field_cache(source_text, key_text, value_text, found)
    character(len=*), intent(in)  :: source_text
    character(len=*), intent(in)  :: key_text
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: value_start
    integer(i32)                  :: remaining_len
    integer(i32)                  :: separator_index
    integer(i32)                  :: copy_len

    value_text = ""
    found = .false.
    if (len_trim(source_text) == 0 .or. len_trim(key_text) == 0) return

    start_index = index(source_text, trim(key_text))
    if (start_index <= 0) return

    value_start = start_index + len_trim(key_text)
    if (value_start > len_trim(source_text)) return

    remaining_len = len_trim(source_text) - value_start + 1_i32
    separator_index = index(source_text(value_start:value_start + remaining_len - 1_i32), "|")
    if (separator_index <= 0) then
      copy_len = min(len_trim(source_text) - value_start + 1_i32, len(value_text))
    else
      copy_len = min(separator_index - 1_i32, len(value_text))
    end if
    if (copy_len <= 0_i32) return

    value_text(1:copy_len) = source_text(value_start:value_start + copy_len - 1_i32)
    found = (len_trim(value_text) > 0)
  end subroutine extract_inline_numeric_field_cache

  logical function parse_i64_text_cache(text, value_out) result(parsed_ok)
    character(len=*), intent(in) :: text
    integer(i64), intent(out)    :: value_out
    integer(i32)                 :: ios

    value_out = 0_i64
    parsed_ok = .false.
    if (len_trim(text) == 0) return
    read(text, *, iostat=ios) value_out
    parsed_ok = (ios == 0_i32)
  end function parse_i64_text_cache

  subroutine extract_payload_field_text_cache(payload_text, key_text, value_text, found)
    character(len=*), intent(in)  :: payload_text
    character(len=*), intent(in)  :: key_text
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: value_start
    integer(i32)                  :: remaining_len
    integer(i32)                  :: separator_index
    integer(i32)                  :: copy_len

    value_text = ""
    found = .false.
    if (len_trim(payload_text) == 0 .or. len_trim(key_text) == 0) return

    start_index = index(payload_text, trim(key_text))
    if (start_index <= 0) return

    value_start = start_index + len_trim(key_text)
    if (value_start > len_trim(payload_text)) return

    remaining_len = len_trim(payload_text) - value_start + 1_i32
    separator_index = index(payload_text(value_start:value_start + remaining_len - 1_i32), ";")
    if (separator_index <= 0) then
      copy_len = min(len_trim(payload_text) - value_start + 1_i32, len(value_text))
    else
      copy_len = min(separator_index - 1_i32, len(value_text))
    end if
    if (copy_len <= 0_i32) return

    value_text(1:copy_len) = payload_text(value_start:value_start + copy_len - 1_i32)
    found = (len_trim(value_text) > 0)
  end subroutine extract_payload_field_text_cache

  subroutine extract_pipe_field_cache(source_text, field_index, value_text, found)
    character(len=*), intent(in)  :: source_text
    integer(i32), intent(in)      :: field_index
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: pipe_index
    integer(i32)                  :: current_field
    integer(i32)                  :: value_len

    value_text = ""
    found = .false.
    if (len_trim(source_text) == 0 .or. field_index <= 0_i32) return

    start_index = 1_i32
    current_field = 1_i32
    do
      pipe_index = index(source_text(start_index:len_trim(source_text)), "|")
      if (current_field == field_index) then
        if (pipe_index <= 0) then
          value_len = min(len_trim(source_text) - start_index + 1_i32, len(value_text))
        else
          value_len = min(pipe_index - 1_i32, len(value_text))
        end if
        if (value_len > 0_i32) then
          value_text(1:value_len) = source_text(start_index:start_index + value_len - 1_i32)
          found = (len_trim(value_text) > 0)
        end if
        return
      end if
      if (pipe_index <= 0) exit
      start_index = start_index + pipe_index
      current_field = current_field + 1_i32
      if (start_index > len_trim(source_text)) exit
    end do
  end subroutine extract_pipe_field_cache

  function shell_quote_text(text) result(quoted_text)
    character(len=*), intent(in) :: text
    character(len=(MAX_PATH_LEN * 2) + 16) :: quoted_text
    integer :: index_char
    integer :: cursor

    quoted_text = ""
    if (len_trim(text) == 0) return

    cursor = 1
    quoted_text(cursor:cursor) = '"'
    cursor = cursor + 1

    do index_char = 1, len_trim(text)
      select case (text(index_char:index_char))
      case ('\')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\\"
          cursor = cursor + 2
        end if
      case ('"')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = '\"'
          cursor = cursor + 2
        end if
      case ('$')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\$"
          cursor = cursor + 2
        end if
      case ('`')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\`"
          cursor = cursor + 2
        end if
      case default
        if (cursor <= len(quoted_text)) then
          quoted_text(cursor:cursor) = text(index_char:index_char)
          cursor = cursor + 1
        end if
      end select
    end do

    if (cursor <= len(quoted_text)) quoted_text(cursor:cursor) = '"'
  end function shell_quote_text

  function copy_c_string_ptr(string_ptr, default_text) result(text)
    type(c_ptr), value         :: string_ptr
    character(len=*), intent(in) :: default_text
    character(len=MAX_PATH_LEN)  :: text

    text = ""
    call copy_c_string_ptr_to_fortran(string_ptr, text)
    if (len_trim(text) == 0) then
      text = trim(default_text)
    end if
  end function copy_c_string_ptr

  subroutine copy_c_string_ptr_to_fortran(string_ptr, text)
    type(c_ptr), value       :: string_ptr
    character(len=*), intent(out) :: text
    character(kind=c_char), pointer :: chars(:)
    integer(c_size_t)        :: c_length
    integer                  :: copy_len

    text = ""

    if (.not. c_associated(string_ptr)) return

    c_length = c_strlen(string_ptr)
    if (c_length == 0_c_size_t) return

    copy_len = min(int(c_length), len(text))
    call c_f_pointer(string_ptr, chars, [copy_len])
    text(1:copy_len) = transfer(chars(1:copy_len), text(1:copy_len))
  end subroutine copy_c_string_ptr_to_fortran

  subroutine copy_fortran_string_to_c(text, buffer_ptr, capacity)
    character(len=*), intent(in) :: text
    type(c_ptr), value           :: buffer_ptr
    integer(c_size_t), value     :: capacity
    character(kind=c_char), pointer :: buffer(:)
    integer                        :: copy_len, index_char

    if (.not. c_associated(buffer_ptr)) return
    if (capacity == 0_c_size_t) return

    call c_f_pointer(buffer_ptr, buffer, [int(capacity)])
    buffer = c_null_char

    copy_len = min(len_trim(text), int(capacity) - 1)
    do index_char = 1, copy_len
      buffer(index_char) = text(index_char:index_char)
    end do
    buffer(copy_len + 1) = c_null_char
  end subroutine copy_fortran_string_to_c

  subroutine write_size_t_pointer(size_ptr, value)
    type(c_ptr), value :: size_ptr
    integer(i64), intent(in) :: value
    integer(c_size_t), pointer :: out_size

    if (.not. c_associated(size_ptr)) return
    call c_f_pointer(size_ptr, out_size)
    if (associated(out_size)) then
      out_size = int(max(0_i64, value), kind=c_size_t)
    end if
  end subroutine write_size_t_pointer

  pure integer(i32) function require_input_struct_size(actual_size, expected_size) &
      result(status_code)
    integer(c_size_t), intent(in) :: actual_size
    integer(c_size_t), intent(in) :: expected_size

    status_code = MIZU_STATUS_OK
    if (actual_size < expected_size) status_code = MIZU_STATUS_ABI_MISMATCH
  end function require_input_struct_size

  pure integer(i32) function require_output_struct_size(actual_size, expected_size) &
      result(status_code)
    integer(c_size_t), intent(in) :: actual_size
    integer(c_size_t), intent(in) :: expected_size

    status_code = MIZU_STATUS_OK
    if (actual_size < expected_size) status_code = MIZU_STATUS_BUFFER_TOO_SMALL
  end function require_output_struct_size

  pure integer(i32) function require_retired_handle_capacity(retired_count) &
      result(status_code)
    integer(i64), intent(in) :: retired_count

    status_code = MIZU_STATUS_OK
    if (retired_count >= MAX_RETIRED_HANDLE_BOXES_PER_KIND) then
      status_code = MIZU_STATUS_BUSY
    end if
  end function require_retired_handle_capacity

  integer(i32) function validate_modal_input_descriptor_c(input) result(status_code)
    type(c_modal_input_desc), intent(in) :: input
    character(len=MAX_PATH_LEN)          :: slot_name

    status_code = MIZU_STATUS_OK

    slot_name = trim(copy_c_string_ptr(input%slot_name_z, "image"))
    if (len_trim(slot_name) == 0) slot_name = "image"
    if (trim(slot_name) /= "image") then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (int(input%placeholder_ordinal, kind=i32) /= 1_i32) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    select case (int(input%modality_kind, kind=i32))
    case (MIZU_MODALITY_KIND_IMAGE)
      if (int(input%storage_kind, kind=i32) /= MIZU_STORAGE_KIND_ENCODED_BYTES) then
        status_code = MIZU_STATUS_UNSUPPORTED_MODALITY
        return
      end if

      if (int(input%dtype, kind=i32) /= MIZU_DTYPE_U8) then
        status_code = MIZU_STATUS_UNSUPPORTED_MODALITY
        return
      end if
    case (MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS)
      if (int(input%storage_kind, kind=i32) /= MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS) then
        status_code = MIZU_STATUS_UNSUPPORTED_MODALITY
        return
      end if

      select case (int(input%dtype, kind=i32))
      case (MIZU_DTYPE_F16, MIZU_DTYPE_BF16, MIZU_DTYPE_F32)
        continue
      case default
        status_code = MIZU_STATUS_UNSUPPORTED_MODALITY
        return
      end select
    case default
      status_code = MIZU_STATUS_UNSUPPORTED_MODALITY
      return
    end select

    if (int(input%rank, kind=i32) /= 0_i32 .or. c_associated(input%shape)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    select case (int(input%lifetime_policy, kind=i32))
    case (MIZU_LIFETIME_POLICY_COPY, MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL)
      continue
    case default
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end select
  end function validate_modal_input_descriptor_c

  pure function make_stage_report(stage_kind, backend_family, execution_route, fallback_reason, &
                                  selection_mode, cold_state, cache_flags, plan_id, elapsed_us) &
      result(report)
    integer(i32), intent(in) :: stage_kind
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    integer(i32), intent(in) :: fallback_reason
    integer(i32), intent(in) :: selection_mode
    integer(i32), intent(in) :: cold_state
    integer(i64), intent(in) :: cache_flags
    integer(i64), intent(in) :: plan_id
    integer(i64), intent(in) :: elapsed_us
    type(execution_report)   :: report

    report%stage_kind      = stage_kind
    report%backend_family  = backend_family
    report%execution_route = execution_route
    report%plan_id         = plan_id
    report%selection_mode  = selection_mode
    report%cold_state      = cold_state
    report%fallback_reason = fallback_reason
    report%cache_flags     = cache_flags
    report%elapsed_us      = elapsed_us
  end function make_stage_report

  subroutine copy_internal_report_to_c(report, c_report)
    type(execution_report), intent(in)     :: report
    type(c_execution_report), intent(inout)  :: c_report
    integer(c_size_t)                       :: struct_size

    struct_size = c_report%struct_size
    if (struct_size == 0_c_size_t) struct_size = c_sizeof(c_report)
    c_report%stage_kind      = int(report%stage_kind, kind=c_int32_t)
    c_report%backend_family  = int(report%backend_family, kind=c_int32_t)
    c_report%execution_route = int(report%execution_route, kind=c_int32_t)
    c_report%plan_id         = int(report%plan_id, kind=c_int64_t)
    c_report%selection_mode  = int(report%selection_mode, kind=c_int32_t)
    c_report%cold_state      = int(report%cold_state, kind=c_int32_t)
    c_report%fallback_reason = int(report%fallback_reason, kind=c_int32_t)
    c_report%cache_flags     = int(report%cache_flags, kind=c_int64_t)
    c_report%elapsed_us      = int(report%elapsed_us, kind=c_int64_t)
    c_report%struct_size     = struct_size
  end subroutine copy_internal_report_to_c

  integer(i32) function prepare_report_buffer(report_buffer_ptr, required_count) result(status_code)
    type(c_ptr), value :: report_buffer_ptr
    integer(i64), intent(in) :: required_count
    type(c_report_buffer), pointer :: report_buffer

    status_code = MIZU_STATUS_OK
    if (.not. c_associated(report_buffer_ptr)) return

    call c_f_pointer(report_buffer_ptr, report_buffer)
    if (.not. associated(report_buffer)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    status_code = require_output_struct_size(report_buffer%struct_size, c_sizeof(report_buffer))
    if (status_code /= MIZU_STATUS_OK) return

    report_buffer%report_count = int(required_count, kind=c_size_t)
    if (report_buffer%report_capacity < int(required_count, kind=c_size_t)) then
      status_code = MIZU_STATUS_BUFFER_TOO_SMALL
    else if (required_count > 0_i64 .and. .not. c_associated(report_buffer%reports)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
    end if
  end function prepare_report_buffer

  pure integer(i64) function decode_kv_shape_band(kv_token_count) result(shape_band)
    integer(i64), intent(in) :: kv_token_count
    integer(i64)             :: normalized_count
    integer(i64)             :: max_doubling_band

    normalized_count = max(0_i64, kv_token_count)
    if (normalized_count <= 0_i64) then
      shape_band = 0_i64
      return
    end if

    shape_band = 16_i64
    max_doubling_band = shiftr(huge(shape_band), 1)
    do while (shape_band < normalized_count .and. shape_band <= max_doubling_band)
      shape_band = shape_band * 2_i64
    end do
  end function decode_kv_shape_band

  subroutine fill_report_buffer(report_buffer_ptr, primary_report, secondary_report)
    type(c_ptr), value           :: report_buffer_ptr
    type(execution_report), intent(in) :: primary_report
    type(execution_report), intent(in) :: secondary_report
    type(c_report_buffer), pointer     :: report_buffer
    type(c_execution_report), pointer  :: reports(:)
    integer                            :: report_count

    if (.not. c_associated(report_buffer_ptr)) return

    call c_f_pointer(report_buffer_ptr, report_buffer)
    if (.not. associated(report_buffer)) return
    if (.not. c_associated(report_buffer%reports)) return

    report_count = int(report_buffer%report_count)
    call c_f_pointer(report_buffer%reports, reports, [report_count])
    if (.not. associated(reports)) return

    if (secondary_report%stage_kind /= MIZU_STAGE_NONE .and. report_count >= 2) then
      call copy_internal_report_to_c(secondary_report, reports(1))
      call copy_internal_report_to_c(primary_report, reports(2))
    else if (report_count >= 1) then
      call copy_internal_report_to_c(primary_report, reports(1))
    end if
  end subroutine fill_report_buffer

end module mod_c_api
