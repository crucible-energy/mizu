module mod_types
  use iso_c_binding, only: c_ptr, c_null_ptr
  use mod_kinds, only: i8, i32, i64, r32, MAX_TENSOR_RANK, MAX_NAME_LEN, &
                       MAX_SLOT_NAME_LEN, MAX_PATH_LEN, MAX_ERROR_MESSAGE_LEN
  use mod_status, only: MIZU_STATUS_OK

  implicit none

  private
  public :: MIZU_ABI_VERSION
  public :: MIZU_OPTIMIZATION_MODE_DISABLED, MIZU_OPTIMIZATION_MODE_MEASURE_ONLY
  public :: MIZU_OPTIMIZATION_MODE_LEARN_AND_REUSE
  public :: MIZU_BACKEND_FAMILY_NONE, MIZU_BACKEND_FAMILY_APPLE
  public :: MIZU_BACKEND_FAMILY_CUDA
  public :: MIZU_EXEC_ROUTE_NONE, MIZU_EXEC_ROUTE_ANE
  public :: MIZU_EXEC_ROUTE_METAL, MIZU_EXEC_ROUTE_CUDA
  public :: MIZU_MODEL_FAMILY_UNKNOWN, MIZU_MODEL_FAMILY_QWEN3_5
  public :: MIZU_MODEL_FAMILY_GEMMA4
  public :: MIZU_STAGE_NONE, MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR
  public :: MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, MIZU_STAGE_PARK
  public :: MIZU_STAGE_RESUME
  public :: MIZU_SELECTION_MODE_NONE, MIZU_SELECTION_MODE_DIRECT
  public :: MIZU_SELECTION_MODE_EXPLORATORY, MIZU_SELECTION_MODE_REUSE
  public :: MIZU_COLD_STATE_UNKNOWN, MIZU_COLD_STATE_COLD
  public :: MIZU_COLD_STATE_WARM
  public :: MIZU_FALLBACK_REASON_NONE, MIZU_FALLBACK_REASON_UNSUPPORTED_OP
  public :: MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE
  public :: MIZU_FALLBACK_REASON_BACKEND_UNAVAILABLE
  public :: MIZU_FALLBACK_REASON_ROUTE_DISALLOWED
  public :: MIZU_FALLBACK_REASON_PLANNER_POLICY
  public :: MIZU_SAMPLER_KIND_NONE, MIZU_SAMPLER_KIND_GREEDY
  public :: MIZU_SAMPLER_KIND_TOP_K_TOP_P
  public :: MIZU_MODALITY_KIND_UNKNOWN, MIZU_MODALITY_KIND_IMAGE
  public :: MIZU_MODALITY_KIND_TENSOR, MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS
  public :: MIZU_STORAGE_KIND_UNKNOWN, MIZU_STORAGE_KIND_ENCODED_BYTES
  public :: MIZU_STORAGE_KIND_HOST_TENSOR, MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS
  public :: MIZU_DTYPE_UNKNOWN, MIZU_DTYPE_U8, MIZU_DTYPE_I32
  public :: MIZU_DTYPE_F16, MIZU_DTYPE_BF16, MIZU_DTYPE_F32
  public :: MIZU_LIFETIME_POLICY_COPY, MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL
  public :: MIZU_OUTPUT_KIND_NONE, MIZU_OUTPUT_KIND_TOKEN_IDS
  public :: MIZU_STOP_REASON_NONE, MIZU_STOP_REASON_EOS
  public :: MIZU_STOP_REASON_TOKEN_BUDGET, MIZU_STOP_REASON_STOP_SEQUENCE
  public :: MIZU_STOP_REASON_CANCELLED
  public :: MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_MASK_APPLE_ANE
  public :: MIZU_BACKEND_MASK_APPLE_METAL, MIZU_BACKEND_MASK_CUDA
  public :: MIZU_RUNTIME_FLAG_NONE, MIZU_MODEL_FLAG_NONE
  public :: MIZU_SESSION_FLAG_NONE, MIZU_ATTACH_FLAG_NONE
  public :: MIZU_INPUT_FLAG_NONE, MIZU_DECODE_FLAG_NONE
  public :: MIZU_STOP_FLAG_NONE, MIZU_OUTPUT_FLAG_NONE
  public :: MIZU_MODEL_FEATURE_NONE, MIZU_MODEL_FEATURE_MULTIMODAL
  public :: MIZU_MODEL_FEATURE_PROJECTOR
  public :: MIZU_SESSION_STATE_NONE, MIZU_SESSION_STATE_PENDING_INPUTS
  public :: MIZU_SESSION_STATE_LIVE_CONTEXT, MIZU_SESSION_STATE_PARKED
  public :: MIZU_CACHE_FLAG_NONE, MIZU_CACHE_FLAG_WEIGHT_HIT
  public :: MIZU_CACHE_FLAG_PLAN_HIT, MIZU_CACHE_FLAG_SESSION_HIT
  public :: MIZU_CACHE_FLAG_MM_HIT, MIZU_CACHE_FLAG_WINNER_REUSED
  public :: SOURCE_FORMAT_UNKNOWN, SOURCE_FORMAT_MIZU_MANIFEST
  public :: SOURCE_FORMAT_BUILTIN_TARGET, SOURCE_FORMAT_MIZU_IMPORT_BUNDLE
  public :: runtime_handle, model_handle, session_handle, workspace_handle
  public :: backend_descriptor, tensor_descriptor, projector_descriptor, import_tensor_state
  public :: runtime_config, model_open_config, session_config
  public :: model_info, session_info, modal_input_descriptor
  public :: decode_options, decode_result, output_buffer
  public :: execution_report, runtime_state, model_state
  public :: session_state, workspace_state
  public :: MAX_RUNTIME_BACKENDS
  public :: MAX_RECENT_OUTPUT_TOKENS, MAX_LIVE_CONTEXT_BYTES

  integer(i32), parameter :: MIZU_ABI_VERSION = int(z'00010000', kind=i32)

  integer(i32), parameter :: MIZU_OPTIMIZATION_MODE_DISABLED       = 0_i32
  integer(i32), parameter :: MIZU_OPTIMIZATION_MODE_MEASURE_ONLY   = 1_i32
  integer(i32), parameter :: MIZU_OPTIMIZATION_MODE_LEARN_AND_REUSE = 2_i32

  integer(i32), parameter :: MIZU_BACKEND_FAMILY_NONE  = 0_i32
  integer(i32), parameter :: MIZU_BACKEND_FAMILY_APPLE = 1_i32
  integer(i32), parameter :: MIZU_BACKEND_FAMILY_CUDA  = 2_i32

  integer(i32), parameter :: MIZU_EXEC_ROUTE_NONE  = 0_i32
  integer(i32), parameter :: MIZU_EXEC_ROUTE_ANE   = 1_i32
  integer(i32), parameter :: MIZU_EXEC_ROUTE_METAL = 2_i32
  integer(i32), parameter :: MIZU_EXEC_ROUTE_CUDA  = 3_i32

  integer(i32), parameter :: MIZU_MODEL_FAMILY_UNKNOWN = 0_i32
  integer(i32), parameter :: MIZU_MODEL_FAMILY_QWEN3_5 = 1_i32
  integer(i32), parameter :: MIZU_MODEL_FAMILY_GEMMA4  = 2_i32

  integer(i32), parameter :: MIZU_STAGE_NONE       = 0_i32
  integer(i32), parameter :: MIZU_STAGE_MODEL_LOAD = 1_i32
  integer(i32), parameter :: MIZU_STAGE_PROJECTOR  = 2_i32
  integer(i32), parameter :: MIZU_STAGE_PREFILL    = 3_i32
  integer(i32), parameter :: MIZU_STAGE_DECODE     = 4_i32
  integer(i32), parameter :: MIZU_STAGE_PARK       = 5_i32
  integer(i32), parameter :: MIZU_STAGE_RESUME     = 6_i32

  integer(i32), parameter :: MIZU_SELECTION_MODE_NONE        = 0_i32
  integer(i32), parameter :: MIZU_SELECTION_MODE_DIRECT      = 1_i32
  integer(i32), parameter :: MIZU_SELECTION_MODE_EXPLORATORY = 2_i32
  integer(i32), parameter :: MIZU_SELECTION_MODE_REUSE       = 3_i32

  integer(i32), parameter :: MIZU_COLD_STATE_UNKNOWN = 0_i32
  integer(i32), parameter :: MIZU_COLD_STATE_COLD    = 1_i32
  integer(i32), parameter :: MIZU_COLD_STATE_WARM    = 2_i32

  integer(i32), parameter :: MIZU_FALLBACK_REASON_NONE                = 0_i32
  integer(i32), parameter :: MIZU_FALLBACK_REASON_UNSUPPORTED_OP      = 1_i32
  integer(i32), parameter :: MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE   = 2_i32
  integer(i32), parameter :: MIZU_FALLBACK_REASON_BACKEND_UNAVAILABLE = 3_i32
  integer(i32), parameter :: MIZU_FALLBACK_REASON_ROUTE_DISALLOWED    = 4_i32
  integer(i32), parameter :: MIZU_FALLBACK_REASON_PLANNER_POLICY      = 5_i32

  integer(i32), parameter :: MIZU_SAMPLER_KIND_NONE       = 0_i32
  integer(i32), parameter :: MIZU_SAMPLER_KIND_GREEDY     = 1_i32
  integer(i32), parameter :: MIZU_SAMPLER_KIND_TOP_K_TOP_P = 2_i32

  integer(i32), parameter :: MIZU_MODALITY_KIND_UNKNOWN              = 0_i32
  integer(i32), parameter :: MIZU_MODALITY_KIND_IMAGE                = 1_i32
  integer(i32), parameter :: MIZU_MODALITY_KIND_TENSOR               = 2_i32
  integer(i32), parameter :: MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS = 3_i32

  integer(i32), parameter :: MIZU_STORAGE_KIND_UNKNOWN              = 0_i32
  integer(i32), parameter :: MIZU_STORAGE_KIND_ENCODED_BYTES        = 1_i32
  integer(i32), parameter :: MIZU_STORAGE_KIND_HOST_TENSOR          = 2_i32
  integer(i32), parameter :: MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS = 3_i32

  integer(i32), parameter :: MIZU_DTYPE_UNKNOWN = 0_i32
  integer(i32), parameter :: MIZU_DTYPE_U8      = 1_i32
  integer(i32), parameter :: MIZU_DTYPE_I32     = 2_i32
  integer(i32), parameter :: MIZU_DTYPE_F16     = 3_i32
  integer(i32), parameter :: MIZU_DTYPE_BF16    = 4_i32
  integer(i32), parameter :: MIZU_DTYPE_F32     = 5_i32

  integer(i32), parameter :: MIZU_LIFETIME_POLICY_COPY                = 0_i32
  integer(i32), parameter :: MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL = 1_i32

  integer(i32), parameter :: MIZU_OUTPUT_KIND_NONE     = 0_i32
  integer(i32), parameter :: MIZU_OUTPUT_KIND_TOKEN_IDS = 1_i32

  integer(i32), parameter :: MIZU_STOP_REASON_NONE          = 0_i32
  integer(i32), parameter :: MIZU_STOP_REASON_EOS           = 1_i32
  integer(i32), parameter :: MIZU_STOP_REASON_TOKEN_BUDGET  = 2_i32
  integer(i32), parameter :: MIZU_STOP_REASON_STOP_SEQUENCE = 3_i32
  integer(i32), parameter :: MIZU_STOP_REASON_CANCELLED     = 4_i32

  integer(i64), parameter :: MIZU_BACKEND_MASK_NONE        = 0_i64
  integer(i64), parameter :: MIZU_BACKEND_MASK_APPLE_ANE   = shiftl(1_i64, 0)
  integer(i64), parameter :: MIZU_BACKEND_MASK_APPLE_METAL = shiftl(1_i64, 1)
  integer(i64), parameter :: MIZU_BACKEND_MASK_CUDA        = shiftl(1_i64, 2)

  integer(i64), parameter :: MIZU_RUNTIME_FLAG_NONE = 0_i64
  integer(i64), parameter :: MIZU_MODEL_FLAG_NONE   = 0_i64
  integer(i64), parameter :: MIZU_SESSION_FLAG_NONE = 0_i64
  integer(i32), parameter :: MIZU_ATTACH_FLAG_NONE  = 0_i32
  integer(i64), parameter :: MIZU_INPUT_FLAG_NONE   = 0_i64
  integer(i64), parameter :: MIZU_DECODE_FLAG_NONE  = 0_i64
  integer(i64), parameter :: MIZU_STOP_FLAG_NONE    = 0_i64
  integer(i64), parameter :: MIZU_OUTPUT_FLAG_NONE  = 0_i64

  integer(i64), parameter :: MIZU_MODEL_FEATURE_NONE       = 0_i64
  integer(i64), parameter :: MIZU_MODEL_FEATURE_MULTIMODAL = shiftl(1_i64, 0)
  integer(i64), parameter :: MIZU_MODEL_FEATURE_PROJECTOR  = shiftl(1_i64, 1)

  integer(i64), parameter :: MIZU_SESSION_STATE_NONE           = 0_i64
  integer(i64), parameter :: MIZU_SESSION_STATE_PENDING_INPUTS = shiftl(1_i64, 0)
  integer(i64), parameter :: MIZU_SESSION_STATE_LIVE_CONTEXT   = shiftl(1_i64, 1)
  integer(i64), parameter :: MIZU_SESSION_STATE_PARKED         = shiftl(1_i64, 2)

  integer(i64), parameter :: MIZU_CACHE_FLAG_NONE          = 0_i64
  integer(i64), parameter :: MIZU_CACHE_FLAG_WEIGHT_HIT    = shiftl(1_i64, 0)
  integer(i64), parameter :: MIZU_CACHE_FLAG_PLAN_HIT      = shiftl(1_i64, 1)
  integer(i64), parameter :: MIZU_CACHE_FLAG_SESSION_HIT   = shiftl(1_i64, 2)
  integer(i64), parameter :: MIZU_CACHE_FLAG_MM_HIT        = shiftl(1_i64, 3)
  integer(i64), parameter :: MIZU_CACHE_FLAG_WINNER_REUSED = shiftl(1_i64, 4)

  integer(i32), parameter :: SOURCE_FORMAT_UNKNOWN            = 0_i32
  integer(i32), parameter :: SOURCE_FORMAT_MIZU_MANIFEST      = 1_i32
  integer(i32), parameter :: SOURCE_FORMAT_BUILTIN_TARGET     = 2_i32
  integer(i32), parameter :: SOURCE_FORMAT_MIZU_IMPORT_BUNDLE = 3_i32

  integer(i32), parameter :: MAX_RUNTIME_BACKENDS = 2_i32
  integer(i32), parameter :: MAX_MODEL_IMPORT_PREVIEW = 6_i32
  integer(i32), parameter :: MAX_RECENT_OUTPUT_TOKENS = 8_i32
  integer(i32), parameter :: MAX_LIVE_CONTEXT_BYTES = 912_i32

  type :: runtime_handle
    integer(i64) :: value = 0_i64
  end type runtime_handle

  type :: model_handle
    integer(i64) :: value = 0_i64
  end type model_handle

  type :: session_handle
    integer(i64) :: value = 0_i64
  end type session_handle

  type :: workspace_handle
    integer(i64) :: value = 0_i64
  end type workspace_handle

  type :: backend_descriptor
    integer(i32)                     :: family           = MIZU_BACKEND_FAMILY_NONE
    integer(i64)                     :: route_mask       = MIZU_BACKEND_MASK_NONE
    integer(i64)                     :: planner_version  = 0_i64
    logical                          :: is_available     = .false.
    character(len=MAX_NAME_LEN)      :: backend_name     = ""
    character(len=MAX_NAME_LEN)      :: device_name      = ""
  end type backend_descriptor

  type :: tensor_descriptor
    integer(i32)                     :: dtype            = MIZU_DTYPE_UNKNOWN
    integer(i32)                     :: rank             = 0_i32
    integer(i64)                     :: shape(MAX_TENSOR_RANK)  = 0_i64
    integer(i64)                     :: stride(MAX_TENSOR_RANK) = 0_i64
    integer(i64)                     :: byte_count       = 0_i64
    logical                          :: is_contiguous    = .false.
    character(len=MAX_NAME_LEN)      :: layout_name      = ""
  end type tensor_descriptor

  type :: projector_descriptor
    logical                          :: is_present          = .false.
    integer(i32)                     :: placeholder_count   = 0_i32
    integer(i32)                     :: input_dtype         = MIZU_DTYPE_UNKNOWN
    integer(i32)                     :: embedding_dtype     = MIZU_DTYPE_UNKNOWN
    character(len=MAX_SLOT_NAME_LEN) :: slot_name           = ""
  end type projector_descriptor

  type :: import_tensor_state
    integer(i32)                     :: dtype       = MIZU_DTYPE_UNKNOWN
    integer(i32)                     :: rank        = 0_i32
    integer(i64)                     :: shape(MAX_TENSOR_RANK) = 0_i64
    integer(i64)                     :: source_offset = -1_i64
    character(len=MAX_NAME_LEN)      :: tensor_name = ""
    character(len=MAX_NAME_LEN)      :: tensor_role = ""
    character(len=MAX_NAME_LEN)      :: layout_name = ""
    character(len=MAX_NAME_LEN)      :: storage_type = ""
    character(len=MAX_PATH_LEN)      :: source_path = ""
  end type import_tensor_state

  type :: runtime_config
    integer(i32)                     :: abi_version         = MIZU_ABI_VERSION
    integer(i32)                     :: optimization_mode   = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY
    integer(i32)                     :: exploration_budget  = 0_i32
    integer(i64)                     :: runtime_flags       = MIZU_RUNTIME_FLAG_NONE
    character(len=MAX_PATH_LEN)      :: cache_root          = ""
  end type runtime_config

  type :: model_open_config
    integer(i32)                     :: abi_version         = MIZU_ABI_VERSION
    integer(i64)                     :: allowed_backend_mask = MIZU_BACKEND_MASK_NONE
    integer(i64)                     :: model_flags         = MIZU_MODEL_FLAG_NONE
    character(len=MAX_PATH_LEN)      :: model_root          = ""
  end type model_open_config

  type :: session_config
    integer(i32)                     :: abi_version         = MIZU_ABI_VERSION
    integer(i64)                     :: max_context_tokens  = 0_i64
    integer(i64)                     :: max_decode_tokens   = 0_i64
    integer(i32)                     :: sampler_kind        = MIZU_SAMPLER_KIND_GREEDY
    integer(i64)                     :: seed                = 0_i64
    real(r32)                        :: temperature         = 0.0_r32
    integer(i32)                     :: top_k               = 0_i32
    real(r32)                        :: top_p               = 0.0_r32
    integer(i64)                     :: session_flags       = MIZU_SESSION_FLAG_NONE
  end type session_config

  type :: model_info
    integer(i32)                     :: model_family        = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i64)                     :: allowed_backend_mask = MIZU_BACKEND_MASK_NONE
    integer(i64)                     :: model_features      = MIZU_MODEL_FEATURE_NONE
    integer(i32)                     :: projector_slot_count = 0_i32
  end type model_info

  type :: session_info
    integer(i64)                     :: session_state_flags = MIZU_SESSION_STATE_NONE
    integer(i64)                     :: kv_token_count      = 0_i64
    integer(i64)                     :: staged_token_count  = 0_i64
    integer(i32)                     :: staged_modal_count  = 0_i32
  end type session_info

  type :: modal_input_descriptor
    integer(i32)                     :: placeholder_ordinal = 0_i32
    integer(i32)                     :: modality_kind       = MIZU_MODALITY_KIND_UNKNOWN
    integer(i32)                     :: storage_kind        = MIZU_STORAGE_KIND_UNKNOWN
    integer(i32)                     :: dtype               = MIZU_DTYPE_UNKNOWN
    integer(i32)                     :: rank                = 0_i32
    integer(i64)                     :: shape(MAX_TENSOR_RANK) = 0_i64
    integer(i64)                     :: byte_count          = 0_i64
    integer(i32)                     :: lifetime_policy     = MIZU_LIFETIME_POLICY_COPY
    integer(i64)                     :: input_flags         = MIZU_INPUT_FLAG_NONE
    character(len=MAX_SLOT_NAME_LEN) :: slot_name           = ""
  end type modal_input_descriptor

  type :: decode_options
    integer(i64)                     :: token_budget        = 0_i64
    integer(i64)                     :: stop_flags          = MIZU_STOP_FLAG_NONE
    integer(i64)                     :: decode_flags        = MIZU_DECODE_FLAG_NONE
  end type decode_options

  type :: decode_result
    integer(i64)                     :: token_capacity      = 0_i64
    integer(i64)                     :: token_count         = 0_i64
    integer(i32)                     :: stop_reason         = MIZU_STOP_REASON_NONE
    integer(i64)                     :: result_flags        = 0_i64
  end type decode_result

  type :: output_buffer
    integer(i32)                     :: output_kind         = MIZU_OUTPUT_KIND_NONE
    integer(i64)                     :: byte_capacity       = 0_i64
    integer(i64)                     :: bytes_written       = 0_i64
    integer(i64)                     :: output_flags        = MIZU_OUTPUT_FLAG_NONE
  end type output_buffer

  type :: execution_report
    integer(i32)                     :: stage_kind          = MIZU_STAGE_NONE
    integer(i32)                     :: backend_family      = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                     :: execution_route     = MIZU_EXEC_ROUTE_NONE
    integer(i64)                     :: plan_id             = 0_i64
    integer(i32)                     :: selection_mode      = MIZU_SELECTION_MODE_NONE
    integer(i32)                     :: cold_state          = MIZU_COLD_STATE_UNKNOWN
    integer(i32)                     :: fallback_reason     = MIZU_FALLBACK_REASON_NONE
    integer(i64)                     :: cache_flags         = MIZU_CACHE_FLAG_NONE
    integer(i64)                     :: elapsed_us          = 0_i64
  end type execution_report

  type :: workspace_state
    type(workspace_handle)            :: handle
    type(c_ptr)                       :: host_buffer         = c_null_ptr
    integer(i64)                      :: bytes_reserved      = 0_i64
    integer(i64)                      :: bytes_in_use        = 0_i64
    integer(i64)                      :: host_alignment_bytes = 0_i64
    integer(i64)                      :: allocation_count    = 0_i64
    logical                           :: is_ready            = .false.
  end type workspace_state

  type :: runtime_state
    type(runtime_handle)              :: handle
    type(runtime_config)              :: config
    type(workspace_state)             :: workspace
    integer(i32)                      :: last_status_code    = MIZU_STATUS_OK
    integer(i32)                      :: live_model_count    = 0_i32
    integer(i64)                      :: detected_backend_mask = MIZU_BACKEND_MASK_NONE
    integer(i32)                      :: detected_backend_count = 0_i32
    type(backend_descriptor)          :: detected_backends(MAX_RUNTIME_BACKENDS)
    logical                           :: is_initialized      = .false.
    character(len=MAX_ERROR_MESSAGE_LEN) :: last_error_message = ""
  end type runtime_state

  type :: model_state
    type(model_handle)                :: handle
    type(runtime_handle)              :: runtime_owner
    type(model_open_config)           :: open_config
    type(model_info)                  :: info
    type(execution_report)            :: last_report
    integer(i32)                      :: source_format       = SOURCE_FORMAT_UNKNOWN
    integer(i64)                      :: logical_model_hash  = 0_i64
    integer(i64)                      :: projector_revision  = 0_i64
    integer(i32)                      :: tensor_count        = 0_i32
    integer(i32)                      :: modality_count      = 0_i32
    character(len=MAX_NAME_LEN)       :: source_model_id     = ""
    logical                           :: has_import_bundle   = .false.
    integer(i64)                      :: import_inventory_hash = 0_i64
    integer(i64)                      :: import_tensor_bytes = 0_i64
    integer(i64)                      :: import_weight_pack_bytes = 0_i64
    integer(i64)                      :: import_weight_pack_hash = 0_i64
    integer(i64)                      :: import_projector_bytes = 0_i64
    integer(i32)                      :: import_weight_pack_count = 0_i32
    integer(i32)                      :: import_preview_count = 0_i32
    character(len=MAX_PATH_LEN)       :: import_projector_artifact_path = ""
    character(len=MAX_NAME_LEN)       :: import_tensor_names(MAX_MODEL_IMPORT_PREVIEW) = ""
    character(len=MAX_NAME_LEN)       :: import_tensor_roles(MAX_MODEL_IMPORT_PREVIEW) = ""
    character(len=MAX_PATH_LEN)       :: import_tensor_paths(MAX_MODEL_IMPORT_PREVIEW) = ""
    type(import_tensor_state), allocatable :: import_tensors(:)
    integer(i32)                      :: live_session_count  = 0_i32
    logical                           :: is_open             = .false.
  end type model_state

  type :: session_state
    type(session_handle)              :: handle
    type(model_handle)                :: model_owner
    type(session_config)              :: config
    type(execution_report)            :: last_report
    integer(i64)                      :: kv_token_count      = 0_i64
    integer(i64)                      :: live_context_hash   = 0_i64
    integer(i32)                      :: live_context_backend_family = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                      :: live_context_execution_route = MIZU_EXEC_ROUTE_NONE
    integer(i32)                      :: live_context_producer_stage = MIZU_STAGE_NONE
    integer(i64)                      :: live_context_artifact_hash = 0_i64
    integer(i32)                      :: live_context_byte_count = 0_i32
    integer(i8)                       :: live_context_bytes(MAX_LIVE_CONTEXT_BYTES) = 0_i8
    logical                           :: has_resident_live_context = .false.
    integer(i64)                      :: staged_token_count  = 0_i64
    integer(i64)                      :: staged_token_hash   = 0_i64
    integer(i32), allocatable         :: staged_tokens(:)
    integer(i64)                      :: last_output_token_count = 0_i64
    integer(i32)                      :: last_output_tokens(MAX_RECENT_OUTPUT_TOKENS) = 0_i32
    integer(i32)                      :: staged_modal_count  = 0_i32
    integer(i64)                      :: staged_modal_byte_count = 0_i64
    integer(i64)                      :: staged_modal_hash   = 0_i64
    integer(i8), allocatable          :: staged_modal_bytes(:)
    integer(i32)                      :: staged_modal_kind   = MIZU_MODALITY_KIND_UNKNOWN
    integer(i32)                      :: staged_modal_dtype  = MIZU_DTYPE_UNKNOWN
    character(len=MAX_SLOT_NAME_LEN)  :: staged_modal_slot_name = ""
    integer(i32)                      :: last_stop_reason    = MIZU_STOP_REASON_NONE
    logical                           :: is_open             = .false.
    logical                           :: has_pending_inputs  = .false.
    logical                           :: has_live_context    = .false.
    logical                           :: is_parked           = .false.
    logical                           :: has_decode_result   = .false.
    logical                           :: is_evicted          = .false.
  end type session_state

end module mod_types
