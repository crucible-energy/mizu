module mod_session
  use mod_kinds,  only: i8, i32, i64
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                        MIZU_STATUS_INVALID_STATE, MIZU_STATUS_SESSION_EVICTED
  use mod_types,  only: MIZU_SESSION_STATE_NONE, MIZU_SESSION_STATE_PENDING_INPUTS, &
                        MIZU_SESSION_STATE_LIVE_CONTEXT, MIZU_SESSION_STATE_PARKED, &
                        MIZU_STOP_REASON_NONE, MIZU_STAGE_NONE, MIZU_MODALITY_KIND_UNKNOWN, &
                        MIZU_DTYPE_UNKNOWN, session_config, session_info, &
                        session_state, execution_report, MIZU_BACKEND_FAMILY_NONE, &
                        MIZU_EXEC_ROUTE_NONE, MAX_LIVE_CONTEXT_BYTES

  implicit none

  private
  public :: initialize_session_state, reset_session_state
  public :: validate_attach_tokens, validate_attach_modal_input
  public :: validate_clear_pending_inputs, validate_prefill, prefill_would_exceed_context
  public :: validate_decode, validate_park, validate_resume
  public :: validate_read_output, decode_token_limit_reached
  public :: stage_tokens, stage_modal_input, clear_pending_inputs
  public :: complete_prefill, complete_decode, complete_decode_terminal
  public :: store_live_context_record, update_live_context_record, offload_live_context_record
  public :: park_session_state, resume_session_state
  public :: evict_parked_session, build_session_info

contains

  subroutine initialize_session_state(session, config)
    type(session_state), intent(out) :: session
    type(session_config), intent(in) :: config

    session%config             = config
    session%last_report        = execution_report()
    session%kv_token_count     = 0_i64
    session%live_context_hash  = 0_i64
    call clear_live_context_record(session)
    call clear_staged_inputs_state(session)
    session%decoded_token_count = 0_i64
    session%last_output_token_count = 0_i64
    session%last_output_tokens = 0_i32
    session%last_stop_reason   = MIZU_STOP_REASON_NONE
    session%is_open            = .true.
    session%has_pending_inputs = .false.
    session%has_live_context   = .false.
    session%is_parked          = .false.
    session%has_decode_result  = .false.
    session%is_evicted         = .false.
  end subroutine initialize_session_state

  subroutine reset_session_state(session)
    type(session_state), intent(inout) :: session

    call clear_staged_inputs_state(session)
    session%config                = session_config()
    session%last_report           = execution_report()
    session%kv_token_count        = 0_i64
    session%live_context_hash     = 0_i64
    call clear_live_context_record(session)
    session%decoded_token_count   = 0_i64
    session%last_output_token_count = 0_i64
    session%last_output_tokens    = 0_i32
    session%last_stop_reason      = MIZU_STOP_REASON_NONE
    session%is_open               = .false.
    session%has_pending_inputs    = .false.
    session%has_live_context      = .false.
    session%is_parked             = .false.
    session%has_decode_result     = .false.
    session%is_evicted            = .false.
  end subroutine reset_session_state

  pure integer(i32) function validate_attach_tokens(session) result(status_code)
    type(session_state), intent(in) :: session

    status_code = validate_attach_common(session)
  end function validate_attach_tokens

  pure integer(i32) function validate_attach_modal_input(session) result(status_code)
    type(session_state), intent(in) :: session

    status_code = validate_attach_common(session)
  end function validate_attach_modal_input

  pure integer(i32) function validate_clear_pending_inputs(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_clear_pending_inputs

  pure integer(i32) function validate_prefill(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_pending_inputs) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_prefill

  pure logical function prefill_would_exceed_context(session, token_count) result(would_exceed)
    type(session_state), intent(in) :: session
    integer(i64), intent(in)        :: token_count

    would_exceed = token_count < 0_i64
    if (would_exceed .or. session%config%max_context_tokens <= 0_i64) return
    if (session%kv_token_count < 0_i64 .or. session%kv_token_count > session%config%max_context_tokens) then
      would_exceed = .true.
      return
    end if
    would_exceed = token_count > session%config%max_context_tokens - session%kv_token_count
  end function prefill_would_exceed_context

  pure integer(i32) function validate_decode(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_live_context) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%live_context_byte_count > 0_i32 .and. &
             .not. session%has_resident_live_context) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%has_pending_inputs) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_decode

  pure integer(i32) function validate_park(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_live_context) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_park

  pure integer(i32) function validate_resume(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_evicted) then
      status_code = MIZU_STATUS_SESSION_EVICTED
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_resume

  pure integer(i32) function validate_read_output(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_decode_result) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_read_output

  pure logical function decode_token_limit_reached(session) result(is_reached)
    type(session_state), intent(in) :: session

    is_reached = session%config%max_decode_tokens > 0_i64 .and. &
      session%decoded_token_count >= session%config%max_decode_tokens
  end function decode_token_limit_reached

  subroutine stage_tokens(session, token_count, status_code, token_values)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: token_count
    integer(i32), intent(out)          :: status_code
    integer(i32), intent(in), optional :: token_values(:)

    status_code = validate_attach_tokens(session)
    if (status_code /= MIZU_STATUS_OK) return

    if (token_count <= 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (present(token_values)) then
      if (int(size(token_values), kind=i64) /= token_count) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if

      call append_staged_tokens(session, token_values)
      session%staged_token_hash = update_hash_i32_values(session%staged_token_hash, token_values)
    end if

    session%staged_token_count = session%staged_token_count + token_count
    session%has_pending_inputs = .true.
  end subroutine stage_tokens

  subroutine stage_modal_input(session, status_code, byte_count, modality_kind, dtype, slot_name, byte_values)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code
    integer(i64), intent(in), optional :: byte_count
    integer(i32), intent(in), optional :: modality_kind
    integer(i32), intent(in), optional :: dtype
    character(len=*), intent(in), optional :: slot_name
    integer(i8), intent(in), optional  :: byte_values(:)
    integer(i64)                       :: normalized_byte_count

    status_code = validate_attach_modal_input(session)
    if (status_code /= MIZU_STATUS_OK) return

    normalized_byte_count = 0_i64
    if (present(byte_count)) normalized_byte_count = max(0_i64, byte_count)
    if (present(byte_values)) then
      if (int(size(byte_values), kind=i64) /= normalized_byte_count) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if

      call append_staged_modal_bytes(session, byte_values)
      session%staged_modal_hash = update_hash_i8_values(session%staged_modal_hash, byte_values)
    end if

    session%staged_modal_count = session%staged_modal_count + 1_i32
    session%staged_modal_byte_count = session%staged_modal_byte_count + normalized_byte_count
    if (present(modality_kind)) then
      session%staged_modal_kind = modality_kind
    end if
    if (present(dtype)) then
      session%staged_modal_dtype = dtype
    end if
    if (present(slot_name)) then
      if (len_trim(slot_name) > 0) then
        session%staged_modal_slot_name = slot_name
      end if
    end if
    session%has_pending_inputs = .true.
  end subroutine stage_modal_input

  subroutine clear_pending_inputs(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_clear_pending_inputs(session)
    if (status_code /= MIZU_STATUS_OK) return

    call clear_staged_inputs_state(session)
    session%has_pending_inputs = .false.
  end subroutine clear_pending_inputs

  subroutine complete_prefill(session, consumed_token_count, status_code, token_content_hash, &
                              modal_content_hash, projector_embedding_count)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in), optional :: consumed_token_count
    integer(i32), intent(out)          :: status_code
    integer(i64), intent(in), optional :: token_content_hash
    integer(i64), intent(in), optional :: modal_content_hash
    integer(i64), intent(in), optional :: projector_embedding_count
    integer(i64)                       :: token_count
    integer(i64)                       :: effective_token_hash
    integer(i64)                       :: effective_modal_hash
    integer(i64)                       :: effective_embedding_count

    status_code = validate_prefill(session)
    if (status_code /= MIZU_STATUS_OK) return

    token_count = session%staged_token_count
    if (present(consumed_token_count)) token_count = consumed_token_count

    if (token_count < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if
    if (prefill_would_exceed_context(session, token_count)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    effective_token_hash = session%staged_token_hash
    if (present(token_content_hash)) effective_token_hash = token_content_hash
    effective_modal_hash = session%staged_modal_hash
    if (present(modal_content_hash)) effective_modal_hash = modal_content_hash
    effective_embedding_count = 0_i64
    if (present(projector_embedding_count)) effective_embedding_count = projector_embedding_count

    session%kv_token_count     = session%kv_token_count + token_count
    call update_live_context_after_prefill(session, token_count, effective_token_hash, effective_modal_hash, &
      effective_embedding_count)
    call clear_staged_inputs_state(session)
    session%has_pending_inputs = .false.
    session%has_live_context   = .true.
    session%is_parked          = .false.
    session%is_evicted         = .false.
    session%has_decode_result  = .false.
    session%last_output_token_count = 0_i64
    session%last_output_tokens = 0_i32
    session%last_stop_reason   = MIZU_STOP_REASON_NONE
  end subroutine complete_prefill

  subroutine complete_decode(session, emitted_token_count, stop_reason, status_code, emitted_tokens)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: emitted_token_count
    integer(i32), intent(in), optional :: stop_reason
    integer(i32), intent(out)          :: status_code
    integer(i32), intent(in), optional :: emitted_tokens(:)
    integer(i32)                       :: stored_token_count

    status_code = validate_decode(session)
    if (status_code /= MIZU_STATUS_OK) return

    if (emitted_token_count < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session%kv_token_count        = session%kv_token_count + emitted_token_count
    session%decoded_token_count   = session%decoded_token_count + emitted_token_count
    session%last_output_token_count = emitted_token_count
    session%has_decode_result     = .true.
    session%last_output_tokens    = 0_i32

    if (present(stop_reason)) then
      session%last_stop_reason = stop_reason
    else
      session%last_stop_reason = MIZU_STOP_REASON_NONE
    end if

    if (present(emitted_tokens)) then
      stored_token_count = min(size(emitted_tokens), int(min(emitted_token_count, &
        int(size(session%last_output_tokens), kind=i64)), kind=i32))
      if (stored_token_count > 0_i32) then
        session%last_output_tokens(1:stored_token_count) = emitted_tokens(1:stored_token_count)
      end if
    end if

    call update_live_context_after_decode(session, emitted_token_count, session%last_stop_reason, &
      session%last_output_tokens)
  end subroutine complete_decode

  subroutine complete_decode_terminal(session, stop_reason, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(in)           :: stop_reason
    integer(i32), intent(out)          :: status_code

    status_code = validate_decode(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%last_output_token_count = 0_i64
    session%last_output_tokens = 0_i32
    session%last_stop_reason = stop_reason
    session%has_decode_result = .true.
  end subroutine complete_decode_terminal

  subroutine park_session_state(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_park(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%is_parked = .true.
  end subroutine park_session_state

  subroutine resume_session_state(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_resume(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%is_parked = .false.
  end subroutine resume_session_state

  subroutine evict_parked_session(session)
    type(session_state), intent(inout) :: session

    session%is_evicted         = .true.
    session%is_parked          = .true.
    session%has_live_context   = .false.
    session%kv_token_count     = 0_i64
    session%live_context_hash  = 0_i64
    call clear_live_context_record(session)
    call clear_staged_inputs_state(session)
    session%has_decode_result  = .false.
    session%last_output_token_count = 0_i64
    session%last_output_tokens = 0_i32
  end subroutine evict_parked_session

  pure function build_session_info(session) result(info)
    type(session_state), intent(in) :: session
    type(session_info)              :: info

    info%session_state_flags = pack_session_state_flags(session)
    info%kv_token_count      = session%kv_token_count
    info%staged_token_count  = session%staged_token_count
    info%staged_modal_count  = session%staged_modal_count
  end function build_session_info

  pure integer(i32) function validate_attach_common(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_attach_common

  pure integer(i64) function pack_session_state_flags(session) result(state_flags)
    type(session_state), intent(in) :: session

    state_flags = MIZU_SESSION_STATE_NONE

    if (session%has_pending_inputs) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_PENDING_INPUTS)
    end if

    if (session%has_live_context) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_LIVE_CONTEXT)
    end if

    if (session%is_parked) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_PARKED)
    end if
  end function pack_session_state_flags

  subroutine clear_staged_inputs_state(session)
    type(session_state), intent(inout) :: session

    session%staged_token_count = 0_i64
    session%staged_token_hash = 0_i64
    if (allocated(session%staged_tokens)) deallocate(session%staged_tokens)
    session%staged_modal_count = 0_i32
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_hash = 0_i64
    if (allocated(session%staged_modal_bytes)) deallocate(session%staged_modal_bytes)
    session%staged_modal_kind  = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
  end subroutine clear_staged_inputs_state

  subroutine append_staged_tokens(session, token_values)
    type(session_state), intent(inout) :: session
    integer(i32), intent(in)           :: token_values(:)
    integer(i32), allocatable          :: resized_tokens(:)
    integer(i64)                       :: prior_count
    integer(i64)                       :: total_count

    if (size(token_values) == 0) return

    prior_count = 0_i64
    if (allocated(session%staged_tokens)) prior_count = int(size(session%staged_tokens), kind=i64)
    total_count = prior_count + int(size(token_values), kind=i64)
    allocate(resized_tokens(total_count))
    if (prior_count > 0_i64) resized_tokens(1:prior_count) = session%staged_tokens(1:prior_count)
    resized_tokens(prior_count + 1_i64:total_count) = token_values
    call move_alloc(resized_tokens, session%staged_tokens)
  end subroutine append_staged_tokens

  subroutine append_staged_modal_bytes(session, byte_values)
    type(session_state), intent(inout) :: session
    integer(i8), intent(in)            :: byte_values(:)
    integer(i8), allocatable           :: resized_bytes(:)
    integer(i64)                       :: prior_count
    integer(i64)                       :: total_count

    if (size(byte_values) == 0) return

    prior_count = 0_i64
    if (allocated(session%staged_modal_bytes)) prior_count = int(size(session%staged_modal_bytes), kind=i64)
    total_count = prior_count + int(size(byte_values), kind=i64)
    allocate(resized_bytes(total_count))
    if (prior_count > 0_i64) resized_bytes(1:prior_count) = session%staged_modal_bytes(1:prior_count)
    resized_bytes(prior_count + 1_i64:total_count) = byte_values
    call move_alloc(resized_bytes, session%staged_modal_bytes)
  end subroutine append_staged_modal_bytes

  pure integer(i64) function update_hash_i32_values(seed_hash, values) result(hash_value)
    integer(i64), intent(in) :: seed_hash
    integer(i32), intent(in) :: values(:)
    integer(i32)             :: index_value

    hash_value = ensure_nonzero_hash(seed_hash)
    do index_value = 1_i32, int(size(values), kind=i32)
      hash_value = hash_mix64(hash_value, int(values(index_value), kind=i64))
    end do
  end function update_hash_i32_values

  pure integer(i64) function update_hash_i8_values(seed_hash, values) result(hash_value)
    integer(i64), intent(in) :: seed_hash
    integer(i8), intent(in)  :: values(:)
    integer(i32)             :: index_value

    hash_value = ensure_nonzero_hash(seed_hash)
    do index_value = 1_i32, int(size(values), kind=i32)
      hash_value = hash_mix64(hash_value, int(values(index_value), kind=i64))
    end do
  end function update_hash_i8_values

  pure integer(i64) function ensure_nonzero_hash(seed_hash) result(hash_value)
    integer(i64), intent(in) :: seed_hash

    hash_value = seed_hash
    if (hash_value == 0_i64) hash_value = int(z'9E3779B97F4A7C15', kind=i64)
  end function ensure_nonzero_hash

  pure integer(i64) function hash_mix64(seed_hash, value) result(hash_value)
    integer(i64), intent(in) :: seed_hash
    integer(i64), intent(in) :: value
    integer(i64)             :: mixed_value

    mixed_value = value + int(z'9E3779B97F4A7C15', kind=i64)
    mixed_value = ieor(mixed_value, shiftr(mixed_value, 30))
    mixed_value = mixed_value * int(z'BF58476D1CE4E5B9', kind=i64)
    mixed_value = ieor(mixed_value, shiftr(mixed_value, 27))
    mixed_value = mixed_value * int(z'94D049BB133111EB', kind=i64)
    mixed_value = ieor(mixed_value, shiftr(mixed_value, 31))

    hash_value = ieor(seed_hash, mixed_value)
    if (hash_value == 0_i64) hash_value = 1_i64
  end function hash_mix64

  subroutine clear_live_context_record(session)
    type(session_state), intent(inout) :: session

    session%live_context_backend_family = MIZU_BACKEND_FAMILY_NONE
    session%live_context_execution_route = MIZU_EXEC_ROUTE_NONE
    session%live_context_producer_stage = MIZU_STAGE_NONE
    session%live_context_artifact_hash = 0_i64
    session%live_context_byte_count = 0_i32
    session%live_context_bytes = 0_i8
    session%has_resident_live_context = .false.
  end subroutine clear_live_context_record

  subroutine store_live_context_record(session, backend_family, execution_route, context_bytes, context_byte_count, &
                                       producer_stage, artifact_hash)
    type(session_state), intent(inout) :: session
    integer(i32), intent(in)           :: backend_family
    integer(i32), intent(in)           :: execution_route
    integer(i8), intent(in)            :: context_bytes(:)
    integer(i32), intent(in)           :: context_byte_count
    integer(i32), intent(in), optional :: producer_stage
    integer(i64), intent(in), optional :: artifact_hash
    integer(i32)                       :: stored_count

    call clear_live_context_record(session)
    session%live_context_backend_family = backend_family
    session%live_context_execution_route = execution_route
    if (present(producer_stage)) session%live_context_producer_stage = producer_stage
    if (present(artifact_hash)) session%live_context_artifact_hash = artifact_hash
    stored_count = max(0_i32, min(context_byte_count, min(int(size(context_bytes), kind=i32), MAX_LIVE_CONTEXT_BYTES)))
    session%live_context_byte_count = stored_count
    session%has_resident_live_context = (stored_count > 0_i32)
    if (stored_count <= 0_i32) return

    session%live_context_bytes(1:stored_count) = context_bytes(1:stored_count)
  end subroutine store_live_context_record

  subroutine update_live_context_record(session, context_bytes, context_byte_count, producer_stage, artifact_hash, &
                                        backend_family, execution_route)
    type(session_state), intent(inout) :: session
    integer(i8), intent(in)            :: context_bytes(:)
    integer(i32), intent(in)           :: context_byte_count
    integer(i32), intent(in), optional :: producer_stage
    integer(i64), intent(in), optional :: artifact_hash
    integer(i32), intent(in), optional :: backend_family
    integer(i32), intent(in), optional :: execution_route
    integer(i32)                       :: stored_count

    session%live_context_bytes = 0_i8
    if (present(producer_stage)) session%live_context_producer_stage = producer_stage
    if (present(artifact_hash)) session%live_context_artifact_hash = artifact_hash
    if (present(backend_family)) session%live_context_backend_family = backend_family
    if (present(execution_route)) session%live_context_execution_route = execution_route
    stored_count = max(0_i32, min(context_byte_count, min(int(size(context_bytes), kind=i32), MAX_LIVE_CONTEXT_BYTES)))
    session%live_context_byte_count = stored_count
    session%has_resident_live_context = (stored_count > 0_i32)
    if (stored_count > 0_i32) then
      session%live_context_bytes(1:stored_count) = context_bytes(1:stored_count)
    end if
  end subroutine update_live_context_record

  subroutine offload_live_context_record(session)
    type(session_state), intent(inout) :: session

    session%live_context_bytes = 0_i8
    session%has_resident_live_context = .false.
  end subroutine offload_live_context_record

  subroutine update_live_context_after_prefill(session, token_count, token_content_hash, modal_content_hash, &
                                               projector_embedding_count)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: token_count
    integer(i64), intent(in)           :: token_content_hash
    integer(i64), intent(in)           :: modal_content_hash
    integer(i64), intent(in)           :: projector_embedding_count

    session%live_context_hash = ensure_nonzero_hash(session%live_context_hash)
    session%live_context_hash = hash_mix64(session%live_context_hash, max(0_i64, token_count))
    if (token_content_hash /= 0_i64) then
      session%live_context_hash = hash_mix64(session%live_context_hash, token_content_hash)
    end if
    if (modal_content_hash /= 0_i64) then
      session%live_context_hash = hash_mix64(session%live_context_hash, modal_content_hash)
    end if
    if (projector_embedding_count > 0_i64) then
      session%live_context_hash = hash_mix64(session%live_context_hash, projector_embedding_count)
    end if
    session%live_context_hash = hash_mix64(session%live_context_hash, session%kv_token_count)
  end subroutine update_live_context_after_prefill

  subroutine update_live_context_after_decode(session, emitted_token_count, stop_reason, emitted_tokens)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: emitted_token_count
    integer(i32), intent(in)           :: stop_reason
    integer(i32), intent(in)           :: emitted_tokens(:)
    integer(i32)                       :: stored_token_count

    session%live_context_hash = ensure_nonzero_hash(session%live_context_hash)
    session%live_context_hash = hash_mix64(session%live_context_hash, max(0_i64, emitted_token_count))
    session%live_context_hash = hash_mix64(session%live_context_hash, int(stop_reason, kind=i64))

    stored_token_count = min(size(emitted_tokens), int(min(emitted_token_count, &
      int(size(session%last_output_tokens), kind=i64)), kind=i32))
    if (stored_token_count > 0_i32) then
      session%live_context_hash = update_hash_i32_values(session%live_context_hash, &
        emitted_tokens(1:stored_token_count))
    end if
    session%live_context_hash = hash_mix64(session%live_context_hash, session%kv_token_count)
  end subroutine update_live_context_after_decode

end module mod_session
