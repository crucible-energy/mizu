program test_session_staging
  use mod_kinds,   only: i8, i32, i64
  use mod_status,  only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_STATE
  use mod_types,   only: session_state, session_config, MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                         MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_CUDA
  use mod_session, only: initialize_session_state, stage_tokens, stage_modal_input, clear_pending_inputs, &
                         complete_prefill, complete_decode, store_live_context_record, &
                         update_live_context_record, offload_live_context_record, validate_decode

  implicit none

  type(session_state) :: session
  integer(i32)        :: status_code
  integer(i32)        :: tokens_a(3)
  integer(i32)        :: tokens_b(2)
  integer(i32)        :: expected_tokens_a(3)
  integer(i32)        :: expected_tokens_ab(5)
  integer(i32)        :: emitted_tokens(1)
  integer(i8)         :: context_bytes_a(6)
  integer(i8)         :: context_bytes_b(6)
  integer(i8)         :: modal_bytes(4)
  integer(i64)        :: live_context_hash_after_prefill

  tokens_a = [11_i32, 22_i32, 33_i32]
  tokens_b = [44_i32, 55_i32]
  expected_tokens_a = tokens_a
  expected_tokens_ab = [tokens_a, tokens_b]
  emitted_tokens = [101_i32]
  context_bytes_a = [11_i8, 22_i8, 33_i8, 44_i8, 55_i8, 66_i8]
  context_bytes_b = [21_i8, 32_i8, 43_i8, 54_i8, 65_i8, 76_i8]
  modal_bytes = [1_i8, 2_i8, 3_i8, 4_i8]

  call initialize_session_state(session, session_config())
  call stage_tokens(session, 3_i64, status_code, tokens_a)
  call expect_equal_i32("first staged token call should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("staged token count should reflect first batch", session%staged_token_count, 3_i64)
  call expect_true("staged tokens should allocate", allocated(session%staged_tokens))
  call expect_equal_i32("staged tokens should retain first value", session%staged_tokens(1), 11_i32)
  tokens_a = [-1_i32, -1_i32, -1_i32]
  call expect_true("staged tokens should not alias caller storage", all(session%staged_tokens == expected_tokens_a))
  call expect_true("staged token hash should become nonzero", session%staged_token_hash /= 0_i64)

  call stage_tokens(session, 2_i64, status_code, tokens_b)
  call expect_equal_i32("second staged token call should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("staged token count should append", session%staged_token_count, 5_i64)
  call expect_equal_i32("staged tokens should retain appended value", session%staged_tokens(5), 55_i32)
  tokens_b = [-2_i32, -2_i32]
  call expect_true("staged token batches should preserve append order", all(session%staged_tokens == expected_tokens_ab))

  call stage_modal_input(session, status_code, 4_i64, 1_i32, 1_i32, "image", modal_bytes)
  call expect_equal_i32("staged modal input should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("staged modal byte count should reflect copied bytes", session%staged_modal_byte_count, &
    4_i64)
  call expect_true("staged modal bytes should allocate", allocated(session%staged_modal_bytes))
  call expect_equal_i64("staged modal count should increment", int(session%staged_modal_count, kind=i64), 1_i64)
  call expect_equal_i32("staged modal bytes should retain final byte", int(session%staged_modal_bytes(4), kind=i32), &
    4_i32)
  call expect_true("staged modal hash should become nonzero", session%staged_modal_hash /= 0_i64)

  call clear_pending_inputs(session, status_code)
  call expect_equal_i32("clear pending inputs should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("staged token count should clear", session%staged_token_count, 0_i64)
  call expect_equal_i64("staged modal byte count should clear", session%staged_modal_byte_count, 0_i64)
  call expect_true("staged tokens should deallocate on clear", .not. allocated(session%staged_tokens))
  call expect_true("staged modal bytes should deallocate on clear", .not. allocated(session%staged_modal_bytes))
  call expect_equal_i64("staged token hash should clear", session%staged_token_hash, 0_i64)
  call expect_equal_i64("staged modal hash should clear", session%staged_modal_hash, 0_i64)

  tokens_a = expected_tokens_a
  call stage_tokens(session, 3_i64, status_code, tokens_a)
  call expect_equal_i32("restaged token call should succeed", status_code, MIZU_STATUS_OK)
  call stage_modal_input(session, status_code, 4_i64, 1_i32, 1_i32, "image", modal_bytes)
  call expect_equal_i32("restaged modal input should succeed", status_code, MIZU_STATUS_OK)

  call complete_prefill(session, consumed_token_count=3_i64, status_code=status_code, &
    token_content_hash=session%staged_token_hash, modal_content_hash=session%staged_modal_hash, &
    projector_embedding_count=2_i64)
  call expect_equal_i32("complete prefill should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("prefill should create live context", session%has_live_context)
  call expect_true("prefill should produce a live context hash", session%live_context_hash /= 0_i64)
  live_context_hash_after_prefill = session%live_context_hash
  call expect_true("prefill should clear staged tokens", .not. allocated(session%staged_tokens))
  call expect_true("prefill should clear staged modal bytes", .not. allocated(session%staged_modal_bytes))

  call store_live_context_record(session, MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, context_bytes_a, 6_i32)
  call expect_equal_i32("stored context buffer should retain backend family", &
    session%live_context_backend_family, MIZU_BACKEND_FAMILY_CUDA)
  call expect_equal_i32("stored context buffer should retain execution route", &
    session%live_context_execution_route, MIZU_EXEC_ROUTE_CUDA)
  call expect_equal_i32("stored context buffer should retain byte count", session%live_context_byte_count, 6_i32)
  call expect_equal_i32("stored context buffer should retain last byte", &
    int(session%live_context_bytes(6), kind=i32), 66_i32)

  call complete_decode(session, 1_i64, status_code=status_code, emitted_tokens=emitted_tokens)
  call expect_equal_i32("complete decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("decode should advance live context hash", session%live_context_hash /= live_context_hash_after_prefill)
  call expect_equal_i64("decode should update kv token count", session%kv_token_count, 4_i64)
  call expect_equal_i64("decode should expose one output token", session%last_output_token_count, 1_i64)
  call expect_equal_i32("decode should retain emitted token", session%last_output_tokens(1), emitted_tokens(1))

  call update_live_context_record(session, context_bytes_b, 6_i32)
  call expect_equal_i32("updated context buffer should keep byte count", session%live_context_byte_count, 6_i32)
  call expect_equal_i32("updated context buffer should overwrite first byte", &
    int(session%live_context_bytes(1), kind=i32), 21_i32)
  call expect_equal_i32("decode should remain valid with resident CUDA context", validate_decode(session), &
    MIZU_STATUS_OK)

  call offload_live_context_record(session)
  call expect_true("offloaded context buffer should clear residency", .not. session%has_resident_live_context)
  call expect_equal_i32("offloaded backend context should make decode invalid until restored", &
    validate_decode(session), MIZU_STATUS_INVALID_STATE)

  call store_live_context_record(session, MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, context_bytes_a, 6_i32)
  call expect_equal_i32("stored Apple context buffer should retain backend family", &
    session%live_context_backend_family, MIZU_BACKEND_FAMILY_APPLE)
  call expect_equal_i32("stored Apple context buffer should retain execution route", &
    session%live_context_execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_i32("resident Apple context should keep decode valid", validate_decode(session), &
    MIZU_STATUS_OK)

  call offload_live_context_record(session)
  call expect_equal_i32("offloaded Apple context should also make decode invalid until restored", &
    validate_decode(session), MIZU_STATUS_INVALID_STATE)

  write(*, "(A)") "test_session_staging: PASS"

contains

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i32), intent(in)     :: actual
    integer(i32), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i64

end program test_session_staging
