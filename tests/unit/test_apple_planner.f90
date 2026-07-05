program test_apple_planner
  use mod_kinds,            only: i32, i64
  use mod_status,           only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT
  use mod_types,            only: MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, MIZU_STAGE_PREFILL, &
                                  MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                                  MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_METAL, &
                                  MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_MASK_APPLE_ANE, &
                                  MIZU_BACKEND_MASK_APPLE_METAL, &
                                  MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE
  use mod_backend_contract, only: plan_request, planner_result, initialize_plan_request, OP_FAMILY_NONE, &
                                  OP_FAMILY_PROJECTOR, OP_FAMILY_PREFILL, planner_result_is_success
  use mod_apple_planner,    only: APPLE_ARTIFACT_PAYLOAD_LEN, plan_apple_stage, &
                                  build_apple_artifact_payload_text

  implicit none

  type(plan_request)    :: request
  type(planner_result)  :: result
  character(len=APPLE_ARTIFACT_PAYLOAD_LEN) :: payload_text
  integer(i64)          :: payload_bytes
  integer(i32)          :: status_code

  call initialize_plan_request(request, MIZU_STAGE_MODEL_LOAD, OP_FAMILY_NONE, MIZU_MODEL_FAMILY_QWEN3_5, &
    ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL))
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should accept model load requests", status_code, MIZU_STATUS_OK)
  call expect_true("apple planner should report a successful model-load result", planner_result_is_success(result))
  call expect_equal_i32("apple model-load planner should tag the Apple backend family", &
    result%chosen_plan%backend_family, MIZU_BACKEND_FAMILY_APPLE)
  call expect_equal_i32("apple planner should prefer ANE when both Apple routes are allowed", &
    result%chosen_plan%execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_i64("apple ANE model-load planner should use the expected packed-weight workspace", &
    result%chosen_plan%workspace_bytes, 201326592_i64)
  call expect_equal_text("apple ANE model-load planner should use the route-specific weight format", &
    trim(result%chosen_plan%pack_format), "apple_ane_bf16_weight_pack_v1")

  request%preferred_backend_mask = MIZU_BACKEND_MASK_APPLE_METAL
  request%model_family = MIZU_MODEL_FAMILY_GEMMA4
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should accept Metal-preferred model load requests", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("apple planner should honor an explicit Metal preference", &
    result%chosen_plan%execution_route, MIZU_EXEC_ROUTE_METAL)
  call expect_equal_i64("apple Metal model-load planner should use the expected packed-weight workspace", &
    result%chosen_plan%workspace_bytes, 536870912_i64)
  call expect_equal_text("apple Metal model-load planner should use the route-specific weight format", &
    trim(result%chosen_plan%pack_format), "apple_metal_bf16_weight_pack_v1")

  call initialize_plan_request(request, MIZU_STAGE_PROJECTOR, OP_FAMILY_PROJECTOR, MIZU_MODEL_FAMILY_QWEN3_5, &
    MIZU_BACKEND_MASK_APPLE_ANE)
  request%shape_signature(1) = 4096_i64
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should accept projector requests", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("apple projector planner should select ANE when ANE is the only allowed route", &
    result%chosen_plan%execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_text("apple projector planner should use the ANE projector format", &
    trim(result%chosen_plan%pack_format), "apple_ane_u8_bf16_projector_plan_v1")
  call build_apple_artifact_payload_text(request, result%chosen_plan, "candidate=apple_projector", payload_text, &
    payload_bytes)
  call expect_true("apple projector payload should record the Apple planner version token", &
    index(trim(payload_text), "planner=apple_v1") > 0)
  call expect_true("apple projector payload should record the ANE route token", &
    index(trim(payload_text), "route=apple_ane") > 0)
  call expect_true("apple projector payload should record the route-specific projector format", &
    index(trim(payload_text), "apple_ane_u8_bf16_projector_plan_v1") > 0)
  call expect_equal_i64("apple projector payload bytes should match the payload text length plus terminator", &
    payload_bytes, int(len_trim(payload_text) + 1, kind=i64))

  call initialize_plan_request(request, MIZU_STAGE_PREFILL, OP_FAMILY_PREFILL, MIZU_MODEL_FAMILY_GEMMA4, &
    MIZU_BACKEND_MASK_APPLE_METAL)
  request%preferred_backend_mask = MIZU_BACKEND_MASK_APPLE_METAL
  request%shape_signature = [2048_i64, 16_i64, 1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
  request%token_count = 3_i64
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple prefill planner should accept Metal requests", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("apple prefill planner should preserve the Metal route", &
    result%chosen_plan%execution_route, MIZU_EXEC_ROUTE_METAL)
  call expect_equal_text("apple prefill planner should use the Metal prefill format", &
    trim(result%chosen_plan%pack_format), "apple_metal_bf16_prefill_plan_v1")
  call expect_true("apple prefill planner should reserve nonzero workspace", result%chosen_plan%workspace_bytes > 0_i64)

  call initialize_plan_request(request, MIZU_STAGE_PREFILL, OP_FAMILY_PREFILL, MIZU_MODEL_FAMILY_GEMMA4, &
    ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL))
  request%shape_signature = [0_i64, 65_i64, 1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
  request%token_count = 65_i64
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should accept oversized prefill requests when Metal is allowed", &
    status_code, MIZU_STATUS_OK)
  call expect_equal_i32("apple oversized prefill should fall back to Metal", &
    result%chosen_plan%execution_route, MIZU_EXEC_ROUTE_METAL)
  call expect_true("apple oversized prefill should report fallback", result%requires_fallback)
  call expect_equal_i32("apple oversized prefill should report unsupported-shape fallback", &
    result%fallback_reason, MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE)

  call initialize_plan_request(request, MIZU_STAGE_PREFILL, OP_FAMILY_PREFILL, MIZU_MODEL_FAMILY_GEMMA4, &
    MIZU_BACKEND_MASK_APPLE_ANE)
  request%shape_signature = [0_i64, 65_i64, 1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
  request%token_count = 65_i64
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should reject oversized prefill requests when Metal is disallowed", &
    status_code, MIZU_STATUS_INVALID_ARGUMENT)

  call initialize_plan_request(request, MIZU_STAGE_PREFILL, OP_FAMILY_PREFILL, MIZU_MODEL_FAMILY_QWEN3_5, &
    MIZU_BACKEND_MASK_NONE)
  call plan_apple_stage(request, result, status_code)
  call expect_equal_i32("apple planner should reject requests without an Apple route", status_code, &
    MIZU_STATUS_INVALID_ARGUMENT)

  write(*, "(A)") "test_apple_planner: PASS"

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

  subroutine expect_equal_text(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_text

end program test_apple_planner
