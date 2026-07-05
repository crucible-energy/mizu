module mod_apple_planner
  use mod_kinds,            only: i32, i64, KILOBYTE, MEGABYTE
  use mod_status,           only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT
  use mod_types,            only: MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, &
                                  MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                                  MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                                  MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_NONE, &
                                  MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_METAL, &
                                  MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL, &
                                  MIZU_FALLBACK_REASON_NONE, MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE
  use mod_backend_contract, only: plan_request, plan_candidate, planner_result

  implicit none

  private
  public :: APPLE_ARTIFACT_PAYLOAD_LEN
  public :: plan_apple_stage, build_apple_artifact_payload_text

  integer(i32), parameter :: APPLE_ARTIFACT_PAYLOAD_LEN = 1024_i32
  integer(i64), parameter :: APPLE_ANE_PREFILL_TOKEN_LIMIT = 64_i64

contains

  subroutine plan_apple_stage(request, result, status_code)
    type(plan_request), intent(in)    :: request
    type(planner_result), intent(out) :: result
    integer(i32), intent(out)         :: status_code
    integer(i32)                      :: execution_route
    integer(i32)                      :: fallback_reason
    logical                           :: requires_fallback

    result = planner_result()
    execution_route = resolve_apple_route(request, requires_fallback, fallback_reason)
    if (.not. apple_stage_is_supported(request%stage_kind) .or. execution_route == MIZU_EXEC_ROUTE_NONE) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      result%status_code = status_code
      return
    end if

    result%status_code = MIZU_STATUS_OK
    result%requires_fallback = requires_fallback
    result%fallback_reason = fallback_reason
    result%candidate_count = 1_i32
    result%chosen_plan = plan_candidate()
    result%chosen_plan%backend_family = MIZU_BACKEND_FAMILY_APPLE
    result%chosen_plan%execution_route = execution_route
    result%chosen_plan%workspace_bytes = estimate_apple_workspace_bytes(request, execution_route)
    result%chosen_plan%pack_format = apple_pack_format_label(request%stage_kind, execution_route)

    status_code = MIZU_STATUS_OK
  end subroutine plan_apple_stage

  subroutine build_apple_artifact_payload_text(request, candidate, candidate_key_text, payload_text, &
                                               payload_bytes)
    type(plan_request), intent(in)    :: request
    type(plan_candidate), intent(in)  :: candidate
    character(len=*), intent(in)      :: candidate_key_text
    character(len=*), intent(out)     :: payload_text
    integer(i64), intent(out)         :: payload_bytes

    payload_text = ""
    write(payload_text, '(A,";planner=apple_v1;route=",A,";stage=",I0,";model=",I0,";shape0=",I0, &
      &";shape1=",I0,";shape2=",I0,";tokens=",I0,";workspace=",I0,";format=",A)') &
      trim(candidate_key_text), trim(apple_route_token(candidate%execution_route)), request%stage_kind, &
      request%model_family, request%shape_signature(1), request%shape_signature(2), &
      request%shape_signature(3), request%token_count, candidate%workspace_bytes, trim(candidate%pack_format)
    payload_bytes = int(len_trim(payload_text) + 1, kind=i64)
  end subroutine build_apple_artifact_payload_text

  pure logical function apple_stage_is_supported(stage_kind) result(is_supported)
    integer(i32), intent(in) :: stage_kind

    is_supported = (stage_kind == MIZU_STAGE_MODEL_LOAD .or. &
      stage_kind == MIZU_STAGE_PROJECTOR .or. &
      stage_kind == MIZU_STAGE_PREFILL .or. &
      stage_kind == MIZU_STAGE_DECODE)
  end function apple_stage_is_supported

  integer(i32) function resolve_apple_route(request, requires_fallback, fallback_reason) result(execution_route)
    type(plan_request), intent(in) :: request
    integer(i64)                   :: preferred_mask
    integer(i64)                   :: allowed_mask
    integer(i32), intent(out)      :: fallback_reason
    logical, intent(out)           :: requires_fallback
    integer(i32)                   :: preferred_route
    integer(i32)                   :: preferred_failure_reason
    integer(i32)                   :: ane_failure_reason

    preferred_mask = iand(request%preferred_backend_mask, ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL))
    allowed_mask = iand(request%allowed_backend_mask, ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL))

    execution_route = MIZU_EXEC_ROUTE_NONE
    fallback_reason = MIZU_FALLBACK_REASON_NONE
    requires_fallback = .false.
    if (.not. apple_stage_is_supported(request%stage_kind)) return

    preferred_route = preferred_apple_route(preferred_mask)
    if (preferred_route /= MIZU_EXEC_ROUTE_NONE) then
      if (apple_route_is_supported(request, preferred_route, preferred_failure_reason) .and. &
          apple_route_is_allowed(allowed_mask, preferred_route)) then
        execution_route = preferred_route
        return
      end if
      if (preferred_failure_reason /= MIZU_FALLBACK_REASON_NONE) fallback_reason = preferred_failure_reason
    end if

    if (iand(allowed_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64 .and. &
        apple_route_is_supported(request, MIZU_EXEC_ROUTE_ANE, ane_failure_reason)) then
      execution_route = MIZU_EXEC_ROUTE_ANE
      return
    end if

    if (iand(allowed_mask, MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64 .and. &
        apple_route_is_supported(request, MIZU_EXEC_ROUTE_METAL, preferred_failure_reason)) then
      execution_route = MIZU_EXEC_ROUTE_METAL
      if (iand(allowed_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64) then
        if (fallback_reason == MIZU_FALLBACK_REASON_NONE) fallback_reason = ane_failure_reason
        requires_fallback = (fallback_reason /= MIZU_FALLBACK_REASON_NONE)
      end if
    end if
  end function resolve_apple_route

  pure integer(i32) function preferred_apple_route(preferred_mask) result(execution_route)
    integer(i64), intent(in) :: preferred_mask

    execution_route = MIZU_EXEC_ROUTE_NONE
    if (iand(preferred_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64) then
      execution_route = MIZU_EXEC_ROUTE_ANE
      return
    end if
    if (iand(preferred_mask, MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64) then
      execution_route = MIZU_EXEC_ROUTE_METAL
    end if
  end function preferred_apple_route

  pure logical function apple_route_is_allowed(allowed_mask, execution_route) result(is_allowed)
    integer(i64), intent(in) :: allowed_mask
    integer(i32), intent(in) :: execution_route

    is_allowed = .false.
    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      is_allowed = (iand(allowed_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64)
    case (MIZU_EXEC_ROUTE_METAL)
      is_allowed = (iand(allowed_mask, MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64)
    end select
  end function apple_route_is_allowed

  logical function apple_route_is_supported(request, execution_route, failure_reason) result(is_supported)
    type(plan_request), intent(in) :: request
    integer(i32), intent(in)       :: execution_route
    integer(i32), intent(out)      :: failure_reason

    is_supported = .true.
    failure_reason = MIZU_FALLBACK_REASON_NONE
    if (execution_route /= MIZU_EXEC_ROUTE_ANE) return

    if (request%stage_kind == MIZU_STAGE_PREFILL .and. &
        max(0_i64, request%shape_signature(2)) > APPLE_ANE_PREFILL_TOKEN_LIMIT) then
      is_supported = .false.
      failure_reason = MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE
    end if
  end function apple_route_is_supported

  integer(i64) function estimate_apple_workspace_bytes(request, execution_route) result(workspace_bytes)
    type(plan_request), intent(in) :: request
    integer(i32), intent(in)       :: execution_route
    integer(i64)                   :: base_bytes

    select case (request%stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      base_bytes = apple_weight_pack_bytes(request%model_family, execution_route)
    case (MIZU_STAGE_PROJECTOR)
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        base_bytes = (6_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 128_i64
      else
        base_bytes = (12_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 256_i64
      end if
    case (MIZU_STAGE_PREFILL)
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        base_bytes = (12_i64 * MEGABYTE) + &
          (max(0_i64, request%shape_signature(1)) + max(0_i64, request%shape_signature(2))) * 1536_i64 + &
          max(0_i64, request%shape_signature(3)) * (6_i64 * MEGABYTE)
      else
        base_bytes = (20_i64 * MEGABYTE) + &
          (max(0_i64, request%shape_signature(1)) + max(0_i64, request%shape_signature(2))) * 3072_i64 + &
          max(0_i64, request%shape_signature(3)) * (8_i64 * MEGABYTE)
      end if
    case (MIZU_STAGE_DECODE)
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        base_bytes = (4_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 512_i64 + &
          max(1_i64, request%token_count) * (256_i64 * KILOBYTE)
      else
        base_bytes = (10_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 1024_i64 + &
          max(1_i64, request%token_count) * (768_i64 * KILOBYTE)
      end if
    case default
      base_bytes = 4_i64 * MEGABYTE
    end select

    workspace_bytes = align_bytes(max(1_i64 * MEGABYTE, base_bytes), 256_i64)
  end function estimate_apple_workspace_bytes

  integer(i64) function apple_weight_pack_bytes(model_family, execution_route) result(pack_bytes)
    integer(i32), intent(in) :: model_family
    integer(i32), intent(in) :: execution_route

    select case (model_family)
    case (MIZU_MODEL_FAMILY_QWEN3_5)
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        pack_bytes = 192_i64 * MEGABYTE
      else
        pack_bytes = 256_i64 * MEGABYTE
      end if
    case (MIZU_MODEL_FAMILY_GEMMA4)
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        pack_bytes = 384_i64 * MEGABYTE
      else
        pack_bytes = 512_i64 * MEGABYTE
      end if
    case default
      if (execution_route == MIZU_EXEC_ROUTE_ANE) then
        pack_bytes = 160_i64 * MEGABYTE
      else
        pack_bytes = 224_i64 * MEGABYTE
      end if
    end select
  end function apple_weight_pack_bytes

  function apple_pack_format_label(stage_kind, execution_route) result(pack_format)
    integer(i32), intent(in)  :: stage_kind
    integer(i32), intent(in)  :: execution_route
    character(len=128)        :: pack_format
    character(len=16)         :: route_token

    route_token = apple_route_token(execution_route)
    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      write(pack_format, '(A,"_bf16_weight_pack_v1")') trim(route_token)
    case (MIZU_STAGE_PROJECTOR)
      write(pack_format, '(A,"_u8_bf16_projector_plan_v1")') trim(route_token)
    case (MIZU_STAGE_PREFILL)
      write(pack_format, '(A,"_bf16_prefill_plan_v1")') trim(route_token)
    case (MIZU_STAGE_DECODE)
      write(pack_format, '(A,"_bf16_decode_plan_v1")') trim(route_token)
    case default
      write(pack_format, '(A,"_generic_plan_v1")') trim(route_token)
    end select
  end function apple_pack_format_label

  function apple_route_token(execution_route) result(route_token)
    integer(i32), intent(in) :: execution_route
    character(len=16)        :: route_token

    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      route_token = "apple_ane"
    case (MIZU_EXEC_ROUTE_METAL)
      route_token = "apple_metal"
    case default
      route_token = "apple"
    end select
  end function apple_route_token

  integer(i64) function align_bytes(byte_count, alignment) result(aligned_bytes)
    integer(i64), intent(in) :: byte_count
    integer(i64), intent(in) :: alignment
    integer(i64)             :: rounded_bytes

    if (alignment <= 0_i64) then
      aligned_bytes = byte_count
      return
    end if

    rounded_bytes = ((max(0_i64, byte_count) + alignment - 1_i64) / alignment) * alignment
    aligned_bytes = max(alignment, rounded_bytes)
  end function align_bytes

end module mod_apple_planner
