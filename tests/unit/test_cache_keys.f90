program test_cache_keys
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_OK
  use mod_types,          only: MIZU_STAGE_PREFILL, MIZU_BACKEND_FAMILY_APPLE, &
                                MIZU_EXEC_ROUTE_ANE, MIZU_DTYPE_BF16, &
                                MIZU_MODALITY_KIND_IMAGE
  use mod_model_manifest, only: model_manifest
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_cache_keys,     only: invalidation_version_fields, plan_cache_key, &
                                weight_cache_key, session_cache_key, &
                                multimodal_cache_key, build_plan_cache_key, &
                                build_weight_cache_key, build_session_cache_key, &
                                build_multimodal_cache_key

  implicit none

  type(model_manifest)             :: manifest
  type(invalidation_version_fields) :: versions
  type(plan_cache_key)             :: plan_key_a
  type(plan_cache_key)             :: plan_key_b
  type(weight_cache_key)           :: weight_key
  type(weight_cache_key)           :: weight_key_backend_changed
  type(session_cache_key)          :: session_key
  type(multimodal_cache_key)       :: mm_key
  integer(i64)                     :: shape(3)
  integer(i32)                     :: status_code

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i32("load multimodal fixture", status_code, MIZU_STATUS_OK)

  versions%planner_version = 7_i32
  versions%pack_version = 3_i32
  versions%backend_version = 11_i32

  shape = [1_i64, 256_i64, 4608_i64]
  call build_plan_cache_key(manifest, "apple-m2", "packed-bf16", MIZU_STAGE_PREFILL, &
    MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, MIZU_DTYPE_BF16, 3_i32, shape, &
    plan_key_a, versions)
  call build_plan_cache_key(manifest, "apple-m2", "packed-bf16", MIZU_STAGE_PREFILL, &
    MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, MIZU_DTYPE_BF16, 3_i32, shape, &
    plan_key_b, versions)
  call expect_equal_string("deterministic plan key", trim(plan_key_a%key_text), trim(plan_key_b%key_text))

  call build_weight_cache_key(manifest, "apple-m2", "packed-bf16", MIZU_BACKEND_FAMILY_APPLE, &
    MIZU_EXEC_ROUTE_ANE, weight_key, versions)
  versions%backend_version = 12_i32
  call build_weight_cache_key(manifest, "apple-m2", "packed-bf16", MIZU_BACKEND_FAMILY_APPLE, &
    MIZU_EXEC_ROUTE_ANE, weight_key_backend_changed, versions)
  versions%backend_version = 11_i32
  call build_session_cache_key(manifest, "apple-m2", MIZU_BACKEND_FAMILY_APPLE, &
    MIZU_EXEC_ROUTE_ANE, 8192_i64, 512_i64, session_key, versions)
  call build_multimodal_cache_key(manifest, "apple-m2", "image", MIZU_MODALITY_KIND_IMAGE, &
    MIZU_DTYPE_BF16, 32768_i64, mm_key, versions)

  call expect_contains("plan key prefix", trim(plan_key_a%key_text), "plan:v")
  call expect_contains("weight key prefix", trim(weight_key%key_text), "weight:v")
  call expect_contains("weight key backend version", trim(weight_key%key_text), ":backendv=11")
  call expect_contains("session key prefix", trim(session_key%key_text), "session:v")
  call expect_contains("multimodal key prefix", trim(mm_key%key_text), "mm:v")
  call expect_not_equal_string("backend version should change weight key", &
    trim(weight_key%key_text), trim(weight_key_backend_changed%key_text))

  write(*, "(A)") "test_cache_keys: PASS"

contains

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i32), intent(in)     :: actual
    integer(i32), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label) // " mismatch"
      error stop 1
    end if
  end subroutine expect_equal_string

  subroutine expect_contains(label, haystack, needle)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: haystack
    character(len=*), intent(in) :: needle

    if (index(haystack, needle) <= 0) then
      write(*, "(A)") trim(label) // " missing: " // trim(needle)
      error stop 1
    end if
  end subroutine expect_contains

  subroutine expect_not_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) == trim(expected)) then
      write(*, "(A)") trim(label) // " unexpectedly matched"
      error stop 1
    end if
  end subroutine expect_not_equal_string

end program test_cache_keys
