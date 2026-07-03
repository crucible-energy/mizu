program test_plan_cache
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,          only: MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                                MIZU_DTYPE_BF16, MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_CUDA, &
                                MIZU_STAGE_PREFILL
  use mod_model_manifest, only: model_manifest
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_cache_keys,     only: invalidation_version_fields, plan_cache_key, build_plan_cache_key
  use mod_cache_store,    only: artifact_metadata_record
  use mod_plan_cache,     only: runtime_plan_cache, plan_cache_record, &
                                initialize_runtime_plan_cache, reset_runtime_plan_cache, &
                                record_plan_cache_entry, lookup_plan_cache_entry, &
                                load_runtime_plan_cache, save_runtime_plan_cache, &
                                warm_runtime_plan_cache, &
                                plan_cache_key_is_strict

  implicit none

  type(model_manifest)             :: manifest
  type(invalidation_version_fields) :: versions
  type(runtime_plan_cache)         :: cache
  type(runtime_plan_cache)         :: reloaded_cache
  type(runtime_plan_cache)         :: warmed_cache
  type(plan_cache_key)             :: cuda_key
  type(plan_cache_key)             :: route_changed_key
  type(plan_cache_key)             :: planner_changed_key
  type(plan_cache_key)             :: malformed_key
  type(plan_cache_record)          :: record
  type(artifact_metadata_record)   :: metadata
  integer(i64)                     :: shape(3)
  integer(i64)                     :: warmed_count
  integer(i32)                     :: status_code
  logical                          :: found
  logical                          :: saved_ok
  logical                          :: loaded_ok
  character(len=*), parameter      :: cache_path = "/tmp/mizu_test_plan_cache.txt"

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i32("load multimodal fixture", status_code, MIZU_STATUS_OK)

  versions%planner_version = 7_i32
  versions%pack_version = 3_i32
  versions%backend_version = 11_i32
  shape = [1_i64, 256_i64, 4608_i64]

  call build_plan_cache_key(manifest, "cuda sm80", "cuda pack v1", MIZU_STAGE_PREFILL, &
    MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, MIZU_DTYPE_BF16, 3_i32, shape, cuda_key, versions)
  call build_plan_cache_key(manifest, "cuda sm80", "cuda pack v1", MIZU_STAGE_PREFILL, &
    MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, MIZU_DTYPE_BF16, 3_i32, shape, &
    route_changed_key, versions)

  versions%planner_version = 8_i32
  call build_plan_cache_key(manifest, "cuda sm80", "cuda pack v1", MIZU_STAGE_PREFILL, &
    MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, MIZU_DTYPE_BF16, 3_i32, shape, &
    planner_changed_key, versions)

  metadata%stage_kind = MIZU_STAGE_PREFILL
  metadata%backend_family = MIZU_BACKEND_FAMILY_CUDA
  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%workspace_bytes = 4096_i64
  metadata%payload_path = "cache/prefill plan.plan"

  call initialize_runtime_plan_cache(cache)
  call expect_true("generated key should be strict", plan_cache_key_is_strict(cuda_key))

  call lookup_plan_cache_entry(cache, cuda_key, record, found)
  call expect_false("empty plan cache should miss", found)

  call record_plan_cache_entry(cache, cuda_key, 101_i64, metadata, status_code, &
    candidate_key_text="prefill candidate cuda")
  call expect_equal_i32("record strict plan entry", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("plan cache entry count", cache%entry_count, 1_i32)

  call lookup_plan_cache_entry(cache, cuda_key, record, found)
  call expect_true("matching strict key should hit", found)
  call expect_equal_i64("plan cache hit plan id", record%plan_id, 101_i64)
  call expect_equal_i64("plan cache should increment hit count", record%hit_count, 1_i64)
  call expect_equal_i64("plan cache should preserve metadata workspace", &
    record%artifact_metadata%workspace_bytes, 4096_i64)
  call expect_equal_string("plan cache should preserve candidate key", &
    record%candidate_key_text, "prefill candidate cuda")

  call lookup_plan_cache_entry(cache, route_changed_key, record, found)
  call expect_false("route-changed key should miss", found)

  call lookup_plan_cache_entry(cache, planner_changed_key, record, found)
  call expect_false("planner-version-changed key should miss", found)

  metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  call record_plan_cache_entry(cache, cuda_key, 202_i64, metadata, status_code)
  call expect_equal_i32("mismatched artifact metadata should be rejected", &
    status_code, MIZU_STATUS_INVALID_ARGUMENT)
  call expect_equal_i32("rejected metadata should not add entries", cache%entry_count, 1_i32)

  malformed_key = cuda_key
  malformed_key%key_text = ""
  call record_plan_cache_entry(cache, malformed_key, 202_i64, artifact_metadata_record(), status_code)
  call expect_equal_i32("malformed plan key should be rejected", status_code, MIZU_STATUS_INVALID_ARGUMENT)

  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  call record_plan_cache_entry(cache, cuda_key, 303_i64, metadata, status_code, &
    candidate_key_text="prefill candidate cuda v2")
  call expect_equal_i32("same strict key should update existing entry", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("same strict key update should preserve entry count", cache%entry_count, 1_i32)
  call lookup_plan_cache_entry(cache, cuda_key, record, found)
  call expect_true("updated strict key should still hit", found)
  call expect_equal_i64("updated strict key should replace plan id", record%plan_id, 303_i64)
  call expect_equal_i64("updated strict key should preserve hit history", record%hit_count, 2_i64)
  call expect_equal_string("updated strict key should preserve candidate key", &
    record%candidate_key_text, "prefill candidate cuda v2")

  call execute_command_line("rm -f " // cache_path)
  call save_runtime_plan_cache(cache, cache_path, saved_ok)
  call expect_true("plan cache save should succeed", saved_ok)

  call initialize_runtime_plan_cache(reloaded_cache)
  call load_runtime_plan_cache(reloaded_cache, cache_path, loaded_ok)
  call expect_true("plan cache load should succeed", loaded_ok)
  call expect_equal_i32("plan cache load should restore one entry", reloaded_cache%entry_count, 1_i32)
  call lookup_plan_cache_entry(reloaded_cache, cuda_key, record, found)
  call expect_true("reloaded plan cache should hit strict key", found)
  call expect_equal_i64("reloaded plan cache should restore plan id", record%plan_id, 303_i64)
  call expect_equal_i64("reloaded lookup should advance persisted hit count", record%hit_count, 3_i64)
  call expect_equal_string("reloaded plan cache should restore candidate key", &
    record%candidate_key_text, "prefill candidate cuda v2")
  call expect_equal_string("reloaded plan cache should restore payload path", &
    record%artifact_metadata%payload_path, "cache/prefill plan.plan")

  metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  call initialize_runtime_plan_cache(warmed_cache)
  call record_plan_cache_entry(warmed_cache, route_changed_key, 404_i64, metadata, status_code)
  call expect_equal_i32("warm target seed entry should record", status_code, MIZU_STATUS_OK)
  call warm_runtime_plan_cache(warmed_cache, cache_path, warmed_count, loaded_ok)
  call expect_true("plan cache warm should load persisted entries", loaded_ok)
  call expect_equal_i64("plan cache warm should report loaded entries", warmed_count, 1_i64)
  call expect_equal_i32("plan cache warm should merge instead of replacing", warmed_cache%entry_count, 2_i32)
  call lookup_plan_cache_entry(warmed_cache, cuda_key, record, found)
  call expect_true("warmed plan cache should hit loaded strict key", found)
  call expect_equal_i64("warmed plan cache should restore plan id", record%plan_id, 303_i64)
  call lookup_plan_cache_entry(warmed_cache, route_changed_key, record, found)
  call expect_true("warmed plan cache should keep existing entries", found)
  call expect_equal_i64("warmed plan cache should keep existing plan id", record%plan_id, 404_i64)
  call execute_command_line("rm -f " // cache_path)

  call reset_runtime_plan_cache(cache)
  call lookup_plan_cache_entry(cache, cuda_key, record, found)
  call expect_false("reset plan cache should clear entries", found)

  write(*, "(A)") "test_plan_cache: PASS"

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

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine expect_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label) // " mismatch"
      error stop 1
    end if
  end subroutine expect_equal_string

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_false(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_false

end program test_plan_cache
