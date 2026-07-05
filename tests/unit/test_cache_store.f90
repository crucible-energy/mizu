program test_cache_store
  use mod_types,       only: MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, &
                             MIZU_EXEC_ROUTE_METAL, MIZU_STAGE_MODEL_LOAD, &
                             MIZU_STAGE_PROJECTOR, MIZU_STAGE_PREFILL
  use mod_cache_store, only: artifact_metadata_record, runtime_cache_bundle, &
                             initialize_runtime_cache_bundle, load_runtime_cache_bundle, &
                             save_runtime_cache_bundle, touch_weight_cache_key, &
                             touch_plan_cache_key, touch_session_cache_key, &
                             touch_multimodal_cache_key, &
                             record_weight_artifact_metadata, record_plan_artifact_metadata, &
                             record_session_artifact_metadata, record_multimodal_artifact_metadata, &
                             lookup_weight_artifact_metadata, lookup_plan_artifact_metadata, &
                             lookup_session_artifact_metadata, lookup_multimodal_artifact_metadata

  implicit none

  type(runtime_cache_bundle)    :: bundle
  type(runtime_cache_bundle)    :: reloaded_bundle
  type(artifact_metadata_record) :: weight_metadata
  type(artifact_metadata_record) :: plan_metadata
  type(artifact_metadata_record) :: session_metadata
  type(artifact_metadata_record) :: multimodal_metadata
  type(artifact_metadata_record) :: reloaded_metadata
  logical                       :: was_hit
  logical                       :: saved_ok
  logical                       :: loaded_ok
  logical                       :: found
  character(len=*), parameter :: store_path = "/tmp/mizu_test_artifact_cache.txt"
  character(len=*), parameter :: blocked_store_path = "/tmp/mizu_test_artifact_cache_blocked.txt"
  character(len=*), parameter :: partial_store_path = "/tmp/mizu_test_artifact_cache_partial.txt"

  call initialize_runtime_cache_bundle(bundle)
  call touch_weight_cache_key(bundle, 'weight key "ane"', was_hit)
  call expect_false("first weight touch should miss", was_hit)
  call touch_plan_cache_key(bundle, 'plan key "metal"', was_hit)
  call expect_false("first plan touch should miss", was_hit)
  call touch_session_cache_key(bundle, 'session key "cuda"', was_hit)
  call expect_false("first session touch should miss", was_hit)
  call touch_multimodal_cache_key(bundle, 'mm key "ane"', was_hit)
  call expect_false("first multimodal touch should miss", was_hit)

  weight_metadata = artifact_metadata_record()
  weight_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  weight_metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  weight_metadata%stage_kind = MIZU_STAGE_MODEL_LOAD
  weight_metadata%artifact_format = 'apple ane weight "pack" v1'
  weight_metadata%workspace_bytes = 1048576_8
  weight_metadata%payload_fingerprint = '1111 "AAAA"'
  weight_metadata%payload_path = 'artifacts/apple/ane/weights/1111 "AAAA".pack'
  call record_weight_artifact_metadata(bundle, 'weight key "ane"', weight_metadata)

  plan_metadata = artifact_metadata_record()
  plan_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  plan_metadata%execution_route = MIZU_EXEC_ROUTE_METAL
  plan_metadata%stage_kind = MIZU_STAGE_PREFILL
  plan_metadata%is_materialized = .true.
  plan_metadata%payload_bytes = 4096
  plan_metadata%workspace_bytes = 8388608_8
  plan_metadata%artifact_format = 'apple metal prefill "plan" v1'
  plan_metadata%payload_fingerprint = '2222 "BBBB"'
  plan_metadata%payload_path = 'artifacts/apple/metal/plans/prefill/2222 "BBBB".plan'
  call record_plan_artifact_metadata(bundle, 'plan key "metal"', plan_metadata)

  session_metadata = artifact_metadata_record()
  session_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  session_metadata%execution_route = MIZU_EXEC_ROUTE_METAL
  session_metadata%stage_kind = MIZU_STAGE_PREFILL
  session_metadata%is_materialized = .true.
  session_metadata%payload_bytes = 512
  session_metadata%workspace_bytes = 4096_8
  session_metadata%artifact_format = 'apple metal session "checkpoint" v1'
  session_metadata%payload_fingerprint = '2A2A "2A2A"'
  session_metadata%payload_path = 'artifacts/apple/metal/sessions/2A2A "2A2A".session'
  call record_session_artifact_metadata(bundle, 'session key "cuda"', session_metadata)

  multimodal_metadata = artifact_metadata_record()
  multimodal_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  multimodal_metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  multimodal_metadata%stage_kind = MIZU_STAGE_PROJECTOR
  multimodal_metadata%artifact_format = 'apple ane projector "cache" v1'
  multimodal_metadata%workspace_bytes = 2097152_8
  multimodal_metadata%payload_fingerprint = '3333 "CCCC"'
  multimodal_metadata%payload_path = 'artifacts/apple/ane/projector/3333 "CCCC".mm'
  call record_multimodal_artifact_metadata(bundle, 'mm key "ane"', multimodal_metadata)

  call execute_command_line("rm -f " // store_path)
  call save_runtime_cache_bundle(bundle, store_path, saved_ok)
  call expect_true("artifact cache save should succeed", saved_ok)

  call initialize_runtime_cache_bundle(reloaded_bundle)
  call load_runtime_cache_bundle(reloaded_bundle, store_path, loaded_ok)
  call expect_true("artifact cache load should succeed", loaded_ok)

  call touch_weight_cache_key(reloaded_bundle, 'weight key "ane"', was_hit)
  call expect_true("reloaded weight key should hit", was_hit)
  call touch_plan_cache_key(reloaded_bundle, 'plan key "metal"', was_hit)
  call expect_true("reloaded plan key should hit", was_hit)
  call touch_session_cache_key(reloaded_bundle, 'session key "cuda"', was_hit)
  call expect_true("reloaded session key should hit", was_hit)
  call touch_multimodal_cache_key(reloaded_bundle, 'mm key "ane"', was_hit)
  call expect_true("reloaded multimodal key should hit", was_hit)

  call lookup_weight_artifact_metadata(reloaded_bundle, 'weight key "ane"', reloaded_metadata, found)
  call expect_true("reloaded weight metadata should exist", found)
  call expect_equal_i32("weight metadata backend", reloaded_metadata%backend_family, MIZU_BACKEND_FAMILY_APPLE)
  call expect_equal_i32("weight metadata route", reloaded_metadata%execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_i32("weight metadata stage", reloaded_metadata%stage_kind, MIZU_STAGE_MODEL_LOAD)
  call expect_false("weight metadata should remain virtual", reloaded_metadata%is_materialized)
  call expect_equal_i64("weight metadata workspace", reloaded_metadata%workspace_bytes, 1048576_8)
  call expect_equal_string("weight metadata format", trim(reloaded_metadata%artifact_format), &
    'apple ane weight "pack" v1')
  call expect_equal_string("weight metadata path", trim(reloaded_metadata%payload_path), &
    'artifacts/apple/ane/weights/1111 "AAAA".pack')
  call expect_equal_string("weight metadata fingerprint", trim(reloaded_metadata%payload_fingerprint), &
    '1111 "AAAA"')

  call lookup_plan_artifact_metadata(reloaded_bundle, 'plan key "metal"', reloaded_metadata, found)
  call expect_true("reloaded plan metadata should exist", found)
  call expect_equal_i32("plan metadata route", reloaded_metadata%execution_route, MIZU_EXEC_ROUTE_METAL)
  call expect_equal_i32("plan metadata stage", reloaded_metadata%stage_kind, MIZU_STAGE_PREFILL)
  call expect_true("plan metadata should remain materialized", reloaded_metadata%is_materialized)
  call expect_equal_i64("plan metadata bytes", reloaded_metadata%payload_bytes, 4096_8)
  call expect_equal_i64("plan metadata workspace", reloaded_metadata%workspace_bytes, 8388608_8)
  call expect_equal_string("plan metadata format", trim(reloaded_metadata%artifact_format), &
    'apple metal prefill "plan" v1')
  call expect_equal_string("plan metadata fingerprint", trim(reloaded_metadata%payload_fingerprint), &
    '2222 "BBBB"')

  call lookup_session_artifact_metadata(reloaded_bundle, 'session key "cuda"', reloaded_metadata, found)
  call expect_true("reloaded session metadata should exist", found)
  call expect_true("session metadata should remain materialized", reloaded_metadata%is_materialized)
  call expect_equal_i64("session metadata bytes", reloaded_metadata%payload_bytes, 512_8)
  call expect_equal_string("session metadata path", trim(reloaded_metadata%payload_path), &
    'artifacts/apple/metal/sessions/2A2A "2A2A".session')

  call lookup_multimodal_artifact_metadata(reloaded_bundle, 'mm key "ane"', reloaded_metadata, found)
  call expect_true("reloaded multimodal metadata should exist", found)
  call expect_equal_i32("multimodal metadata stage", reloaded_metadata%stage_kind, MIZU_STAGE_PROJECTOR)
  call expect_equal_i64("multimodal metadata workspace", reloaded_metadata%workspace_bytes, 2097152_8)
  call expect_equal_string("multimodal metadata fingerprint", trim(reloaded_metadata%payload_fingerprint), &
    '3333 "CCCC"')

  call initialize_runtime_cache_bundle(bundle)
  call touch_weight_cache_key(bundle, "-", was_hit)
  call expect_false("literal dash weight key should start cold", was_hit)
  weight_metadata = artifact_metadata_record()
  weight_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  weight_metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  weight_metadata%stage_kind = MIZU_STAGE_MODEL_LOAD
  weight_metadata%artifact_format = "-"
  weight_metadata%payload_fingerprint = "-"
  weight_metadata%payload_path = "-"
  call record_weight_artifact_metadata(bundle, "-", weight_metadata)
  call execute_command_line("rm -f " // store_path)
  call save_runtime_cache_bundle(bundle, store_path, saved_ok)
  call expect_true("dash-valued artifact cache save should succeed", saved_ok)
  call initialize_runtime_cache_bundle(reloaded_bundle)
  call load_runtime_cache_bundle(reloaded_bundle, store_path, loaded_ok)
  call expect_true("dash-valued artifact cache load should succeed", loaded_ok)
  call touch_weight_cache_key(reloaded_bundle, "-", was_hit)
  call expect_true("literal dash weight key should round-trip", was_hit)
  call lookup_weight_artifact_metadata(reloaded_bundle, "-", reloaded_metadata, found)
  call expect_true("dash-valued weight metadata should exist", found)
  call expect_equal_string("dash-valued weight format should round-trip", &
    trim(reloaded_metadata%artifact_format), "-")
  call expect_equal_string("dash-valued weight fingerprint should round-trip", &
    trim(reloaded_metadata%payload_fingerprint), "-")
  call expect_equal_string("dash-valued weight path should round-trip", &
    trim(reloaded_metadata%payload_path), "-")

  call initialize_runtime_cache_bundle(bundle)
  call touch_weight_cache_key(bundle, 'stale weight key', was_hit)
  call expect_false("stale weight key seed should miss", was_hit)
  call record_weight_artifact_metadata(bundle, 'stale weight key', weight_metadata)
  call load_runtime_cache_bundle(bundle, store_path, loaded_ok)
  call expect_true("missing artifact cache load should succeed", loaded_ok)
  call touch_weight_cache_key(bundle, 'stale weight key', was_hit)
  call expect_false("missing artifact cache load should clear stale key", was_hit)
  call lookup_weight_artifact_metadata(bundle, 'stale weight key', reloaded_metadata, found)
  call expect_false("missing artifact cache load should clear stale metadata", found)

  call initialize_runtime_cache_bundle(bundle)
  call touch_plan_cache_key(bundle, 'stale plan key', was_hit)
  call expect_false("stale plan key seed should miss", was_hit)
  call record_plan_artifact_metadata(bundle, 'stale plan key', plan_metadata)
  call prepare_unreadable_file(blocked_store_path)
  call load_runtime_cache_bundle(bundle, blocked_store_path, loaded_ok)
  call expect_false("unreadable artifact cache load should fail", loaded_ok)
  call touch_plan_cache_key(bundle, 'stale plan key', was_hit)
  call expect_false("unreadable artifact cache load should clear stale key", was_hit)
  call lookup_plan_artifact_metadata(bundle, 'stale plan key', reloaded_metadata, found)
  call expect_false("unreadable artifact cache load should clear stale metadata", found)
  call execute_command_line("rm -f " // blocked_store_path)

  call write_partial_store_fixture(partial_store_path)
  call initialize_runtime_cache_bundle(reloaded_bundle)
  call load_runtime_cache_bundle(reloaded_bundle, partial_store_path, loaded_ok)
  call expect_true("partial artifact cache load should still succeed", loaded_ok)
  call touch_weight_cache_key(reloaded_bundle, 'partial weight key', was_hit)
  call expect_false("partial artifact cache should not create false weight hit", was_hit)
  call lookup_weight_artifact_metadata(reloaded_bundle, 'partial weight key', reloaded_metadata, found)
  call expect_false("partial artifact cache should not create weight metadata", found)
  call execute_command_line("rm -f " // partial_store_path)

  call execute_command_line("rm -f " // store_path)
  write(*, "(A)") "test_cache_store: PASS"

contains

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

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer, intent(in)          :: actual
    integer, intent(in)          :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(kind=8), intent(in)  :: actual
    integer(kind=8), intent(in)  :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine expect_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_string

  subroutine prepare_unreadable_file(file_path)
    character(len=*), intent(in) :: file_path
    integer :: exitstat

    call execute_command_line("rm -rf " // trim(file_path), exitstat=exitstat)
    call expect_equal_i32("remove stale blocked artifact cache path", exitstat, 0)
    call execute_command_line("touch " // trim(file_path), exitstat=exitstat)
    call expect_equal_i32("create blocked artifact cache file", exitstat, 0)
    call execute_command_line("chmod 000 " // trim(file_path), exitstat=exitstat)
    call expect_equal_i32("chmod blocked artifact cache file", exitstat, 0)
  end subroutine prepare_unreadable_file

  subroutine write_partial_store_fixture(file_path)
    character(len=*), intent(in) :: file_path
    integer :: unit_id

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write")
    write(unit_id, "(A)") 'weight "partial weight key"'
    close(unit_id)
  end subroutine write_partial_store_fixture

end program test_cache_store
