program test_weight_cache
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,          only: MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                                MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_CUDA, MIZU_STAGE_MODEL_LOAD
  use mod_model_manifest, only: model_manifest
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_cache_keys,     only: invalidation_version_fields, weight_cache_key, build_weight_cache_key
  use mod_cache_store,    only: artifact_metadata_record
  use mod_weight_cache,   only: runtime_weight_cache, weight_cache_record, &
                                initialize_runtime_weight_cache, reset_runtime_weight_cache, &
                                record_weight_cache_entry, lookup_weight_cache_entry, &
                                load_runtime_weight_cache, save_runtime_weight_cache, &
                                warm_runtime_weight_cache, weight_cache_key_is_strict

  implicit none

  type(model_manifest)             :: manifest
  type(invalidation_version_fields) :: versions
  type(runtime_weight_cache)       :: cache
  type(runtime_weight_cache)       :: reloaded_cache
  type(runtime_weight_cache)       :: warmed_cache
  type(weight_cache_key)           :: cuda_key
  type(weight_cache_key)           :: backend_changed_key
  type(weight_cache_key)           :: route_changed_key
  type(weight_cache_key)           :: pack_changed_key
  type(weight_cache_key)           :: malformed_key
  type(weight_cache_record)        :: record
  type(artifact_metadata_record)   :: metadata
  integer(i64)                     :: warmed_count
  integer(i32)                     :: status_code
  logical                          :: found
  logical                          :: saved_ok
  logical                          :: loaded_ok
  character(len=*), parameter      :: cache_path = "/tmp/mizu_test_weight_cache.txt"
  character(len=*), parameter      :: blocked_cache_path = "/tmp/mizu_test_weight_cache_blocked.txt"

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i32("load multimodal fixture", status_code, MIZU_STATUS_OK)

  versions%planner_version = 7_i32
  versions%pack_version = 3_i32
  versions%backend_version = 11_i32

  call build_weight_cache_key(manifest, 'cuda "sm80"', 'cuda "pack" v1', &
    MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, cuda_key, versions)
  versions%backend_version = 12_i32
  call build_weight_cache_key(manifest, 'cuda "sm80"', 'cuda "pack" v1', &
    MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, backend_changed_key, versions)
  versions%backend_version = 11_i32
  call build_weight_cache_key(manifest, 'cuda "sm80"', 'cuda "pack" v1', &
    MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, route_changed_key, versions)
  call build_weight_cache_key(manifest, 'cuda "sm80"', 'cuda "pack" v2', &
    MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, pack_changed_key, versions)

  metadata%stage_kind = MIZU_STAGE_MODEL_LOAD
  metadata%backend_family = MIZU_BACKEND_FAMILY_CUDA
  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%is_materialized = .true.
  metadata%payload_bytes = 1048576_i64
  metadata%workspace_bytes = 2097152_i64
  metadata%artifact_format = 'cuda "pack" v1'
  metadata%payload_fingerprint = 'PACK "A"'
  metadata%payload_path = 'cache/weights/PACK "A".pack'

  call initialize_runtime_weight_cache(cache)
  call expect_true("generated weight key should be strict", weight_cache_key_is_strict(cuda_key))

  call lookup_weight_cache_entry(cache, cuda_key, record, found)
  call expect_false("empty weight cache should miss", found)

  call record_weight_cache_entry(cache, cuda_key, metadata, status_code)
  call expect_equal_i32("record strict weight entry", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("weight cache entry count", cache%entry_count, 1_i32)

  call lookup_weight_cache_entry(cache, cuda_key, record, found)
  call expect_true("matching strict weight key should hit", found)
  call expect_equal_i64("weight cache should increment hit count", record%hit_count, 1_i64)
  call expect_equal_string("weight cache should derive pack identity", &
    record%pack_identity_text, 'PACK "A"')
  call expect_equal_i64("weight cache should preserve payload bytes", &
    record%artifact_metadata%payload_bytes, 1048576_i64)

  call lookup_weight_cache_entry(cache, route_changed_key, record, found)
  call expect_false("route-changed weight key should miss", found)
  call lookup_weight_cache_entry(cache, backend_changed_key, record, found)
  call expect_false("backend-version-changed weight key should miss", found)
  call lookup_weight_cache_entry(cache, pack_changed_key, record, found)
  call expect_false("pack-format-changed weight key should miss", found)

  metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  call record_weight_cache_entry(cache, cuda_key, metadata, status_code)
  call expect_equal_i32("mismatched weight metadata should be rejected", &
    status_code, MIZU_STATUS_INVALID_ARGUMENT)
  call expect_equal_i32("rejected weight metadata should not add entries", cache%entry_count, 1_i32)

  malformed_key = cuda_key
  malformed_key%key_text = ""
  call record_weight_cache_entry(cache, malformed_key, artifact_metadata_record(), status_code)
  call expect_equal_i32("malformed weight key should be rejected", status_code, MIZU_STATUS_INVALID_ARGUMENT)
  malformed_key = cuda_key
  malformed_key%key_text = trim(remove_backend_version_segment(cuda_key%key_text))
  call expect_false("weight key missing backend version should not be strict", &
    weight_cache_key_is_strict(malformed_key))

  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%payload_fingerprint = 'PACK "B"'
  metadata%payload_path = 'cache/weights/PACK "B".pack'
  call record_weight_cache_entry(cache, cuda_key, metadata, status_code, &
    pack_identity_text='PACK "B"')
  call expect_equal_i32("same strict weight key should update existing entry", &
    status_code, MIZU_STATUS_OK)
  call expect_equal_i32("same strict weight key update should preserve entry count", &
    cache%entry_count, 1_i32)
  call lookup_weight_cache_entry(cache, cuda_key, record, found)
  call expect_true("updated strict weight key should still hit", found)
  call expect_equal_string("updated strict weight key should replace pack identity", &
    record%pack_identity_text, 'PACK "B"')
  call expect_equal_i64("updated strict weight key should preserve hit history", record%hit_count, 2_i64)

  call execute_command_line("rm -f " // cache_path)
  call save_runtime_weight_cache(cache, cache_path, saved_ok)
  call expect_true("weight cache save should succeed", saved_ok)

  call initialize_runtime_weight_cache(reloaded_cache)
  call load_runtime_weight_cache(reloaded_cache, cache_path, loaded_ok)
  call expect_true("weight cache load should succeed", loaded_ok)
  call expect_equal_i32("weight cache load should restore one entry", reloaded_cache%entry_count, 1_i32)
  call lookup_weight_cache_entry(reloaded_cache, cuda_key, record, found)
  call expect_true("reloaded weight cache should hit strict key", found)
  call expect_equal_string("reloaded weight cache should restore pack identity", &
    record%pack_identity_text, 'PACK "B"')
  call expect_equal_i64("reloaded lookup should advance persisted hit count", record%hit_count, 3_i64)
  call expect_equal_string("reloaded weight cache should restore payload path", &
    record%artifact_metadata%payload_path, 'cache/weights/PACK "B".pack')

  metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  metadata%artifact_format = 'cuda "pack" v1'
  metadata%payload_fingerprint = 'PACK "C"'
  metadata%payload_path = 'cache/weights/PACK "C".pack'
  call initialize_runtime_weight_cache(warmed_cache)
  call record_weight_cache_entry(warmed_cache, route_changed_key, metadata, status_code, &
    pack_identity_text='PACK "C"')
  call expect_equal_i32("warm target seed weight entry should record", status_code, MIZU_STATUS_OK)
  call warm_runtime_weight_cache(warmed_cache, cache_path, warmed_count, loaded_ok)
  call expect_true("weight cache warm should load persisted entries", loaded_ok)
  call expect_equal_i64("weight cache warm should report loaded entries", warmed_count, 1_i64)
  call expect_equal_i32("weight cache warm should merge instead of replacing", warmed_cache%entry_count, 2_i32)
  call lookup_weight_cache_entry(warmed_cache, cuda_key, record, found)
  call expect_true("warmed weight cache should hit loaded strict key", found)
  call expect_equal_string("warmed weight cache should restore pack identity", &
    record%pack_identity_text, 'PACK "B"')
  call lookup_weight_cache_entry(warmed_cache, route_changed_key, record, found)
  call expect_true("warmed weight cache should keep existing entries", found)
  call expect_equal_string("warmed weight cache should keep existing pack identity", &
    record%pack_identity_text, 'PACK "C"')

  metadata%backend_family = MIZU_BACKEND_FAMILY_CUDA
  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%artifact_format = 'cuda "pack" v1'
  metadata%payload_fingerprint = "-"
  metadata%payload_path = "-"
  call initialize_runtime_weight_cache(cache)
  call record_weight_cache_entry(cache, cuda_key, metadata, status_code, pack_identity_text="-")
  call expect_equal_i32("literal dash weight entry should record", status_code, MIZU_STATUS_OK)
  call execute_command_line("rm -f " // cache_path)
  call save_runtime_weight_cache(cache, cache_path, saved_ok)
  call expect_true("dash-valued weight cache save should succeed", saved_ok)
  call initialize_runtime_weight_cache(reloaded_cache)
  call load_runtime_weight_cache(reloaded_cache, cache_path, loaded_ok)
  call expect_true("dash-valued weight cache load should succeed", loaded_ok)
  call lookup_weight_cache_entry(reloaded_cache, cuda_key, record, found)
  call expect_true("dash-valued weight cache should hit strict key", found)
  call expect_equal_string("dash-valued weight pack identity should round-trip", &
    record%pack_identity_text, "-")
  call expect_equal_string("dash-valued weight format should stay aligned to key", &
    record%artifact_metadata%artifact_format, 'cuda "pack" v1')
  call expect_equal_string("dash-valued weight fingerprint should round-trip", &
    record%artifact_metadata%payload_fingerprint, "-")
  call expect_equal_string("dash-valued weight path should round-trip", &
    record%artifact_metadata%payload_path, "-")
  call execute_command_line("rm -f " // cache_path)

  call write_legacy_weight_cache_entry(cache_path, cuda_key)
  call initialize_runtime_weight_cache(reloaded_cache)
  call load_runtime_weight_cache(reloaded_cache, cache_path, loaded_ok)
  call expect_true("legacy weight cache load should succeed", loaded_ok)
  call lookup_weight_cache_entry(reloaded_cache, cuda_key, record, found)
  call expect_true("legacy weight cache should preserve strict key", found)
  call expect_equal_string("legacy weight cache should preserve derived pack identity", &
    record%pack_identity_text, 'PACK LEGACY')
  call expect_equal_string("legacy weight cache should restore empty artifact format", &
    record%artifact_metadata%artifact_format, "")
  call expect_equal_string("legacy weight cache should restore empty fingerprint", &
    record%artifact_metadata%payload_fingerprint, "")
  call expect_equal_string("legacy weight cache should restore empty path", &
    record%artifact_metadata%payload_path, "")
  call execute_command_line("rm -f " // cache_path)

  call initialize_runtime_weight_cache(cache)
  call record_weight_cache_entry(cache, cuda_key, metadata, status_code, pack_identity_text="PACK RELOAD")
  call expect_equal_i32("seed weight cache before unreadable reload", status_code, MIZU_STATUS_OK)
  call prepare_unreadable_file(blocked_cache_path)
  call load_runtime_weight_cache(cache, blocked_cache_path, loaded_ok)
  call expect_false("weight cache load from unreadable file should fail", loaded_ok)
  call lookup_weight_cache_entry(cache, cuda_key, record, found)
  call expect_false("failed replace-load should clear stale weight entries", found)
  call execute_command_line("rm -f " // blocked_cache_path)

  call reset_runtime_weight_cache(cache)
  call lookup_weight_cache_entry(cache, cuda_key, record, found)
  call expect_false("reset weight cache should clear entries", found)

  write(*, "(A)") "test_weight_cache: PASS"

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

  subroutine write_legacy_weight_cache_entry(file_path, key)
    character(len=*), intent(in) :: file_path
    type(weight_cache_key), intent(in) :: key
    integer(i32) :: unit_id
    integer(i32) :: ios

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write", iostat=ios)
    call expect_equal_i32("legacy weight cache fixture should open", ios, 0_i32)
    write(unit_id, "(A,1X,A,1X,I0,1X,A,5(1X,I0),2(1X,I0),3(1X,I0),2(1X,A),4(1X,I0),2(1X,I0),3(1X,A))", &
        iostat=ios) &
      "entry", trim(quote_field(key%key_text)), 9_i64, trim(quote_field('PACK LEGACY')), &
      key%versions%schema_version, key%versions%abi_version, key%versions%planner_version, &
      key%versions%pack_version, key%versions%backend_version, key%logical_model_hash, &
      key%projector_revision, key%model_family, key%backend_family, key%execution_route, &
      trim(quote_field(key%device_key)), trim(quote_field(key%pack_format)), key%backend_family, &
      key%execution_route, MIZU_STAGE_MODEL_LOAD, 1_i32, 1048576_i64, 2097152_i64, "-", "-", "-"
    call expect_equal_i32("legacy weight cache fixture should write", ios, 0_i32)
    close(unit_id)
  end subroutine write_legacy_weight_cache_entry

  function quote_field(text) result(quoted_text)
    character(len=*), intent(in) :: text
    character(len=(2 * len(text)) + 2) :: quoted_text
    integer(i32) :: src_index
    integer(i32) :: dest_index

    quoted_text = '""'
    dest_index = 0_i32
    do src_index = 1_i32, len_trim(text)
      dest_index = dest_index + 1_i32
      quoted_text(dest_index + 1_i32:dest_index + 1_i32) = text(src_index:src_index)
      if (text(src_index:src_index) == '"') then
        dest_index = dest_index + 1_i32
        quoted_text(dest_index + 1_i32:dest_index + 1_i32) = '"'
      end if
    end do
    if (dest_index > 0_i32) quoted_text = '"' // quoted_text(2:dest_index + 1_i32) // '"'
  end function quote_field

  function remove_backend_version_segment(key_text) result(rewritten_key)
    character(len=*), intent(in) :: key_text
    character(len=len(key_text)) :: rewritten_key
    integer(i32) :: start_index
    integer(i32) :: end_index

    rewritten_key = key_text
    start_index = index(key_text, ":backendv=")
    if (start_index <= 0_i32) return

    end_index = index(key_text(start_index + 1:), ":")
    if (end_index <= 0_i32) then
      rewritten_key = key_text(:start_index - 1_i32)
      return
    end if

    end_index = start_index + end_index - 1_i32
    rewritten_key = key_text(:start_index - 1_i32) // key_text(end_index:)
  end function remove_backend_version_segment

  subroutine prepare_unreadable_file(path)
    character(len=*), intent(in) :: path
    integer(i32) :: exitstat

    call execute_command_line("rm -rf " // trim(path), exitstat=exitstat)
    call expect_equal_i32("remove stale blocked weight cache path", exitstat, 0_i32)
    call execute_command_line("touch " // trim(path), exitstat=exitstat)
    call expect_equal_i32("create blocked weight cache file", exitstat, 0_i32)
    call execute_command_line("chmod 000 " // trim(path), exitstat=exitstat)
    call expect_equal_i32("chmod blocked weight cache file", exitstat, 0_i32)
  end subroutine prepare_unreadable_file

end program test_weight_cache
