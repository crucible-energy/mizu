module mod_plan_cache
  use mod_kinds,       only: i32, i64, MAX_NAME_LEN, MAX_PATH_LEN, MAX_TENSOR_RANK
  use mod_status,      only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,       only: MIZU_BACKEND_FAMILY_NONE, MIZU_DTYPE_UNKNOWN, &
                             MIZU_EXEC_ROUTE_NONE, MIZU_STAGE_NONE
  use mod_cache_keys,  only: MAX_CACHE_KEY_LEN, plan_cache_key
  use mod_cache_store, only: artifact_metadata_record, normalize_legacy_persisted_field, &
                             quote_persisted_text

  implicit none

  private
  public :: plan_cache_record, runtime_plan_cache
  public :: initialize_runtime_plan_cache, reset_runtime_plan_cache
  public :: record_plan_cache_entry, lookup_plan_cache_entry
  public :: load_runtime_plan_cache, save_runtime_plan_cache, warm_runtime_plan_cache
  public :: plan_cache_key_is_strict

  integer(i32), parameter :: INITIAL_PLAN_CACHE_CAPACITY = 16_i32
  integer(i32), parameter :: MAX_PLAN_CACHE_RECORD_LINE_LEN = &
    (4_i32 * MAX_CACHE_KEY_LEN) + (8_i32 * MAX_NAME_LEN) + (2_i32 * MAX_PATH_LEN) + 512_i32

  type :: plan_cache_record
    type(plan_cache_key)             :: key
    integer(i64)                     :: plan_id = 0_i64
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text = ""
    type(artifact_metadata_record)   :: artifact_metadata
    integer(i64)                     :: hit_count = 0_i64
  end type plan_cache_record

  type :: runtime_plan_cache
    integer(i32) :: entry_count = 0_i32
    type(plan_cache_record), allocatable :: entries(:)
  end type runtime_plan_cache

contains

  subroutine initialize_runtime_plan_cache(cache)
    type(runtime_plan_cache), intent(out) :: cache

    cache = runtime_plan_cache()
  end subroutine initialize_runtime_plan_cache

  subroutine reset_runtime_plan_cache(cache)
    type(runtime_plan_cache), intent(inout) :: cache

    if (allocated(cache%entries)) deallocate(cache%entries)
    cache%entry_count = 0_i32
  end subroutine reset_runtime_plan_cache

  subroutine record_plan_cache_entry(cache, key, plan_id, artifact_metadata, status_code, &
                                     candidate_key_text)
    type(runtime_plan_cache), intent(inout)       :: cache
    type(plan_cache_key), intent(in)              :: key
    integer(i64), intent(in)                      :: plan_id
    type(artifact_metadata_record), intent(in)    :: artifact_metadata
    integer(i32), intent(out)                     :: status_code
    character(len=*), intent(in), optional        :: candidate_key_text
    integer(i32)                                  :: entry_index
    integer(i64)                                  :: existing_hits

    if (.not. plan_cache_key_is_strict(key) .or. plan_id == 0_i64 .or. &
        .not. metadata_matches_key(artifact_metadata, key)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    entry_index = ensure_entry_index(cache, trim(key%key_text))
    existing_hits = cache%entries(entry_index)%hit_count

    cache%entries(entry_index)%key = key
    cache%entries(entry_index)%plan_id = plan_id
    cache%entries(entry_index)%artifact_metadata = artifact_metadata
    cache%entries(entry_index)%hit_count = existing_hits
    cache%entries(entry_index)%candidate_key_text = ""
    if (present(candidate_key_text)) then
      cache%entries(entry_index)%candidate_key_text = trim(candidate_key_text)
    end if

    status_code = MIZU_STATUS_OK
  end subroutine record_plan_cache_entry

  subroutine lookup_plan_cache_entry(cache, key, record, found)
    type(runtime_plan_cache), intent(inout) :: cache
    type(plan_cache_key), intent(in)        :: key
    type(plan_cache_record), intent(out)    :: record
    logical, intent(out)                    :: found
    integer(i32)                            :: entry_index

    record = plan_cache_record()
    found = .false.
    if (.not. plan_cache_key_is_strict(key)) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = cache%entries(entry_index)%hit_count + 1_i64
    record = cache%entries(entry_index)
    found = .true.
  end subroutine lookup_plan_cache_entry

  subroutine load_runtime_plan_cache(cache, file_path, loaded_ok)
    type(runtime_plan_cache), intent(inout) :: cache
    character(len=*), intent(in)            :: file_path
    logical, intent(out)                    :: loaded_ok
    integer(i64)                            :: loaded_count

    call read_runtime_plan_cache_file(cache, file_path, .true., loaded_count, loaded_ok)
  end subroutine load_runtime_plan_cache

  subroutine warm_runtime_plan_cache(cache, file_path, warmed_count, loaded_ok)
    type(runtime_plan_cache), intent(inout) :: cache
    character(len=*), intent(in)            :: file_path
    integer(i64), intent(out)               :: warmed_count
    logical, intent(out)                    :: loaded_ok

    call read_runtime_plan_cache_file(cache, file_path, .false., warmed_count, loaded_ok)
  end subroutine warm_runtime_plan_cache

  subroutine save_runtime_plan_cache(cache, file_path, saved_ok)
    type(runtime_plan_cache), intent(in) :: cache
    character(len=*), intent(in)         :: file_path
    logical, intent(out)                 :: saved_ok
    integer(i32)                         :: unit_id
    integer(i32)                         :: ios
    integer(i32)                         :: index

    saved_ok = .false.
    if (len_trim(file_path) == 0) return

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return

    do index = 1_i32, cache%entry_count
      call write_plan_cache_record(unit_id, cache%entries(index), ios)
      if (ios /= 0_i32) then
        close(unit_id)
        return
      end if
    end do

    close(unit_id)
    saved_ok = .true.
  end subroutine save_runtime_plan_cache

  pure logical function plan_cache_key_is_strict(key) result(is_strict)
    type(plan_cache_key), intent(in) :: key

    is_strict = len_trim(key%key_text) > 0 .and. &
      key%versions%schema_version > 0_i32 .and. &
      key%versions%abi_version > 0_i32 .and. &
      key%logical_model_hash /= 0_i64 .and. &
      key%stage_kind /= MIZU_STAGE_NONE .and. &
      key%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
      key%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
      key%dtype /= MIZU_DTYPE_UNKNOWN .and. &
      key%rank >= 0_i32 .and. key%rank <= MAX_TENSOR_RANK .and. &
      len_trim(key%device_key) > 0 .and. &
      len_trim(key%pack_format) > 0 .and. &
      index(key%key_text, "plan:v") == 1 .and. &
      index(key%key_text, ":abi=" // trim(i32_to_text(key%versions%abi_version))) > 0 .and. &
      index(key%key_text, ":planner=" // trim(i32_to_text(key%versions%planner_version))) > 0 .and. &
      index(key%key_text, ":packv=" // trim(i32_to_text(key%versions%pack_version))) > 0 .and. &
      index(key%key_text, ":backendv=" // trim(i32_to_text(key%versions%backend_version))) > 0 .and. &
      index(key%key_text, ":model=" // trim(i64_to_text(key%logical_model_hash))) > 0 .and. &
      index(key%key_text, ":stage=" // trim(i32_to_text(key%stage_kind))) > 0 .and. &
      index(key%key_text, ":backend=" // trim(i32_to_text(key%backend_family))) > 0 .and. &
      index(key%key_text, ":route=" // trim(i32_to_text(key%execution_route))) > 0 .and. &
      index(key%key_text, ":dtype=" // trim(i32_to_text(key%dtype))) > 0 .and. &
      index(key%key_text, ":device=" // trim(key%device_key)) > 0 .and. &
      index(key%key_text, ":pack=" // trim(key%pack_format)) > 0 .and. &
      index(key%key_text, ":shape=") > 0
  end function plan_cache_key_is_strict

  pure logical function metadata_matches_key(metadata, key) result(matches)
    type(artifact_metadata_record), intent(in) :: metadata
    type(plan_cache_key), intent(in)           :: key

    matches = .true.
    if (metadata%stage_kind /= MIZU_STAGE_NONE .and. metadata%stage_kind /= key%stage_kind) matches = .false.
    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        metadata%backend_family /= key%backend_family) matches = .false.
    if (metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        metadata%execution_route /= key%execution_route) matches = .false.
  end function metadata_matches_key

  subroutine read_runtime_plan_cache_file(cache, file_path, replace_existing, loaded_count, loaded_ok)
    type(runtime_plan_cache), intent(inout) :: cache
    character(len=*), intent(in)            :: file_path
    logical, intent(in)                     :: replace_existing
    integer(i64), intent(out)               :: loaded_count
    logical, intent(out)                    :: loaded_ok
    character(len=MAX_PLAN_CACHE_RECORD_LINE_LEN) :: line
    character(len=16)                       :: tag
    type(plan_cache_key)                    :: key
    type(artifact_metadata_record)          :: metadata
    integer(i64)                            :: plan_id
    integer(i64)                            :: hit_count
    character(len=MAX_CACHE_KEY_LEN)        :: candidate_key_text
    integer(i32)                            :: materialized_flag
    integer(i32)                            :: unit_id
    integer(i32)                            :: ios
    integer(i32)                            :: shape_index
    logical                                 :: exists

    loaded_count = 0_i64
    loaded_ok = .false.
    if (len_trim(file_path) == 0) return
    if (replace_existing) call reset_runtime_plan_cache(cache)

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) then
      loaded_ok = .true.
      return
    end if

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return

    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit
      if (len_trim(line) == 0) cycle

      tag = ""
      key = plan_cache_key()
      metadata = artifact_metadata_record()
      plan_id = 0_i64
      hit_count = 0_i64
      candidate_key_text = ""
      materialized_flag = 0_i32

      read(line, *, iostat=ios) tag, key%key_text, plan_id, hit_count, candidate_key_text, &
        key%versions%schema_version, key%versions%abi_version, key%versions%planner_version, &
        key%versions%pack_version, key%versions%backend_version, key%logical_model_hash, &
        key%projector_revision, key%model_family, key%stage_kind, key%backend_family, &
        key%execution_route, key%dtype, key%rank, (key%shape(shape_index), shape_index = 1, MAX_TENSOR_RANK), &
        key%device_key, key%pack_format, metadata%backend_family, metadata%execution_route, &
        metadata%stage_kind, materialized_flag, metadata%payload_bytes, metadata%workspace_bytes, &
        metadata%artifact_format, metadata%payload_fingerprint, metadata%payload_path
      if (ios /= 0_i32) cycle
      if (trim(tag) /= "entry") cycle

      metadata%is_materialized = (materialized_flag /= 0_i32)
      call normalize_legacy_persisted_field(line, 5_i32, candidate_key_text)
      call normalize_legacy_persisted_field(line, 35_i32, metadata%artifact_format)
      call normalize_legacy_persisted_field(line, 36_i32, metadata%payload_fingerprint)
      call normalize_legacy_persisted_field(line, 37_i32, metadata%payload_path)
      call remember_loaded_plan_record(cache, key, plan_id, max(0_i64, hit_count), &
        candidate_key_text, metadata, loaded_count)
    end do

    close(unit_id)
    loaded_ok = .true.
  end subroutine read_runtime_plan_cache_file

  subroutine remember_loaded_plan_record(cache, key, plan_id, hit_count, candidate_key_text, &
                                         metadata, loaded_count)
    type(runtime_plan_cache), intent(inout)      :: cache
    type(plan_cache_key), intent(in)             :: key
    integer(i64), intent(in)                     :: plan_id
    integer(i64), intent(in)                     :: hit_count
    character(len=*), intent(in)                 :: candidate_key_text
    type(artifact_metadata_record), intent(in)   :: metadata
    integer(i64), intent(inout)                  :: loaded_count
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: status_code

    call record_plan_cache_entry(cache, key, plan_id, metadata, status_code, candidate_key_text)
    if (status_code /= MIZU_STATUS_OK) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = max(cache%entries(entry_index)%hit_count, max(0_i64, hit_count))
    loaded_count = loaded_count + 1_i64
  end subroutine remember_loaded_plan_record

  subroutine write_plan_cache_record(unit_id, record, ios)
    integer(i32), intent(in)          :: unit_id
    type(plan_cache_record), intent(in) :: record
    integer(i32), intent(inout)       :: ios
    integer(i32)                      :: materialized_flag
    character(len=MAX_CACHE_KEY_LEN)  :: key_text
    character(len=MAX_CACHE_KEY_LEN)  :: candidate_key_text
    character(len=MAX_NAME_LEN)       :: device_key
    character(len=MAX_NAME_LEN)       :: pack_format
    character(len=MAX_NAME_LEN)       :: artifact_format
    character(len=MAX_NAME_LEN)       :: payload_fingerprint
    character(len=MAX_PATH_LEN)       :: payload_path
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2) :: quoted_key_text
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2) :: quoted_candidate_key_text
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_device_key
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_pack_format
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_artifact_format
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_payload_fingerprint
    character(len=(2 * MAX_PATH_LEN) + 2)   :: quoted_payload_path

    if (.not. plan_cache_key_is_strict(record%key)) return
    if (record%plan_id == 0_i64) return
    if (.not. metadata_matches_key(record%artifact_metadata, record%key)) return

    key_text = record%key%key_text
    candidate_key_text = record%candidate_key_text
    device_key = record%key%device_key
    pack_format = record%key%pack_format
    materialized_flag = merge(1_i32, 0_i32, record%artifact_metadata%is_materialized)
    artifact_format = record%artifact_metadata%artifact_format
    payload_fingerprint = record%artifact_metadata%payload_fingerprint
    payload_path = record%artifact_metadata%payload_path
    quoted_key_text = quote_persisted_text(key_text, MAX_CACHE_KEY_LEN)
    quoted_candidate_key_text = quote_persisted_text(candidate_key_text, MAX_CACHE_KEY_LEN)
    quoted_device_key = quote_persisted_text(device_key, MAX_NAME_LEN)
    quoted_pack_format = quote_persisted_text(pack_format, MAX_NAME_LEN)
    quoted_artifact_format = quote_persisted_text(artifact_format, MAX_NAME_LEN)
    quoted_payload_fingerprint = quote_persisted_text(payload_fingerprint, MAX_NAME_LEN)
    quoted_payload_path = quote_persisted_text(payload_path, MAX_PATH_LEN)

    write(unit_id, "(A,1X,A,1X,I0,1X,I0,1X,A,5(1X,I0),2(1X,I0),6(1X,I0),8(1X,I0),2(1X,A),4(1X,I0),2(1X,I0),3(1X,A))", &
        iostat=ios) &
      "entry", trim(quoted_key_text), record%plan_id, max(0_i64, record%hit_count), &
      trim(quoted_candidate_key_text), record%key%versions%schema_version, record%key%versions%abi_version, &
      record%key%versions%planner_version, record%key%versions%pack_version, &
      record%key%versions%backend_version, record%key%logical_model_hash, record%key%projector_revision, &
      record%key%model_family, record%key%stage_kind, record%key%backend_family, record%key%execution_route, &
      record%key%dtype, record%key%rank, record%key%shape, trim(quoted_device_key), &
      trim(quoted_pack_format), record%artifact_metadata%backend_family, &
      record%artifact_metadata%execution_route, record%artifact_metadata%stage_kind, materialized_flag, &
      max(0_i64, record%artifact_metadata%payload_bytes), max(0_i64, record%artifact_metadata%workspace_bytes), &
      trim(quoted_artifact_format), trim(quoted_payload_fingerprint), trim(quoted_payload_path)
  end subroutine write_plan_cache_record

  integer(i32) function ensure_entry_index(cache, key_text) result(entry_index)
    type(runtime_plan_cache), intent(inout) :: cache
    character(len=*), intent(in)            :: key_text

    entry_index = find_entry_index(cache, key_text)
    if (entry_index > 0_i32) return

    call ensure_entry_capacity(cache, cache%entry_count + 1_i32)
    cache%entry_count = cache%entry_count + 1_i32
    entry_index = cache%entry_count
    cache%entries(entry_index) = plan_cache_record()
    cache%entries(entry_index)%key%key_text = trim(key_text)
  end function ensure_entry_index

  integer(i32) function find_entry_index(cache, key_text) result(entry_index)
    type(runtime_plan_cache), intent(in) :: cache
    character(len=*), intent(in)         :: key_text
    integer(i32)                         :: index

    entry_index = 0_i32
    if (len_trim(key_text) == 0) return
    if (.not. allocated(cache%entries)) return

    do index = 1_i32, cache%entry_count
      if (trim(cache%entries(index)%key%key_text) == trim(key_text)) then
        entry_index = index
        return
      end if
    end do
  end function find_entry_index

  subroutine ensure_entry_capacity(cache, required_capacity)
    type(runtime_plan_cache), intent(inout) :: cache
    integer(i32), intent(in)                :: required_capacity
    type(plan_cache_record), allocatable    :: resized_entries(:)
    integer(i32)                            :: new_capacity

    if (.not. allocated(cache%entries)) then
      allocate(cache%entries(max(INITIAL_PLAN_CACHE_CAPACITY, required_capacity)))
      cache%entries = plan_cache_record()
      return
    end if

    if (size(cache%entries) >= required_capacity) return

    new_capacity = max(required_capacity, int(size(cache%entries), kind=i32) * 2_i32)
    allocate(resized_entries(new_capacity))
    resized_entries = plan_cache_record()
    if (cache%entry_count > 0_i32) resized_entries(1:cache%entry_count) = &
      cache%entries(1:cache%entry_count)
    call move_alloc(resized_entries, cache%entries)
  end subroutine ensure_entry_capacity

  pure function i32_to_text(value) result(text)
    integer(i32), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i32_to_text

  pure function i64_to_text(value) result(text)
    integer(i64), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i64_to_text

end module mod_plan_cache
