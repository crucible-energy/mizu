module mod_weight_cache
  use mod_kinds,       only: i32, i64, MAX_NAME_LEN, MAX_PATH_LEN
  use mod_status,      only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,       only: MIZU_BACKEND_FAMILY_NONE, MIZU_EXEC_ROUTE_NONE, &
                             MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_NONE
  use mod_cache_keys,  only: MAX_CACHE_KEY_LEN, weight_cache_key
  use mod_cache_store, only: artifact_metadata_record, quote_persisted_text

  implicit none

  private
  public :: weight_cache_record, runtime_weight_cache
  public :: initialize_runtime_weight_cache, reset_runtime_weight_cache
  public :: record_weight_cache_entry, lookup_weight_cache_entry
  public :: load_runtime_weight_cache, save_runtime_weight_cache, warm_runtime_weight_cache
  public :: weight_cache_key_is_strict

  integer(i32), parameter :: INITIAL_WEIGHT_CACHE_CAPACITY = 16_i32
  integer(i32), parameter :: MAX_WEIGHT_CACHE_RECORD_LINE_LEN = &
    (4_i32 * MAX_CACHE_KEY_LEN) + (8_i32 * MAX_NAME_LEN) + (2_i32 * MAX_PATH_LEN) + 512_i32

  type :: weight_cache_record
    type(weight_cache_key)           :: key
    character(len=MAX_CACHE_KEY_LEN) :: pack_identity_text = ""
    type(artifact_metadata_record)   :: artifact_metadata
    integer(i64)                     :: hit_count = 0_i64
  end type weight_cache_record

  type :: runtime_weight_cache
    integer(i32) :: entry_count = 0_i32
    type(weight_cache_record), allocatable :: entries(:)
  end type runtime_weight_cache

contains

  subroutine initialize_runtime_weight_cache(cache)
    type(runtime_weight_cache), intent(out) :: cache

    cache = runtime_weight_cache()
  end subroutine initialize_runtime_weight_cache

  subroutine reset_runtime_weight_cache(cache)
    type(runtime_weight_cache), intent(inout) :: cache

    if (allocated(cache%entries)) deallocate(cache%entries)
    cache%entry_count = 0_i32
  end subroutine reset_runtime_weight_cache

  subroutine record_weight_cache_entry(cache, key, artifact_metadata, status_code, &
                                       pack_identity_text)
    type(runtime_weight_cache), intent(inout)  :: cache
    type(weight_cache_key), intent(in)         :: key
    type(artifact_metadata_record), intent(in) :: artifact_metadata
    integer(i32), intent(out)                  :: status_code
    character(len=*), intent(in), optional     :: pack_identity_text
    character(len=MAX_CACHE_KEY_LEN)           :: resolved_pack_identity
    integer(i32)                               :: entry_index
    integer(i64)                               :: existing_hits

    resolved_pack_identity = resolve_pack_identity(artifact_metadata, pack_identity_text)
    if (.not. weight_cache_key_is_strict(key) .or. len_trim(resolved_pack_identity) == 0 .or. &
        .not. metadata_matches_key(artifact_metadata, key)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    entry_index = ensure_entry_index(cache, trim(key%key_text))
    existing_hits = cache%entries(entry_index)%hit_count

    cache%entries(entry_index)%key = key
    cache%entries(entry_index)%pack_identity_text = trim(resolved_pack_identity)
    cache%entries(entry_index)%artifact_metadata = artifact_metadata
    cache%entries(entry_index)%hit_count = existing_hits

    status_code = MIZU_STATUS_OK
  end subroutine record_weight_cache_entry

  subroutine lookup_weight_cache_entry(cache, key, record, found)
    type(runtime_weight_cache), intent(inout) :: cache
    type(weight_cache_key), intent(in)        :: key
    type(weight_cache_record), intent(out)    :: record
    logical, intent(out)                      :: found
    integer(i32)                              :: entry_index

    record = weight_cache_record()
    found = .false.
    if (.not. weight_cache_key_is_strict(key)) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = cache%entries(entry_index)%hit_count + 1_i64
    record = cache%entries(entry_index)
    found = .true.
  end subroutine lookup_weight_cache_entry

  subroutine load_runtime_weight_cache(cache, file_path, loaded_ok)
    type(runtime_weight_cache), intent(inout) :: cache
    character(len=*), intent(in)              :: file_path
    logical, intent(out)                      :: loaded_ok
    integer(i64)                              :: loaded_count

    call read_runtime_weight_cache_file(cache, file_path, .true., loaded_count, loaded_ok)
  end subroutine load_runtime_weight_cache

  subroutine warm_runtime_weight_cache(cache, file_path, warmed_count, loaded_ok)
    type(runtime_weight_cache), intent(inout) :: cache
    character(len=*), intent(in)              :: file_path
    integer(i64), intent(out)                 :: warmed_count
    logical, intent(out)                      :: loaded_ok

    call read_runtime_weight_cache_file(cache, file_path, .false., warmed_count, loaded_ok)
  end subroutine warm_runtime_weight_cache

  subroutine save_runtime_weight_cache(cache, file_path, saved_ok)
    type(runtime_weight_cache), intent(in) :: cache
    character(len=*), intent(in)           :: file_path
    logical, intent(out)                   :: saved_ok
    integer(i32)                           :: unit_id
    integer(i32)                           :: ios
    integer(i32)                           :: index

    saved_ok = .false.
    if (len_trim(file_path) == 0) return

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return

    do index = 1_i32, cache%entry_count
      call write_weight_cache_record(unit_id, cache%entries(index), ios)
      if (ios /= 0_i32) then
        close(unit_id)
        return
      end if
    end do

    close(unit_id)
    saved_ok = .true.
  end subroutine save_runtime_weight_cache

  pure logical function weight_cache_key_is_strict(key) result(is_strict)
    type(weight_cache_key), intent(in) :: key

    is_strict = len_trim(key%key_text) > 0 .and. &
      key%versions%schema_version > 0_i32 .and. &
      key%versions%abi_version > 0_i32 .and. &
      key%logical_model_hash /= 0_i64 .and. &
      key%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
      key%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
      len_trim(key%device_key) > 0 .and. &
      len_trim(key%pack_format) > 0 .and. &
      index(key%key_text, "weight:v") == 1 .and. &
      index(key%key_text, ":abi=" // trim(i32_to_text(key%versions%abi_version))) > 0 .and. &
      index(key%key_text, ":planner=" // trim(i32_to_text(key%versions%planner_version))) > 0 .and. &
      index(key%key_text, ":packv=" // trim(i32_to_text(key%versions%pack_version))) > 0 .and. &
      index(key%key_text, ":model=" // trim(i64_to_text(key%logical_model_hash))) > 0 .and. &
      index(key%key_text, ":backend=" // trim(i32_to_text(key%backend_family))) > 0 .and. &
      index(key%key_text, ":route=" // trim(i32_to_text(key%execution_route))) > 0 .and. &
      index(key%key_text, ":device=" // trim(key%device_key)) > 0 .and. &
      index(key%key_text, ":pack=" // trim(key%pack_format)) > 0
  end function weight_cache_key_is_strict

  pure logical function metadata_matches_key(metadata, key) result(matches)
    type(artifact_metadata_record), intent(in) :: metadata
    type(weight_cache_key), intent(in)         :: key

    matches = .true.
    if (metadata%stage_kind /= MIZU_STAGE_NONE .and. metadata%stage_kind /= MIZU_STAGE_MODEL_LOAD) &
      matches = .false.
    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        metadata%backend_family /= key%backend_family) matches = .false.
    if (metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        metadata%execution_route /= key%execution_route) matches = .false.
    if (len_trim(metadata%artifact_format) > 0 .and. trim(metadata%artifact_format) /= trim(key%pack_format)) &
      matches = .false.
  end function metadata_matches_key

  function resolve_pack_identity(metadata, provided_pack_identity) result(pack_identity)
    type(artifact_metadata_record), intent(in) :: metadata
    character(len=*), intent(in), optional     :: provided_pack_identity
    character(len=MAX_CACHE_KEY_LEN)           :: pack_identity

    pack_identity = ""
    if (present(provided_pack_identity)) then
      if (len_trim(provided_pack_identity) > 0) then
        pack_identity = trim(provided_pack_identity)
        return
      end if
    end if

    if (len_trim(metadata%payload_fingerprint) > 0) then
      pack_identity = trim(metadata%payload_fingerprint)
    else if (len_trim(metadata%payload_path) > 0) then
      pack_identity = trim(metadata%payload_path)
    else if (len_trim(metadata%artifact_format) > 0) then
      pack_identity = trim(metadata%artifact_format)
    end if
  end function resolve_pack_identity

  subroutine read_runtime_weight_cache_file(cache, file_path, replace_existing, loaded_count, loaded_ok)
    type(runtime_weight_cache), intent(inout) :: cache
    character(len=*), intent(in)              :: file_path
    logical, intent(in)                       :: replace_existing
    integer(i64), intent(out)                 :: loaded_count
    logical, intent(out)                      :: loaded_ok
    character(len=MAX_WEIGHT_CACHE_RECORD_LINE_LEN) :: line
    character(len=16)                         :: tag
    type(weight_cache_key)                    :: key
    type(artifact_metadata_record)            :: metadata
    character(len=MAX_CACHE_KEY_LEN)          :: pack_identity_text
    integer(i64)                              :: hit_count
    integer(i32)                              :: materialized_flag
    integer(i32)                              :: unit_id
    integer(i32)                              :: ios
    logical                                   :: exists

    loaded_count = 0_i64
    loaded_ok = .false.
    if (len_trim(file_path) == 0) return

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) then
      if (replace_existing) call reset_runtime_weight_cache(cache)
      loaded_ok = .true.
      return
    end if

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return

    if (replace_existing) call reset_runtime_weight_cache(cache)
    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit
      if (len_trim(line) == 0) cycle

      tag = ""
      key = weight_cache_key()
      metadata = artifact_metadata_record()
      pack_identity_text = ""
      hit_count = 0_i64
      materialized_flag = 0_i32

      read(line, *, iostat=ios) tag, key%key_text, hit_count, pack_identity_text, &
        key%versions%schema_version, key%versions%abi_version, key%versions%planner_version, &
        key%versions%pack_version, key%versions%backend_version, key%logical_model_hash, &
        key%projector_revision, key%model_family, key%backend_family, key%execution_route, &
        key%device_key, key%pack_format, metadata%backend_family, metadata%execution_route, &
        metadata%stage_kind, materialized_flag, metadata%payload_bytes, metadata%workspace_bytes, &
        metadata%artifact_format, metadata%payload_fingerprint, metadata%payload_path
      if (ios /= 0_i32) cycle
      if (trim(tag) /= "entry") cycle

      metadata%is_materialized = (materialized_flag /= 0_i32)
      call remember_loaded_weight_record(cache, key, max(0_i64, hit_count), pack_identity_text, &
        metadata, loaded_count)
    end do

    close(unit_id)
    loaded_ok = .true.
  end subroutine read_runtime_weight_cache_file

  subroutine remember_loaded_weight_record(cache, key, hit_count, pack_identity_text, metadata, loaded_count)
    type(runtime_weight_cache), intent(inout)    :: cache
    type(weight_cache_key), intent(in)           :: key
    integer(i64), intent(in)                     :: hit_count
    character(len=*), intent(in)                 :: pack_identity_text
    type(artifact_metadata_record), intent(in)   :: metadata
    integer(i64), intent(inout)                  :: loaded_count
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: status_code

    call record_weight_cache_entry(cache, key, metadata, status_code, pack_identity_text)
    if (status_code /= MIZU_STATUS_OK) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = max(cache%entries(entry_index)%hit_count, max(0_i64, hit_count))
    loaded_count = loaded_count + 1_i64
  end subroutine remember_loaded_weight_record

  subroutine write_weight_cache_record(unit_id, record, ios)
    integer(i32), intent(in)            :: unit_id
    type(weight_cache_record), intent(in) :: record
    integer(i32), intent(inout)         :: ios
    integer(i32)                        :: materialized_flag
    character(len=MAX_CACHE_KEY_LEN)    :: key_text
    character(len=MAX_CACHE_KEY_LEN)    :: pack_identity_text
    character(len=MAX_NAME_LEN)         :: device_key
    character(len=MAX_NAME_LEN)         :: pack_format
    character(len=MAX_NAME_LEN)         :: artifact_format
    character(len=MAX_NAME_LEN)         :: payload_fingerprint
    character(len=MAX_PATH_LEN)         :: payload_path
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2) :: quoted_key_text
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2) :: quoted_pack_identity_text
    character(len=(2 * MAX_NAME_LEN) + 2)     :: quoted_device_key
    character(len=(2 * MAX_NAME_LEN) + 2)     :: quoted_pack_format
    character(len=(2 * MAX_NAME_LEN) + 2)     :: quoted_artifact_format
    character(len=(2 * MAX_NAME_LEN) + 2)     :: quoted_payload_fingerprint
    character(len=(2 * MAX_PATH_LEN) + 2)     :: quoted_payload_path

    if (.not. weight_cache_key_is_strict(record%key)) return
    if (len_trim(record%pack_identity_text) == 0) return
    if (.not. metadata_matches_key(record%artifact_metadata, record%key)) return

    key_text = record%key%key_text
    pack_identity_text = record%pack_identity_text
    device_key = record%key%device_key
    pack_format = record%key%pack_format
    materialized_flag = merge(1_i32, 0_i32, record%artifact_metadata%is_materialized)
    artifact_format = record%artifact_metadata%artifact_format
    payload_fingerprint = record%artifact_metadata%payload_fingerprint
    payload_path = record%artifact_metadata%payload_path
    quoted_key_text = quote_persisted_text(key_text, MAX_CACHE_KEY_LEN)
    quoted_pack_identity_text = quote_persisted_text(pack_identity_text, MAX_CACHE_KEY_LEN)
    quoted_device_key = quote_persisted_text(device_key, MAX_NAME_LEN)
    quoted_pack_format = quote_persisted_text(pack_format, MAX_NAME_LEN)
    quoted_artifact_format = quote_persisted_text(artifact_format, MAX_NAME_LEN)
    quoted_payload_fingerprint = quote_persisted_text(payload_fingerprint, MAX_NAME_LEN)
    quoted_payload_path = quote_persisted_text(payload_path, MAX_PATH_LEN)

    write(unit_id, "(A,1X,A,1X,I0,1X,A,5(1X,I0),2(1X,I0),3(1X,I0),2(1X,A),4(1X,I0),2(1X,I0),3(1X,A))", &
        iostat=ios) &
      "entry", trim(quoted_key_text), max(0_i64, record%hit_count), trim(quoted_pack_identity_text), &
      record%key%versions%schema_version, record%key%versions%abi_version, &
      record%key%versions%planner_version, record%key%versions%pack_version, &
      record%key%versions%backend_version, record%key%logical_model_hash, record%key%projector_revision, &
      record%key%model_family, record%key%backend_family, record%key%execution_route, &
      trim(quoted_device_key), trim(quoted_pack_format), record%artifact_metadata%backend_family, &
      record%artifact_metadata%execution_route, record%artifact_metadata%stage_kind, materialized_flag, &
      max(0_i64, record%artifact_metadata%payload_bytes), max(0_i64, record%artifact_metadata%workspace_bytes), &
      trim(quoted_artifact_format), trim(quoted_payload_fingerprint), trim(quoted_payload_path)
  end subroutine write_weight_cache_record

  integer(i32) function ensure_entry_index(cache, key_text) result(entry_index)
    type(runtime_weight_cache), intent(inout) :: cache
    character(len=*), intent(in)              :: key_text

    entry_index = find_entry_index(cache, key_text)
    if (entry_index > 0_i32) return

    call ensure_entry_capacity(cache, cache%entry_count + 1_i32)
    cache%entry_count = cache%entry_count + 1_i32
    entry_index = cache%entry_count
    cache%entries(entry_index) = weight_cache_record()
    cache%entries(entry_index)%key%key_text = trim(key_text)
  end function ensure_entry_index

  integer(i32) function find_entry_index(cache, key_text) result(entry_index)
    type(runtime_weight_cache), intent(in) :: cache
    character(len=*), intent(in)           :: key_text
    integer(i32)                           :: index

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
    type(runtime_weight_cache), intent(inout) :: cache
    integer(i32), intent(in)                  :: required_capacity
    type(weight_cache_record), allocatable    :: resized_entries(:)
    integer(i32)                              :: new_capacity

    if (.not. allocated(cache%entries)) then
      allocate(cache%entries(max(INITIAL_WEIGHT_CACHE_CAPACITY, required_capacity)))
      cache%entries = weight_cache_record()
      return
    end if

    if (size(cache%entries) >= required_capacity) return

    new_capacity = max(required_capacity, int(size(cache%entries), kind=i32) * 2_i32)
    allocate(resized_entries(new_capacity))
    resized_entries = weight_cache_record()
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

end module mod_weight_cache
