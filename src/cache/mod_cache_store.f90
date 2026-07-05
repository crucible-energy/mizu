module mod_cache_store
  use mod_kinds,      only: i32, i64, MAX_NAME_LEN, MAX_PATH_LEN
  use mod_cache_keys, only: MAX_CACHE_KEY_LEN
  use mod_types,      only: MIZU_BACKEND_FAMILY_NONE, MIZU_EXEC_ROUTE_NONE, &
                            MIZU_STAGE_NONE

  implicit none

  private
  public :: artifact_metadata_record, artifact_metadata_is_defined
  public :: cache_key_store, runtime_cache_bundle
  public :: initialize_runtime_cache_bundle, reset_runtime_cache_bundle
  public :: touch_weight_cache_key, touch_plan_cache_key
  public :: touch_session_cache_key, touch_multimodal_cache_key
  public :: record_weight_artifact_metadata, record_plan_artifact_metadata
  public :: record_session_artifact_metadata, record_multimodal_artifact_metadata
  public :: lookup_weight_artifact_metadata, lookup_plan_artifact_metadata
  public :: lookup_session_artifact_metadata, lookup_multimodal_artifact_metadata
  public :: load_runtime_cache_bundle, save_runtime_cache_bundle
  public :: quote_persisted_text, normalize_legacy_persisted_field

  integer(i32), parameter :: INITIAL_CACHE_CAPACITY = 16_i32
  integer(i32), parameter :: MAX_RECORD_LINE_LEN = (4_i32 * MAX_PATH_LEN) + &
    (4_i32 * MAX_CACHE_KEY_LEN) + (4_i32 * MAX_NAME_LEN) + 256_i32

  type :: artifact_metadata_record
    integer(i32)                :: backend_family   = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                :: execution_route  = MIZU_EXEC_ROUTE_NONE
    integer(i32)                :: stage_kind       = MIZU_STAGE_NONE
    logical                     :: is_materialized  = .false.
    integer(i64)                :: payload_bytes    = 0_i64
    integer(i64)                :: workspace_bytes  = 0_i64
    character(len=MAX_NAME_LEN) :: artifact_format  = ""
    character(len=MAX_NAME_LEN) :: payload_fingerprint = ""
    character(len=MAX_PATH_LEN) :: payload_path     = ""
  end type artifact_metadata_record

  type :: cache_key_store
    integer(i32)                             :: entry_count = 0_i32
    character(len=MAX_CACHE_KEY_LEN), allocatable :: entries(:)
    type(artifact_metadata_record), allocatable   :: metadata(:)
  end type cache_key_store

  type :: runtime_cache_bundle
    type(cache_key_store) :: weight_store
    type(cache_key_store) :: plan_store
    type(cache_key_store) :: session_store
    type(cache_key_store) :: multimodal_store
  end type runtime_cache_bundle

contains

  subroutine initialize_runtime_cache_bundle(bundle)
    type(runtime_cache_bundle), intent(out) :: bundle

    bundle = runtime_cache_bundle()
  end subroutine initialize_runtime_cache_bundle

  subroutine reset_runtime_cache_bundle(bundle)
    type(runtime_cache_bundle), intent(inout) :: bundle

    call reset_cache_key_store(bundle%weight_store)
    call reset_cache_key_store(bundle%plan_store)
    call reset_cache_key_store(bundle%session_store)
    call reset_cache_key_store(bundle%multimodal_store)
  end subroutine reset_runtime_cache_bundle

  subroutine touch_weight_cache_key(bundle, key_text, was_hit)
    type(runtime_cache_bundle), intent(inout) :: bundle
    character(len=*), intent(in)              :: key_text
    logical, intent(out)                      :: was_hit

    call touch_cache_key_store(bundle%weight_store, key_text, was_hit)
  end subroutine touch_weight_cache_key

  subroutine touch_plan_cache_key(bundle, key_text, was_hit)
    type(runtime_cache_bundle), intent(inout) :: bundle
    character(len=*), intent(in)              :: key_text
    logical, intent(out)                      :: was_hit

    call touch_cache_key_store(bundle%plan_store, key_text, was_hit)
  end subroutine touch_plan_cache_key

  subroutine touch_session_cache_key(bundle, key_text, was_hit)
    type(runtime_cache_bundle), intent(inout) :: bundle
    character(len=*), intent(in)              :: key_text
    logical, intent(out)                      :: was_hit

    call touch_cache_key_store(bundle%session_store, key_text, was_hit)
  end subroutine touch_session_cache_key

  subroutine touch_multimodal_cache_key(bundle, key_text, was_hit)
    type(runtime_cache_bundle), intent(inout) :: bundle
    character(len=*), intent(in)              :: key_text
    logical, intent(out)                      :: was_hit

    call touch_cache_key_store(bundle%multimodal_store, key_text, was_hit)
  end subroutine touch_multimodal_cache_key

  subroutine record_weight_artifact_metadata(bundle, key_text, metadata)
    type(runtime_cache_bundle), intent(inout)    :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(in)   :: metadata

    call record_cache_key_metadata(bundle%weight_store, key_text, metadata)
  end subroutine record_weight_artifact_metadata

  subroutine record_plan_artifact_metadata(bundle, key_text, metadata)
    type(runtime_cache_bundle), intent(inout)    :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(in)   :: metadata

    call record_cache_key_metadata(bundle%plan_store, key_text, metadata)
  end subroutine record_plan_artifact_metadata

  subroutine record_session_artifact_metadata(bundle, key_text, metadata)
    type(runtime_cache_bundle), intent(inout)    :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(in)   :: metadata

    call record_cache_key_metadata(bundle%session_store, key_text, metadata)
  end subroutine record_session_artifact_metadata

  subroutine record_multimodal_artifact_metadata(bundle, key_text, metadata)
    type(runtime_cache_bundle), intent(inout)    :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(in)   :: metadata

    call record_cache_key_metadata(bundle%multimodal_store, key_text, metadata)
  end subroutine record_multimodal_artifact_metadata

  subroutine lookup_weight_artifact_metadata(bundle, key_text, metadata, found)
    type(runtime_cache_bundle), intent(in)       :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(out)  :: metadata
    logical, intent(out)                         :: found

    call lookup_cache_key_metadata(bundle%weight_store, key_text, metadata, found)
  end subroutine lookup_weight_artifact_metadata

  subroutine lookup_plan_artifact_metadata(bundle, key_text, metadata, found)
    type(runtime_cache_bundle), intent(in)       :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(out)  :: metadata
    logical, intent(out)                         :: found

    call lookup_cache_key_metadata(bundle%plan_store, key_text, metadata, found)
  end subroutine lookup_plan_artifact_metadata

  subroutine lookup_session_artifact_metadata(bundle, key_text, metadata, found)
    type(runtime_cache_bundle), intent(in)       :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(out)  :: metadata
    logical, intent(out)                         :: found

    call lookup_cache_key_metadata(bundle%session_store, key_text, metadata, found)
  end subroutine lookup_session_artifact_metadata

  subroutine lookup_multimodal_artifact_metadata(bundle, key_text, metadata, found)
    type(runtime_cache_bundle), intent(in)       :: bundle
    character(len=*), intent(in)                 :: key_text
    type(artifact_metadata_record), intent(out)  :: metadata
    logical, intent(out)                         :: found

    call lookup_cache_key_metadata(bundle%multimodal_store, key_text, metadata, found)
  end subroutine lookup_multimodal_artifact_metadata

  pure logical function artifact_metadata_is_defined(metadata) result(is_defined)
    type(artifact_metadata_record), intent(in) :: metadata

    is_defined = metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .or. &
      metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .or. &
      metadata%stage_kind /= MIZU_STAGE_NONE .or. &
      metadata%is_materialized .or. &
      metadata%payload_bytes > 0_i64 .or. &
      metadata%workspace_bytes > 0_i64 .or. &
      len_trim(metadata%artifact_format) > 0 .or. &
      len_trim(metadata%payload_fingerprint) > 0 .or. &
      len_trim(metadata%payload_path) > 0
  end function artifact_metadata_is_defined

  subroutine load_runtime_cache_bundle(bundle, file_path, loaded_ok)
    type(runtime_cache_bundle), intent(inout) :: bundle
    character(len=*), intent(in)              :: file_path
    logical, intent(out)                      :: loaded_ok
    character(len=MAX_RECORD_LINE_LEN)        :: line
    character(len=16)                         :: tag
    character(len=16)                         :: kind_tag
    character(len=MAX_CACHE_KEY_LEN)          :: key_text
    type(artifact_metadata_record)            :: metadata
    integer(i32)                              :: backend_family
    integer(i32)                              :: execution_route
    integer(i32)                              :: stage_kind
    integer(i32)                              :: materialized_flag
    integer(i32)                              :: unit_id
    integer(i32)                              :: ios
    logical                                   :: exists

    loaded_ok = .false.
    if (len_trim(file_path) == 0) return
    call reset_runtime_cache_bundle(bundle)

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) then
      loaded_ok = .true.
      return
    end if

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0) return

    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle

      tag = ""
      read(line, *, iostat=ios) tag
      if (ios /= 0) cycle

      select case (trim(tag))
      case ("weight", "plan", "session", "mm")
        cycle
      case ("meta")
        kind_tag = ""
        key_text = ""
        metadata = artifact_metadata_record()
        backend_family = MIZU_BACKEND_FAMILY_NONE
        execution_route = MIZU_EXEC_ROUTE_NONE
        stage_kind = MIZU_STAGE_NONE
        materialized_flag = 0_i32
        read(line, *, iostat=ios) tag, kind_tag, key_text, backend_family, execution_route, &
          stage_kind, materialized_flag, metadata%payload_bytes, metadata%workspace_bytes, metadata%artifact_format, &
          metadata%payload_fingerprint, metadata%payload_path
        if (ios /= 0) cycle
        if (len_trim(key_text) == 0) cycle

        metadata%backend_family = backend_family
        metadata%execution_route = execution_route
        metadata%stage_kind = stage_kind
        metadata%is_materialized = (materialized_flag /= 0_i32)
        call normalize_legacy_persisted_field(line, 10_i32, metadata%artifact_format)
        call normalize_legacy_persisted_field(line, 11_i32, metadata%payload_fingerprint)
        call normalize_legacy_persisted_field(line, 12_i32, metadata%payload_path)
        select case (trim(kind_tag))
        case ("weight")
          call record_cache_key_metadata(bundle%weight_store, trim(key_text), metadata)
        case ("plan")
          call record_cache_key_metadata(bundle%plan_store, trim(key_text), metadata)
        case ("session")
          call record_cache_key_metadata(bundle%session_store, trim(key_text), metadata)
        case ("mm")
          call record_cache_key_metadata(bundle%multimodal_store, trim(key_text), metadata)
        case default
          cycle
        end select
      case default
        cycle
      end select
    end do

    close(unit_id)
    loaded_ok = .true.
  end subroutine load_runtime_cache_bundle

  subroutine save_runtime_cache_bundle(bundle, file_path, saved_ok)
    type(runtime_cache_bundle), intent(in) :: bundle
    character(len=*), intent(in)           :: file_path
    logical, intent(out)                   :: saved_ok
    integer(i32)                           :: unit_id
    integer(i32)                           :: ios

    saved_ok = .false.
    if (len_trim(file_path) == 0) return

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write", iostat=ios)
    if (ios /= 0) return

    call write_artifact_metadata_store(unit_id, "weight", bundle%weight_store, ios)
    if (ios == 0_i32) call write_artifact_metadata_store(unit_id, "plan", bundle%plan_store, ios)
    if (ios == 0_i32) call write_artifact_metadata_store(unit_id, "session", bundle%session_store, ios)
    if (ios == 0_i32) call write_artifact_metadata_store(unit_id, "mm", bundle%multimodal_store, ios)
    if (ios /= 0_i32) then
      close(unit_id)
      return
    end if

    close(unit_id)
    saved_ok = .true.
  end subroutine save_runtime_cache_bundle

  subroutine reset_cache_key_store(store)
    type(cache_key_store), intent(inout) :: store

    store%entry_count = 0_i32
    if (allocated(store%entries)) deallocate(store%entries)
    if (allocated(store%metadata)) deallocate(store%metadata)
  end subroutine reset_cache_key_store

  subroutine touch_cache_key_store(store, key_text, was_hit)
    type(cache_key_store), intent(inout) :: store
    character(len=*), intent(in)         :: key_text
    logical, intent(out)                 :: was_hit
    integer(i32)                         :: index
    character(len=MAX_CACHE_KEY_LEN)     :: normalized_key

    normalized_key = ""
    if (len_trim(key_text) > 0) normalized_key = trim(key_text)

    was_hit = .false.
    if (len_trim(normalized_key) == 0) return

    call ensure_cache_key_store_capacity(store, max(INITIAL_CACHE_CAPACITY, store%entry_count + 1_i32))

    do index = 1_i32, store%entry_count
      if (trim(store%entries(index)) == trim(normalized_key)) then
        was_hit = .true.
        return
      end if
    end do

    store%entry_count = store%entry_count + 1_i32
    store%entries(store%entry_count) = normalized_key
    store%metadata(store%entry_count) = artifact_metadata_record()
  end subroutine touch_cache_key_store

  subroutine record_cache_key_metadata(store, key_text, metadata)
    type(cache_key_store), intent(inout)        :: store
    character(len=*), intent(in)                :: key_text
    type(artifact_metadata_record), intent(in)  :: metadata
    integer(i32)                                :: index
    logical                                     :: was_hit
    character(len=MAX_CACHE_KEY_LEN)            :: normalized_key

    normalized_key = ""
    if (len_trim(key_text) > 0) normalized_key = trim(key_text)
    if (len_trim(normalized_key) == 0) return

    call touch_cache_key_store(store, trim(normalized_key), was_hit)
    do index = 1_i32, store%entry_count
      if (trim(store%entries(index)) == trim(normalized_key)) then
        store%metadata(index) = metadata
        return
      end if
    end do
  end subroutine record_cache_key_metadata

  subroutine lookup_cache_key_metadata(store, key_text, metadata, found)
    type(cache_key_store), intent(in)          :: store
    character(len=*), intent(in)               :: key_text
    type(artifact_metadata_record), intent(out) :: metadata
    logical, intent(out)                       :: found
    integer(i32)                               :: index
    character(len=MAX_CACHE_KEY_LEN)           :: normalized_key

    metadata = artifact_metadata_record()
    found = .false.
    normalized_key = ""
    if (len_trim(key_text) > 0) normalized_key = trim(key_text)
    if (len_trim(normalized_key) == 0) return

    do index = 1_i32, store%entry_count
      if (trim(store%entries(index)) == trim(normalized_key)) then
        metadata = store%metadata(index)
        found = artifact_metadata_is_defined(metadata)
        return
      end if
    end do
  end subroutine lookup_cache_key_metadata

  subroutine write_artifact_metadata_store(unit_id, tag, store, ios)
    integer(i32), intent(in)          :: unit_id
    character(len=*), intent(in)      :: tag
    type(cache_key_store), intent(in) :: store
    integer(i32), intent(inout)       :: ios
    integer(i32)                      :: index
    integer(i32)                      :: materialized_flag
    character(len=MAX_CACHE_KEY_LEN)  :: key_text
    character(len=MAX_NAME_LEN)       :: artifact_format
    character(len=MAX_NAME_LEN)       :: payload_fingerprint
    character(len=MAX_PATH_LEN)       :: payload_path
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2) :: quoted_key_text
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_artifact_format
    character(len=(2 * MAX_NAME_LEN) + 2)   :: quoted_payload_fingerprint
    character(len=(2 * MAX_PATH_LEN) + 2)   :: quoted_payload_path

    do index = 1_i32, store%entry_count
      if (.not. artifact_metadata_is_defined(store%metadata(index))) cycle

      materialized_flag = merge(1_i32, 0_i32, store%metadata(index)%is_materialized)
      key_text = store%entries(index)
      artifact_format = store%metadata(index)%artifact_format
      payload_fingerprint = store%metadata(index)%payload_fingerprint
      payload_path = store%metadata(index)%payload_path
      quoted_key_text = quote_persisted_text(key_text, MAX_CACHE_KEY_LEN)
      quoted_artifact_format = quote_persisted_text(artifact_format, MAX_NAME_LEN)
      quoted_payload_fingerprint = quote_persisted_text(payload_fingerprint, MAX_NAME_LEN)
      quoted_payload_path = quote_persisted_text(payload_path, MAX_PATH_LEN)

      write(unit_id, "(A,1X,A,1X,A,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,A,1X,A,1X,A)", iostat=ios) &
        "meta", trim(tag), trim(quoted_key_text), store%metadata(index)%backend_family, &
        store%metadata(index)%execution_route, store%metadata(index)%stage_kind, &
        materialized_flag, max(0_i64, store%metadata(index)%payload_bytes), &
        max(0_i64, store%metadata(index)%workspace_bytes), trim(quoted_artifact_format), &
        trim(quoted_payload_fingerprint), trim(quoted_payload_path)
      if (ios /= 0_i32) return
    end do
  end subroutine write_artifact_metadata_store

  subroutine ensure_cache_key_store_capacity(store, required_capacity)
    type(cache_key_store), intent(inout) :: store
    integer(i32), intent(in)             :: required_capacity
    character(len=MAX_CACHE_KEY_LEN), allocatable :: resized_entries(:)
    type(artifact_metadata_record), allocatable   :: resized_metadata(:)
    integer(i32)                         :: current_capacity
    integer(i32)                         :: next_capacity

    if (.not. allocated(store%entries)) then
      allocate(store%entries(required_capacity))
      allocate(store%metadata(required_capacity))
      store%entries = ""
      store%metadata = artifact_metadata_record()
      return
    end if

    current_capacity = int(size(store%entries), kind=i32)
    if (required_capacity <= current_capacity) return

    next_capacity = max(required_capacity, current_capacity * 2_i32)
    allocate(resized_entries(next_capacity))
    allocate(resized_metadata(next_capacity))
    resized_entries = ""
    resized_metadata = artifact_metadata_record()
    if (store%entry_count > 0_i32) then
      resized_entries(1:store%entry_count) = store%entries(1:store%entry_count)
      resized_metadata(1:store%entry_count) = store%metadata(1:store%entry_count)
    end if
    call move_alloc(resized_entries, store%entries)
    call move_alloc(resized_metadata, store%metadata)
  end subroutine ensure_cache_key_store_capacity

  function quote_persisted_text(text, buffer_len) result(quoted_text)
    character(len=*), intent(in) :: text
    integer(i32), intent(in)     :: buffer_len
    character(len=(2 * buffer_len) + 2) :: quoted_text
    character(len=2 * buffer_len) :: escaped_text
    integer(i32)                  :: src_index
    integer(i32)                  :: dest_index

    quoted_text = '""'
    escaped_text = ""
    dest_index = 0_i32
    do src_index = 1_i32, len_trim(text)
      dest_index = dest_index + 1_i32
      escaped_text(dest_index:dest_index) = text(src_index:src_index)
      if (text(src_index:src_index) == '"') then
        dest_index = dest_index + 1_i32
        escaped_text(dest_index:dest_index) = '"'
      end if
    end do
    if (dest_index > 0_i32) then
      quoted_text = '"' // escaped_text(:dest_index) // '"'
    end if
  end function quote_persisted_text

  subroutine normalize_legacy_persisted_field(line, field_index, text)
    character(len=*), intent(in)    :: line
    integer(i32), intent(in)        :: field_index
    character(len=*), intent(inout) :: text

    if (legacy_persisted_field_is_empty(line, field_index)) text = ""
  end subroutine normalize_legacy_persisted_field

  pure logical function legacy_persisted_field_is_empty(line, field_index) result(is_empty)
    character(len=*), intent(in) :: line
    integer(i32), intent(in)     :: field_index
    integer(i32)                 :: line_len
    integer(i32)                 :: index
    integer(i32)                 :: current_field
    integer(i32)                 :: token_len
    logical                      :: quoted_token
    logical                      :: is_dash_token
    character(len=1)             :: ch

    is_empty = .false.
    if (field_index <= 0_i32) return

    line_len = len_trim(line)
    index = 1_i32
    current_field = 0_i32
    do while (index <= line_len)
      do while (index <= line_len)
        ch = line(index:index)
        if (ch /= " " .and. ch /= achar(9)) exit
        index = index + 1_i32
      end do
      if (index > line_len) exit

      current_field = current_field + 1_i32
      quoted_token = (line(index:index) == '"')
      token_len = 0_i32
      is_dash_token = .true.
      if (quoted_token) index = index + 1_i32

      do while (index <= line_len)
        ch = line(index:index)
        if (quoted_token) then
          if (ch == '"') then
            if (index < line_len .and. line(index + 1_i32:index + 1_i32) == '"') then
              token_len = token_len + 1_i32
              is_dash_token = .false.
              index = index + 2_i32
              cycle
            end if
            index = index + 1_i32
            exit
          end if
        else if (ch == " " .or. ch == achar(9)) then
          exit
        end if

        token_len = token_len + 1_i32
        if (token_len > 1_i32 .or. ch /= "-") is_dash_token = .false.
        index = index + 1_i32
      end do

      do while (index <= line_len)
        ch = line(index:index)
        if (ch == " " .or. ch == achar(9)) exit
        token_len = token_len + 1_i32
        is_dash_token = .false.
        index = index + 1_i32
      end do

      if (current_field == field_index) then
        is_empty = (.not. quoted_token .and. token_len == 1_i32 .and. is_dash_token)
        return
      end if
    end do
  end function legacy_persisted_field_is_empty

end module mod_cache_store
