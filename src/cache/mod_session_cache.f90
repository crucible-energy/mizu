module mod_session_cache
  use mod_kinds,      only: i32, i64
  use mod_status,     only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,      only: MIZU_BACKEND_FAMILY_NONE, MIZU_EXEC_ROUTE_NONE, MIZU_STAGE_NONE, &
                            MIZU_STAGE_PARK, session_handle, session_state
  use mod_cache_keys, only: MAX_CACHE_KEY_LEN, session_cache_key
  use mod_cache_store, only: artifact_metadata_record

  implicit none

  private
  public :: session_cache_record, runtime_session_cache
  public :: initialize_runtime_session_cache, reset_runtime_session_cache
  public :: record_session_cache_entry, lookup_session_cache_entry
  public :: mark_session_cache_entry_live, evict_one_inactive_session_cache_entry
  public :: session_cache_key_is_strict, session_cache_record_is_evictable
  public :: session_cache_retention_score, active_session_cache_entry_count

  integer(i32), parameter :: INITIAL_SESSION_CACHE_CAPACITY = 16_i32

  type :: session_cache_record
    type(session_cache_key)          :: key
    type(session_handle)             :: session_owner
    type(artifact_metadata_record)   :: checkpoint_metadata
    integer(i64)                     :: kv_token_count = 0_i64
    integer(i64)                     :: live_context_hash = 0_i64
    integer(i64)                     :: live_context_artifact_hash = 0_i64
    integer(i32)                     :: live_context_byte_count = 0_i32
    integer(i32)                     :: live_context_backend_family = MIZU_BACKEND_FAMILY_NONE
    integer(i32)                     :: live_context_execution_route = MIZU_EXEC_ROUTE_NONE
    integer(i32)                     :: live_context_producer_stage = MIZU_STAGE_NONE
    integer(i64)                     :: hit_count = 0_i64
    integer(i64)                     :: last_access_tick = 0_i64
    logical                          :: is_parked = .false.
    logical                          :: is_live = .false.
    logical                          :: is_resident = .false.
    logical                          :: is_evicted = .false.
  end type session_cache_record

  type :: runtime_session_cache
    integer(i32) :: entry_count = 0_i32
    integer(i64) :: clock_tick = 0_i64
    type(session_cache_record), allocatable :: entries(:)
  end type runtime_session_cache

contains

  subroutine initialize_runtime_session_cache(cache)
    type(runtime_session_cache), intent(out) :: cache

    cache = runtime_session_cache()
  end subroutine initialize_runtime_session_cache

  subroutine reset_runtime_session_cache(cache)
    type(runtime_session_cache), intent(inout) :: cache

    if (allocated(cache%entries)) deallocate(cache%entries)
    cache%entry_count = 0_i32
    cache%clock_tick = 0_i64
  end subroutine reset_runtime_session_cache

  subroutine record_session_cache_entry(cache, key, session, checkpoint_metadata, status_code, is_live)
    type(runtime_session_cache), intent(inout)  :: cache
    type(session_cache_key), intent(in)         :: key
    type(session_state), intent(in)             :: session
    type(artifact_metadata_record), intent(in)  :: checkpoint_metadata
    integer(i32), intent(out)                   :: status_code
    logical, intent(in), optional               :: is_live
    integer(i32)                                :: entry_index
    integer(i64)                                :: existing_hits
    logical                                     :: resolved_is_live

    if (.not. session_cache_key_is_strict(key) .or. &
        .not. session_matches_key(session, key) .or. &
        .not. metadata_matches_key(checkpoint_metadata, key)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    resolved_is_live = .not. session%is_parked
    if (present(is_live)) resolved_is_live = is_live

    entry_index = ensure_entry_index(cache, trim(key%key_text))
    existing_hits = cache%entries(entry_index)%hit_count

    cache%entries(entry_index)%key = key
    cache%entries(entry_index)%session_owner = session%handle
    cache%entries(entry_index)%checkpoint_metadata = checkpoint_metadata
    cache%entries(entry_index)%kv_token_count = session%kv_token_count
    cache%entries(entry_index)%live_context_hash = session%live_context_hash
    cache%entries(entry_index)%live_context_artifact_hash = session%live_context_artifact_hash
    cache%entries(entry_index)%live_context_byte_count = session%live_context_byte_count
    cache%entries(entry_index)%live_context_backend_family = session%live_context_backend_family
    cache%entries(entry_index)%live_context_execution_route = session%live_context_execution_route
    cache%entries(entry_index)%live_context_producer_stage = session%live_context_producer_stage
    cache%entries(entry_index)%hit_count = existing_hits
    cache%entries(entry_index)%last_access_tick = next_cache_tick(cache)
    cache%entries(entry_index)%is_parked = session%is_parked
    cache%entries(entry_index)%is_live = resolved_is_live
    cache%entries(entry_index)%is_resident = session%has_resident_live_context
    cache%entries(entry_index)%is_evicted = session%is_evicted

    status_code = MIZU_STATUS_OK
  end subroutine record_session_cache_entry

  subroutine lookup_session_cache_entry(cache, key, record, found)
    type(runtime_session_cache), intent(inout) :: cache
    type(session_cache_key), intent(in)        :: key
    type(session_cache_record), intent(out)    :: record
    logical, intent(out)                       :: found
    integer(i32)                               :: entry_index

    record = session_cache_record()
    found = .false.
    if (.not. session_cache_key_is_strict(key)) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = cache%entries(entry_index)%hit_count + 1_i64
    cache%entries(entry_index)%last_access_tick = next_cache_tick(cache)
    record = cache%entries(entry_index)
    found = .true.
  end subroutine lookup_session_cache_entry

  subroutine mark_session_cache_entry_live(cache, key, is_live, status_code)
    type(runtime_session_cache), intent(inout) :: cache
    type(session_cache_key), intent(in)        :: key
    logical, intent(in)                        :: is_live
    integer(i32), intent(out)                  :: status_code
    integer(i32)                               :: entry_index

    status_code = MIZU_STATUS_INVALID_ARGUMENT
    if (.not. session_cache_key_is_strict(key)) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%is_live = is_live
    if (is_live) cache%entries(entry_index)%is_evicted = .false.
    cache%entries(entry_index)%last_access_tick = next_cache_tick(cache)
    status_code = MIZU_STATUS_OK
  end subroutine mark_session_cache_entry_live

  subroutine evict_one_inactive_session_cache_entry(cache, evicted_record, found)
    type(runtime_session_cache), intent(inout) :: cache
    type(session_cache_record), intent(out)    :: evicted_record
    logical, intent(out)                       :: found
    integer(i32)                               :: index
    integer(i32)                               :: candidate_index
    integer(i64)                               :: candidate_score
    integer(i64)                               :: score

    evicted_record = session_cache_record()
    found = .false.
    candidate_index = 0_i32
    candidate_score = 0_i64
    if (.not. allocated(cache%entries)) return

    do index = 1_i32, cache%entry_count
      if (.not. session_cache_record_is_evictable(cache%entries(index))) cycle
      score = session_cache_retention_score(cache%entries(index))
      if (candidate_index <= 0_i32) then
        candidate_index = index
        candidate_score = score
        cycle
      end if
      if (score < candidate_score .or. &
          (score == candidate_score .and. cache%entries(index)%last_access_tick < &
           cache%entries(candidate_index)%last_access_tick)) then
        candidate_index = index
        candidate_score = score
      end if
    end do

    if (candidate_index <= 0_i32) return

    cache%entries(candidate_index)%is_resident = .false.
    cache%entries(candidate_index)%is_evicted = .true.
    cache%entries(candidate_index)%last_access_tick = next_cache_tick(cache)
    evicted_record = cache%entries(candidate_index)
    found = .true.
  end subroutine evict_one_inactive_session_cache_entry

  pure logical function session_cache_key_is_strict(key) result(is_strict)
    type(session_cache_key), intent(in) :: key

    is_strict = len_trim(key%key_text) > 0 .and. &
      key%versions%schema_version > 0_i32 .and. &
      key%versions%abi_version > 0_i32 .and. &
      key%logical_model_hash /= 0_i64 .and. &
      key%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
      key%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
      key%max_context_tokens > 0_i64 .and. &
      key%max_decode_tokens >= 0_i64 .and. &
      len_trim(key%device_key) > 0 .and. &
      index(key%key_text, "session:v") == 1 .and. &
      index(key%key_text, ":abi=" // trim(i32_to_text(key%versions%abi_version))) > 0 .and. &
      index(key%key_text, ":planner=" // trim(i32_to_text(key%versions%planner_version))) > 0 .and. &
      index(key%key_text, ":model=" // trim(i64_to_text(key%logical_model_hash))) > 0 .and. &
      index(key%key_text, ":backend=" // trim(i32_to_text(key%backend_family))) > 0 .and. &
      index(key%key_text, ":route=" // trim(i32_to_text(key%execution_route))) > 0 .and. &
      index(key%key_text, ":ctx=" // trim(i64_to_text(key%max_context_tokens))) > 0 .and. &
      index(key%key_text, ":decode=" // trim(i64_to_text(key%max_decode_tokens))) > 0 .and. &
      index(key%key_text, ":device=" // trim(key%device_key)) > 0
  end function session_cache_key_is_strict

  pure logical function session_cache_record_is_evictable(record) result(is_evictable)
    type(session_cache_record), intent(in) :: record

    is_evictable = .not. record%is_live .and. record%is_parked .and. &
      .not. record%is_evicted .and. &
      (record%checkpoint_metadata%is_materialized .or. .not. record%is_resident)
  end function session_cache_record_is_evictable

  pure integer(i64) function session_cache_retention_score(record) result(score)
    type(session_cache_record), intent(in) :: record

    score = max(0_i64, record%kv_token_count) * 1000000_i64 + &
      max(0_i64, int(record%live_context_byte_count, kind=i64)) * 1000_i64 + &
      max(0_i64, record%hit_count) * 100_i64 + &
      max(0_i64, record%last_access_tick)
  end function session_cache_retention_score

  pure integer(i32) function active_session_cache_entry_count(cache) result(entry_count)
    type(runtime_session_cache), intent(in) :: cache
    integer(i32)                            :: entry_index

    entry_count = 0_i32
    if (.not. allocated(cache%entries)) return

    do entry_index = 1_i32, cache%entry_count
      if (.not. cache%entries(entry_index)%is_evicted) entry_count = entry_count + 1_i32
    end do
  end function active_session_cache_entry_count

  pure logical function session_matches_key(session, key) result(matches)
    type(session_state), intent(in)     :: session
    type(session_cache_key), intent(in) :: key

    matches = session%has_live_context .and. &
      session%kv_token_count > 0_i64 .and. &
      session%live_context_hash /= 0_i64 .and. &
      session%live_context_backend_family == key%backend_family .and. &
      session%live_context_execution_route == key%execution_route
  end function session_matches_key

  pure logical function metadata_matches_key(metadata, key) result(matches)
    type(artifact_metadata_record), intent(in) :: metadata
    type(session_cache_key), intent(in)        :: key

    matches = .true.
    if (metadata%stage_kind /= MIZU_STAGE_NONE .and. metadata%stage_kind /= MIZU_STAGE_PARK) matches = .false.
    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        metadata%backend_family /= key%backend_family) matches = .false.
    if (metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        metadata%execution_route /= key%execution_route) matches = .false.
  end function metadata_matches_key

  integer(i32) function ensure_entry_index(cache, key_text) result(entry_index)
    type(runtime_session_cache), intent(inout) :: cache
    character(len=*), intent(in)               :: key_text

    entry_index = find_entry_index(cache, key_text)
    if (entry_index > 0_i32) return

    call ensure_entry_capacity(cache, cache%entry_count + 1_i32)
    cache%entry_count = cache%entry_count + 1_i32
    entry_index = cache%entry_count
    cache%entries(entry_index) = session_cache_record()
    cache%entries(entry_index)%key%key_text = trim(key_text)
  end function ensure_entry_index

  integer(i32) function find_entry_index(cache, key_text) result(entry_index)
    type(runtime_session_cache), intent(in) :: cache
    character(len=*), intent(in)            :: key_text
    integer(i32)                            :: index

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
    type(runtime_session_cache), intent(inout) :: cache
    integer(i32), intent(in)                   :: required_capacity
    type(session_cache_record), allocatable    :: resized_entries(:)
    integer(i32)                               :: new_capacity

    if (.not. allocated(cache%entries)) then
      allocate(cache%entries(max(INITIAL_SESSION_CACHE_CAPACITY, required_capacity)))
      cache%entries = session_cache_record()
      return
    end if

    if (size(cache%entries) >= required_capacity) return

    new_capacity = max(required_capacity, int(size(cache%entries), kind=i32) * 2_i32)
    allocate(resized_entries(new_capacity))
    resized_entries = session_cache_record()
    if (cache%entry_count > 0_i32) resized_entries(1:cache%entry_count) = &
      cache%entries(1:cache%entry_count)
    call move_alloc(resized_entries, cache%entries)
  end subroutine ensure_entry_capacity

  integer(i64) function next_cache_tick(cache) result(tick_value)
    type(runtime_session_cache), intent(inout) :: cache

    cache%clock_tick = cache%clock_tick + 1_i64
    tick_value = cache%clock_tick
  end function next_cache_tick

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

end module mod_session_cache
