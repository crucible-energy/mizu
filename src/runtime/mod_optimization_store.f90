module mod_optimization_store
  use mod_kinds,      only: i32, i64
  use mod_cache_keys, only: MAX_CACHE_KEY_LEN
  use mod_cache_store, only: quote_persisted_text

  implicit none

  private
  public :: OPT_EVIDENCE_VALID, OPT_INVALIDATION_WORKLOAD_CHANGED
  public :: OPT_INVALIDATION_CANDIDATE_CHANGED, OPT_INVALIDATION_PLAN_CHANGED
  public :: OPT_INVALIDATION_STALE_CANDIDATE_SET
  public :: optimization_candidate_record, optimization_entry_record
  public :: runtime_optimization_store
  public :: initialize_runtime_optimization_store, reset_runtime_optimization_store
  public :: record_execution_sample, lookup_winner_plan_id
  public :: lookup_winner_candidate, lookup_optimization_entry_stats
  public :: invalidate_optimization_entry, invalidate_optimization_candidate
  public :: invalidate_optimization_plan, invalidate_stale_optimization_candidates
  public :: load_runtime_optimization_store, save_runtime_optimization_store

  integer(i32), parameter :: INITIAL_ENTRY_CAPACITY = 16_i32
  integer(i32), parameter :: INITIAL_CANDIDATE_CAPACITY = 4_i32
  integer(i32), parameter :: MAX_RECORD_LINE_LEN = (4_i32 * MAX_CACHE_KEY_LEN) + 128_i32

  integer(i32), parameter :: OPT_EVIDENCE_VALID = 0_i32
  integer(i32), parameter :: OPT_INVALIDATION_WORKLOAD_CHANGED = 1_i32
  integer(i32), parameter :: OPT_INVALIDATION_CANDIDATE_CHANGED = 2_i32
  integer(i32), parameter :: OPT_INVALIDATION_PLAN_CHANGED = 3_i32
  integer(i32), parameter :: OPT_INVALIDATION_STALE_CANDIDATE_SET = 4_i32

  type :: optimization_candidate_record
    integer(i64) :: plan_id = 0_i64
    integer(i64) :: sample_count = 0_i64
    integer(i64) :: cumulative_elapsed_us = 0_i64
    integer(i32) :: invalidation_reason = OPT_EVIDENCE_VALID
    logical      :: is_invalid = .false.
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text = ""
  end type optimization_candidate_record

  type :: optimization_entry_record
    character(len=MAX_CACHE_KEY_LEN) :: key_text = ""
    integer(i32) :: candidate_count = 0_i32
    type(optimization_candidate_record), allocatable :: candidates(:)
  end type optimization_entry_record

  type :: runtime_optimization_store
    integer(i32) :: entry_count = 0_i32
    type(optimization_entry_record), allocatable :: entries(:)
  end type runtime_optimization_store

contains

  subroutine initialize_runtime_optimization_store(store)
    type(runtime_optimization_store), intent(out) :: store

    store = runtime_optimization_store()
  end subroutine initialize_runtime_optimization_store

  subroutine reset_runtime_optimization_store(store)
    type(runtime_optimization_store), intent(inout) :: store
    integer(i32) :: index

    if (allocated(store%entries)) then
      do index = 1_i32, store%entry_count
        if (allocated(store%entries(index)%candidates)) then
          deallocate(store%entries(index)%candidates)
        end if
      end do
      deallocate(store%entries)
    end if
    store%entry_count = 0_i32
  end subroutine reset_runtime_optimization_store

  subroutine record_execution_sample(store, key_text, plan_id, elapsed_us, candidate_key_text)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i64), intent(in)                        :: elapsed_us
    character(len=*), intent(in), optional          :: candidate_key_text
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index
    integer(i64)                                    :: measured_elapsed_us

    if (len_trim(key_text) == 0) return
    if (plan_id == 0_i64) return

    measured_elapsed_us = max(0_i64, elapsed_us)
    entry_index = ensure_entry_index(store, trim(key_text))
    candidate_index = ensure_candidate_index(store%entries(entry_index), plan_id, candidate_key_text)

    store%entries(entry_index)%candidates(candidate_index)%sample_count = &
      store%entries(entry_index)%candidates(candidate_index)%sample_count + 1_i64
    store%entries(entry_index)%candidates(candidate_index)%cumulative_elapsed_us = &
      store%entries(entry_index)%candidates(candidate_index)%cumulative_elapsed_us + measured_elapsed_us
    store%entries(entry_index)%candidates(candidate_index)%invalidation_reason = OPT_EVIDENCE_VALID
    store%entries(entry_index)%candidates(candidate_index)%is_invalid = .false.
  end subroutine record_execution_sample

  subroutine lookup_winner_plan_id(store, key_text, winner_plan_id, has_winner)
    type(runtime_optimization_store), intent(in) :: store
    character(len=*), intent(in)                 :: key_text
    integer(i64), intent(out)                    :: winner_plan_id
    logical, intent(out)                         :: has_winner
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: best_index

    winner_plan_id = 0_i64
    has_winner = .false.
    if (len_trim(key_text) == 0) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return
    best_index = find_best_candidate_index(store%entries(entry_index))
    if (best_index > 0_i32) then
      winner_plan_id = store%entries(entry_index)%candidates(best_index)%plan_id
      has_winner = .true.
    end if
  end subroutine lookup_winner_plan_id

  subroutine lookup_winner_candidate(store, key_text, winner_plan_id, winner_candidate_key_text, &
                                     has_winner)
    type(runtime_optimization_store), intent(in) :: store
    character(len=*), intent(in)                 :: key_text
    integer(i64), intent(out)                    :: winner_plan_id
    character(len=*), intent(out)                :: winner_candidate_key_text
    logical, intent(out)                         :: has_winner
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: best_index

    winner_plan_id = 0_i64
    winner_candidate_key_text = ""
    has_winner = .false.
    if (len_trim(key_text) == 0) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return
    best_index = find_best_candidate_index(store%entries(entry_index))
    if (best_index > 0_i32) then
      winner_plan_id = store%entries(entry_index)%candidates(best_index)%plan_id
      winner_candidate_key_text = trim(store%entries(entry_index)%candidates(best_index)%candidate_key_text)
      has_winner = .true.
    end if
  end subroutine lookup_winner_candidate

  subroutine lookup_optimization_entry_stats(store, key_text, total_samples, candidate_count)
    type(runtime_optimization_store), intent(in) :: store
    character(len=*), intent(in)                 :: key_text
    integer(i64), intent(out)                    :: total_samples
    integer(i32), intent(out)                    :: candidate_count
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: candidate_index

    total_samples = 0_i64
    candidate_count = 0_i32
    if (len_trim(key_text) == 0) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return

    do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
      if (.not. candidate_is_selectable(store%entries(entry_index)%candidates(candidate_index))) cycle
      candidate_count = candidate_count + 1_i32
      total_samples = total_samples + store%entries(entry_index)%candidates(candidate_index)%sample_count
    end do
  end subroutine lookup_optimization_entry_stats

  subroutine invalidate_optimization_entry(store, key_text, reason_code, invalidated_count)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text
    integer(i32), intent(in)                        :: reason_code
    integer(i32), intent(out)                       :: invalidated_count
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index

    invalidated_count = 0_i32
    if (len_trim(key_text) == 0) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return

    do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
      call mark_candidate_invalid(store%entries(entry_index)%candidates(candidate_index), &
        reason_code, invalidated_count)
    end do
  end subroutine invalidate_optimization_entry

  subroutine invalidate_optimization_candidate(store, key_text, candidate_key_text, reason_code, &
                                               invalidated_count)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text
    character(len=*), intent(in)                    :: candidate_key_text
    integer(i32), intent(in)                        :: reason_code
    integer(i32), intent(out)                       :: invalidated_count
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index

    invalidated_count = 0_i32
    if (len_trim(key_text) == 0 .or. len_trim(candidate_key_text) == 0) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return

    do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
      if (trim(store%entries(entry_index)%candidates(candidate_index)%candidate_key_text) /= &
          trim(candidate_key_text)) cycle
      call mark_candidate_invalid(store%entries(entry_index)%candidates(candidate_index), &
        reason_code, invalidated_count)
    end do
  end subroutine invalidate_optimization_candidate

  subroutine invalidate_optimization_plan(store, key_text, plan_id, reason_code, invalidated_count)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i32), intent(in)                        :: reason_code
    integer(i32), intent(out)                       :: invalidated_count
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index

    invalidated_count = 0_i32
    if (len_trim(key_text) == 0 .or. plan_id == 0_i64) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return

    do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
      if (store%entries(entry_index)%candidates(candidate_index)%plan_id /= plan_id) cycle
      call mark_candidate_invalid(store%entries(entry_index)%candidates(candidate_index), &
        reason_code, invalidated_count)
    end do
  end subroutine invalidate_optimization_plan

  subroutine invalidate_stale_optimization_candidates(store, key_text, valid_candidate_key_texts, &
                                                      valid_candidate_count, invalidated_count)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text
    character(len=*), intent(in)                    :: valid_candidate_key_texts(:)
    integer(i32), intent(in)                        :: valid_candidate_count
    integer(i32), intent(out)                       :: invalidated_count
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index

    invalidated_count = 0_i32
    if (len_trim(key_text) == 0 .or. valid_candidate_count <= 0_i32) return

    entry_index = find_entry_index(store, trim(key_text))
    if (entry_index <= 0_i32) return

    do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
      if (.not. candidate_is_selectable(store%entries(entry_index)%candidates(candidate_index))) cycle
      if (candidate_key_is_current(store%entries(entry_index)%candidates(candidate_index)%candidate_key_text, &
          valid_candidate_key_texts, valid_candidate_count)) cycle
      call mark_candidate_invalid(store%entries(entry_index)%candidates(candidate_index), &
        OPT_INVALIDATION_STALE_CANDIDATE_SET, invalidated_count)
    end do
  end subroutine invalidate_stale_optimization_candidates

  subroutine load_runtime_optimization_store(store, file_path, loaded_ok)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: file_path
    logical, intent(out)                            :: loaded_ok
    character(len=MAX_RECORD_LINE_LEN)              :: line
    character(len=16)                               :: tag
    character(len=MAX_CACHE_KEY_LEN)                :: key_text
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_text
    integer(i64)                                    :: plan_id
    integer(i64)                                    :: sample_count
    integer(i64)                                    :: cumulative_elapsed_us
    integer(i32)                                    :: unit_id
    integer(i32)                                    :: ios
    integer(i32)                                    :: entry_index
    integer(i32)                                    :: candidate_index
    logical                                         :: exists

    loaded_ok = .false.
    if (len_trim(file_path) == 0) return

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) then
      loaded_ok = .true.
      return
    end if

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0) return

    call reset_runtime_optimization_store(store)
    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle

      tag = ""
      key_text = ""
      candidate_key_text = ""
      plan_id = 0_i64
      sample_count = 0_i64
      cumulative_elapsed_us = 0_i64

      read(line, *, iostat=ios) tag, key_text, plan_id, sample_count, cumulative_elapsed_us, candidate_key_text
      if (ios /= 0) then
        ios = 0
        read(line, *, iostat=ios) tag, key_text, plan_id, sample_count, cumulative_elapsed_us
        if (ios /= 0) cycle
        candidate_key_text = ""
      end if
      if (trim(tag) /= "candidate") cycle
      if (len_trim(key_text) == 0 .or. plan_id == 0_i64 .or. sample_count <= 0_i64) cycle

      entry_index = ensure_entry_index(store, trim(key_text))
      candidate_index = ensure_candidate_index(store%entries(entry_index), plan_id, candidate_key_text)
      store%entries(entry_index)%candidates(candidate_index)%sample_count = sample_count
      store%entries(entry_index)%candidates(candidate_index)%cumulative_elapsed_us = &
        max(0_i64, cumulative_elapsed_us)
    end do

    close(unit_id)
    loaded_ok = .true.
  end subroutine load_runtime_optimization_store

  subroutine save_runtime_optimization_store(store, file_path, saved_ok)
    type(runtime_optimization_store), intent(in) :: store
    character(len=*), intent(in)                 :: file_path
    logical, intent(out)                         :: saved_ok
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2)   :: quoted_key_text
    character(len=(2 * MAX_CACHE_KEY_LEN) + 2)   :: quoted_candidate_key
    integer(i32)                                 :: unit_id
    integer(i32)                                 :: ios
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: candidate_index

    saved_ok = .false.
    if (len_trim(file_path) == 0) return

    open(newunit=unit_id, file=trim(file_path), status="replace", action="write", iostat=ios)
    if (ios /= 0) return

    do entry_index = 1_i32, store%entry_count
      do candidate_index = 1_i32, store%entries(entry_index)%candidate_count
        if (store%entries(entry_index)%candidates(candidate_index)%plan_id == 0_i64) cycle
        if (store%entries(entry_index)%candidates(candidate_index)%sample_count <= 0_i64) cycle
        if (store%entries(entry_index)%candidates(candidate_index)%is_invalid) cycle
        quoted_key_text = quote_persisted_text(store%entries(entry_index)%key_text, MAX_CACHE_KEY_LEN)
        quoted_candidate_key = quote_persisted_text( &
          store%entries(entry_index)%candidates(candidate_index)%candidate_key_text, MAX_CACHE_KEY_LEN)
        write(unit_id, "(A,1X,A,1X,I0,1X,I0,1X,I0,1X,A)", iostat=ios) &
          "candidate", trim(quoted_key_text), &
          store%entries(entry_index)%candidates(candidate_index)%plan_id, &
          store%entries(entry_index)%candidates(candidate_index)%sample_count, &
          store%entries(entry_index)%candidates(candidate_index)%cumulative_elapsed_us, &
          trim(quoted_candidate_key)
        if (ios /= 0) then
          close(unit_id)
          return
        end if
      end do
    end do

    close(unit_id)
    saved_ok = .true.
  end subroutine save_runtime_optimization_store

  integer(i32) function ensure_entry_index(store, key_text) result(entry_index)
    type(runtime_optimization_store), intent(inout) :: store
    character(len=*), intent(in)                    :: key_text

    entry_index = find_entry_index(store, key_text)
    if (entry_index > 0_i32) return

    call ensure_store_capacity(store, max(INITIAL_ENTRY_CAPACITY, store%entry_count + 1_i32))
    store%entry_count = store%entry_count + 1_i32
    entry_index = store%entry_count
    store%entries(entry_index)%key_text = trim(key_text)
    store%entries(entry_index)%candidate_count = 0_i32
    if (allocated(store%entries(entry_index)%candidates)) then
      deallocate(store%entries(entry_index)%candidates)
    end if
  end function ensure_entry_index

  integer(i32) function ensure_candidate_index(entry, plan_id, candidate_key_text) result(candidate_index)
    type(optimization_entry_record), intent(inout) :: entry
    integer(i64), intent(in)                       :: plan_id
    character(len=*), intent(in), optional         :: candidate_key_text
    integer(i32)                                   :: index
    character(len=MAX_CACHE_KEY_LEN)               :: provided_candidate_key
    logical                                        :: has_provided_candidate_key

    provided_candidate_key = ""
    has_provided_candidate_key = .false.
    if (present(candidate_key_text)) then
      provided_candidate_key = trim(candidate_key_text)
      has_provided_candidate_key = len_trim(provided_candidate_key) > 0
    end if

    if (allocated(entry%candidates)) then
      do index = 1_i32, entry%candidate_count
        if (entry%candidates(index)%plan_id == plan_id) then
          if (has_provided_candidate_key) then
            if (len_trim(entry%candidates(index)%candidate_key_text) == 0) then
              entry%candidates(index)%candidate_key_text = trim(provided_candidate_key)
              candidate_index = index
              return
            end if
            if (trim(entry%candidates(index)%candidate_key_text) /= trim(provided_candidate_key)) cycle
          end if
          candidate_index = index
          return
        end if
      end do
    end if

    call ensure_candidate_capacity(entry, max(INITIAL_CANDIDATE_CAPACITY, entry%candidate_count + 1_i32))
    entry%candidate_count = entry%candidate_count + 1_i32
    candidate_index = entry%candidate_count
    entry%candidates(candidate_index) = optimization_candidate_record()
    entry%candidates(candidate_index)%plan_id = plan_id
    if (has_provided_candidate_key) then
      entry%candidates(candidate_index)%candidate_key_text = trim(provided_candidate_key)
    end if
  end function ensure_candidate_index

  integer(i32) function find_best_candidate_index(entry) result(best_index)
    type(optimization_entry_record), intent(in) :: entry
    integer(i32)                                :: candidate_index

    best_index = 0_i32
    do candidate_index = 1_i32, entry%candidate_count
      if (.not. candidate_is_selectable(entry%candidates(candidate_index))) cycle
      if (best_index == 0_i32 .or. &
          candidate_is_better(entry%candidates(candidate_index), entry%candidates(best_index))) then
        best_index = candidate_index
      end if
    end do
  end function find_best_candidate_index

  integer(i32) function find_entry_index(store, key_text) result(entry_index)
    type(runtime_optimization_store), intent(in) :: store
    character(len=*), intent(in)                 :: key_text
    integer(i32)                                 :: index

    entry_index = 0_i32
    do index = 1_i32, store%entry_count
      if (trim(store%entries(index)%key_text) == trim(key_text)) then
        entry_index = index
        return
      end if
    end do
  end function find_entry_index

  subroutine ensure_store_capacity(store, required_capacity)
    type(runtime_optimization_store), intent(inout) :: store
    integer(i32), intent(in)                        :: required_capacity
    type(optimization_entry_record), allocatable    :: resized(:)
    integer(i32)                                    :: current_capacity
    integer(i32)                                    :: next_capacity

    if (.not. allocated(store%entries)) then
      allocate(store%entries(required_capacity))
      store%entries = optimization_entry_record()
      return
    end if

    current_capacity = int(size(store%entries), kind=i32)
    if (required_capacity <= current_capacity) return

    next_capacity = max(required_capacity, current_capacity * 2_i32)
    allocate(resized(next_capacity))
    resized = optimization_entry_record()
    if (store%entry_count > 0_i32) then
      resized(1:store%entry_count) = store%entries(1:store%entry_count)
    end if
    call move_alloc(resized, store%entries)
  end subroutine ensure_store_capacity

  subroutine ensure_candidate_capacity(entry, required_capacity)
    type(optimization_entry_record), intent(inout) :: entry
    integer(i32), intent(in)                       :: required_capacity
    type(optimization_candidate_record), allocatable :: resized(:)
    integer(i32)                                     :: current_capacity
    integer(i32)                                     :: next_capacity

    if (.not. allocated(entry%candidates)) then
      allocate(entry%candidates(required_capacity))
      entry%candidates = optimization_candidate_record()
      return
    end if

    current_capacity = int(size(entry%candidates), kind=i32)
    if (required_capacity <= current_capacity) return

    next_capacity = max(required_capacity, current_capacity * 2_i32)
    allocate(resized(next_capacity))
    resized = optimization_candidate_record()
    if (entry%candidate_count > 0_i32) then
      resized(1:entry%candidate_count) = entry%candidates(1:entry%candidate_count)
    end if
    call move_alloc(resized, entry%candidates)
  end subroutine ensure_candidate_capacity

  pure logical function candidate_is_better(candidate, incumbent) result(is_better)
    type(optimization_candidate_record), intent(in) :: candidate
    type(optimization_candidate_record), intent(in) :: incumbent
    integer(i64) :: candidate_score_left
    integer(i64) :: candidate_score_right

    if (.not. candidate_is_selectable(candidate)) then
      is_better = .false.
      return
    end if
    if (.not. candidate_is_selectable(incumbent)) then
      is_better = .true.
      return
    end if

    candidate_score_left = candidate%cumulative_elapsed_us * incumbent%sample_count
    candidate_score_right = incumbent%cumulative_elapsed_us * candidate%sample_count

    if (candidate_score_left < candidate_score_right) then
      is_better = .true.
    else if (candidate_score_left > candidate_score_right) then
      is_better = .false.
    else if (candidate%sample_count > incumbent%sample_count) then
      is_better = .true.
    else if (candidate%sample_count < incumbent%sample_count) then
      is_better = .false.
    else
      is_better = candidate%plan_id < incumbent%plan_id
    end if
  end function candidate_is_better

  pure logical function candidate_is_selectable(candidate) result(is_selectable)
    type(optimization_candidate_record), intent(in) :: candidate

    is_selectable = .not. candidate%is_invalid .and. &
      candidate%plan_id /= 0_i64 .and. candidate%sample_count > 0_i64
  end function candidate_is_selectable

  pure logical function candidate_key_is_current(candidate_key_text, valid_candidate_key_texts, &
                                                 valid_candidate_count) result(is_current)
    character(len=*), intent(in) :: candidate_key_text
    character(len=*), intent(in) :: valid_candidate_key_texts(:)
    integer(i32), intent(in)     :: valid_candidate_count
    integer(i32)                 :: candidate_index

    is_current = .false.
    if (len_trim(candidate_key_text) == 0) return

    do candidate_index = 1_i32, min(valid_candidate_count, int(size(valid_candidate_key_texts), kind=i32))
      if (trim(candidate_key_text) == trim(valid_candidate_key_texts(candidate_index))) then
        is_current = .true.
        return
      end if
    end do
  end function candidate_key_is_current

  pure integer(i32) function sanitize_invalidation_reason(reason_code) result(resolved_reason)
    integer(i32), intent(in) :: reason_code

    select case (reason_code)
    case (OPT_INVALIDATION_WORKLOAD_CHANGED, OPT_INVALIDATION_CANDIDATE_CHANGED, &
          OPT_INVALIDATION_PLAN_CHANGED, OPT_INVALIDATION_STALE_CANDIDATE_SET)
      resolved_reason = reason_code
    case default
      resolved_reason = OPT_INVALIDATION_CANDIDATE_CHANGED
    end select
  end function sanitize_invalidation_reason

  subroutine mark_candidate_invalid(candidate, reason_code, invalidated_count)
    type(optimization_candidate_record), intent(inout) :: candidate
    integer(i32), intent(in)                           :: reason_code
    integer(i32), intent(inout)                        :: invalidated_count

    if (.not. candidate_is_selectable(candidate)) return

    candidate%is_invalid = .true.
    candidate%invalidation_reason = sanitize_invalidation_reason(reason_code)
    invalidated_count = invalidated_count + 1_i32
  end subroutine mark_candidate_invalid

end module mod_optimization_store
