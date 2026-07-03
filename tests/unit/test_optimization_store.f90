program test_optimization_store
  use mod_kinds,              only: i32, i64
  use mod_cache_keys,         only: MAX_CACHE_KEY_LEN
  use mod_optimization_store, only: runtime_optimization_store, &
                                    OPT_INVALIDATION_CANDIDATE_CHANGED, &
                                    OPT_INVALIDATION_PLAN_CHANGED, &
                                    OPT_INVALIDATION_WORKLOAD_CHANGED, &
                                    initialize_runtime_optimization_store, &
                                    reset_runtime_optimization_store, &
                                    record_execution_sample, lookup_winner_plan_id, &
                                    lookup_winner_candidate, &
                                    lookup_optimization_entry_stats, &
                                    invalidate_optimization_entry, &
                                    invalidate_optimization_candidate, &
                                    invalidate_optimization_plan, &
                                    invalidate_stale_optimization_candidates, &
                                    load_runtime_optimization_store, &
                                    save_runtime_optimization_store

  implicit none

  type(runtime_optimization_store) :: store
  type(runtime_optimization_store) :: reloaded_store
  character(len=MAX_CACHE_KEY_LEN) :: winner_candidate_key_text
  character(len=MAX_CACHE_KEY_LEN) :: valid_candidate_key_texts(3)
  integer(i64)                     :: total_samples
  integer(i64)                     :: winner_plan_id
  integer(i32)                     :: candidate_count
  integer(i32)                     :: invalidated_count
  logical                          :: has_winner
  logical                          :: saved_ok
  logical                          :: loaded_ok
  character(len=*), parameter      :: store_path = "/tmp/mizu_test_optimization_store.txt"

  call initialize_runtime_optimization_store(store)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_false("empty store should not have winner", has_winner)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 10_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_true("winner should exist after first sample", has_winner)
  call expect_equal_i64("first winner", winner_plan_id, 101_i64)

  call record_execution_sample(store, "prefill:key:a", 202_i64, 5_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("lower average candidate should take winner", winner_plan_id, 202_i64)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 1_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("winner should remain faster incumbent", winner_plan_id, 202_i64)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 1_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("winner should promote after better measured average", winner_plan_id, 101_i64)

  call record_execution_sample(store, "prefill:key:stale", 301_i64, 10_i64, &
    "candidate:cuda:planner=1")
  call record_execution_sample(store, "prefill:key:stale", 302_i64, 3_i64, &
    "candidate:ane:planner=1")
  call lookup_winner_candidate(store, "prefill:key:stale", winner_plan_id, &
    winner_candidate_key_text, has_winner)
  call expect_true("stale fixture should start with winner", has_winner)
  call expect_equal_i64("stale fixture initial winner", winner_plan_id, 302_i64)
  call expect_equal_string("stale fixture initial winner key", winner_candidate_key_text, &
    "candidate:ane:planner=1")

  call invalidate_optimization_candidate(store, "prefill:key:stale", &
    "candidate:ane:planner=1", OPT_INVALIDATION_CANDIDATE_CHANGED, invalidated_count)
  call expect_equal_i32("candidate invalidation should retire one record", invalidated_count, 1_i32)
  call lookup_winner_candidate(store, "prefill:key:stale", winner_plan_id, &
    winner_candidate_key_text, has_winner)
  call expect_true("valid candidate should remain after stale candidate invalidation", has_winner)
  call expect_equal_i64("candidate invalidation should expose slower still-valid plan", &
    winner_plan_id, 301_i64)
  call expect_equal_string("candidate invalidation should expose cuda key", &
    winner_candidate_key_text, "candidate:cuda:planner=1")

  call lookup_optimization_entry_stats(store, "prefill:key:stale", total_samples, candidate_count)
  call expect_equal_i64("stats should ignore invalidated candidate samples", total_samples, 1_i64)
  call expect_equal_i32("stats should ignore invalidated candidate count", candidate_count, 1_i32)

  call record_execution_sample(store, "prefill:key:stale", 302_i64, 2_i64, &
    "candidate:ane:planner=1")
  call lookup_winner_plan_id(store, "prefill:key:stale", winner_plan_id, has_winner)
  call expect_true("fresh evidence should revive invalidated candidate", has_winner)
  call expect_equal_i64("revived candidate should compete again", winner_plan_id, 302_i64)

  call invalidate_optimization_plan(store, "prefill:key:stale", 302_i64, &
    OPT_INVALIDATION_PLAN_CHANGED, invalidated_count)
  call expect_equal_i32("plan invalidation should retire revived plan", invalidated_count, 1_i32)
  call lookup_winner_plan_id(store, "prefill:key:stale", winner_plan_id, has_winner)
  call expect_true("plan invalidation should leave other candidate valid", has_winner)
  call expect_equal_i64("plan invalidation fallback winner", winner_plan_id, 301_i64)

  call record_execution_sample(store, "prefill:key:set", 401_i64, 7_i64, &
    "candidate:cuda:pack=v1")
  call record_execution_sample(store, "prefill:key:set", 402_i64, 1_i64, &
    "candidate:ane:pack=v1")
  valid_candidate_key_texts = ""
  valid_candidate_key_texts(1) = "candidate:cuda:pack=v2"
  valid_candidate_key_texts(2) = "candidate:ane:pack=v2"
  call invalidate_stale_optimization_candidates(store, "prefill:key:set", &
    valid_candidate_key_texts, 2_i32, invalidated_count)
  call expect_equal_i32("candidate-set invalidation should retire all stale records", &
    invalidated_count, 2_i32)
  call lookup_winner_plan_id(store, "prefill:key:set", winner_plan_id, has_winner)
  call expect_false("fully stale candidate set should not have winner", has_winner)
  call lookup_optimization_entry_stats(store, "prefill:key:set", total_samples, candidate_count)
  call expect_equal_i64("fully stale candidate set should have no active samples", total_samples, 0_i64)
  call expect_equal_i32("fully stale candidate set should have no active candidates", &
    candidate_count, 0_i32)

  call record_execution_sample(store, "prefill:key:set", 501_i64, 4_i64, &
    "candidate:cuda:pack=v2")
  call lookup_winner_candidate(store, "prefill:key:set", winner_plan_id, &
    winner_candidate_key_text, has_winner)
  call expect_true("new candidate-set evidence should produce winner", has_winner)
  call expect_equal_i64("new candidate-set winner", winner_plan_id, 501_i64)
  call expect_equal_string("new candidate-set winner key", winner_candidate_key_text, &
    "candidate:cuda:pack=v2")

  call invalidate_optimization_entry(store, "prefill:key:set", &
    OPT_INVALIDATION_WORKLOAD_CHANGED, invalidated_count)
  call expect_equal_i32("entry invalidation should retire remaining active candidate", &
    invalidated_count, 1_i32)
  call lookup_winner_plan_id(store, "prefill:key:set", winner_plan_id, has_winner)
  call expect_false("entry invalidation should clear winner", has_winner)

  call record_execution_sample(store, "prefill key spaced", 601_i64, 9_i64, &
    "candidate cuda planner 1")
  call record_execution_sample(store, "prefill key spaced", 602_i64, 2_i64, &
    "candidate ane planner 1")
  call lookup_winner_candidate(store, "prefill key spaced", winner_plan_id, &
    winner_candidate_key_text, has_winner)
  call expect_true("spaced-key fixture should produce winner", has_winner)
  call expect_equal_i64("spaced-key fixture winner", winner_plan_id, 602_i64)
  call expect_equal_string("spaced-key fixture winner key", winner_candidate_key_text, &
    "candidate ane planner 1")

  call execute_command_line("rm -f " // store_path)
  call save_runtime_optimization_store(store, store_path, saved_ok)
  call expect_true("optimization store save should succeed", saved_ok)

  call initialize_runtime_optimization_store(reloaded_store)
  call load_runtime_optimization_store(reloaded_store, store_path, loaded_ok)
  call expect_true("optimization store load should succeed", loaded_ok)
  call lookup_winner_plan_id(reloaded_store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_true("reloaded store should preserve winner", has_winner)
  call expect_equal_i64("reloaded winner should match", winner_plan_id, 101_i64)
  call lookup_winner_plan_id(reloaded_store, "prefill:key:set", winner_plan_id, has_winner)
  call expect_false("reloaded store should not preserve invalidated entry winner", has_winner)
  call lookup_winner_plan_id(reloaded_store, "prefill:key:stale", winner_plan_id, has_winner)
  call expect_true("reloaded store should preserve surviving non-stale evidence", has_winner)
  call expect_equal_i64("reloaded surviving non-stale winner", winner_plan_id, 301_i64)
  call lookup_winner_candidate(reloaded_store, "prefill key spaced", winner_plan_id, &
    winner_candidate_key_text, has_winner)
  call expect_true("reloaded spaced-key fixture should preserve winner", has_winner)
  call expect_equal_i64("reloaded spaced-key fixture winner", winner_plan_id, 602_i64)
  call expect_equal_string("reloaded spaced-key fixture winner key", winner_candidate_key_text, &
    "candidate ane planner 1")
  call execute_command_line("rm -f " // store_path)

  call reset_runtime_optimization_store(store)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_false("reset store should clear winner", has_winner)

  write(*, "(A)") "test_optimization_store: PASS"

contains

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i64

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

end program test_optimization_store
