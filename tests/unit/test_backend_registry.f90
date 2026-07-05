program test_backend_registry
  use mod_kinds,            only: i32, i64
  use mod_types,            only: runtime_state, runtime_config, backend_descriptor, &
                                  MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                                  MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL, &
                                  MIZU_BACKEND_MASK_CUDA
  use mod_runtime,          only: initialize_runtime_state
  use mod_backend_contract, only: capability_probe_result
  use mod_backend_registry, only: runtime_backend_registry, initialize_runtime_backend_registry, &
                                  register_backend_probe_result, apply_backend_registry_to_runtime

  implicit none

  type(runtime_backend_registry) :: registry
  type(runtime_state)            :: runtime
  type(capability_probe_result)  :: apple_result
  type(capability_probe_result)  :: cuda_result

  call initialize_runtime_backend_registry(registry)
  call expect_equal_i32("registry should start empty", registry%backend_count, 0_i32)
  call expect_equal_i64("registry should start with empty mask", registry%available_backend_mask, 0_i64)

  apple_result = capability_probe_result()
  apple_result%descriptor = backend_descriptor()
  apple_result%descriptor%family = MIZU_BACKEND_FAMILY_APPLE
  apple_result%descriptor%route_mask = ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL)
  apple_result%descriptor%is_available = .true.
  apple_result%descriptor%backend_name = "apple"
  apple_result%descriptor%device_name = "mac_mini"
  apple_result%constraints%planner_version = 1_i64
  call register_backend_probe_result(registry, apple_result)

  cuda_result = capability_probe_result()
  cuda_result%descriptor = backend_descriptor()
  cuda_result%descriptor%family = MIZU_BACKEND_FAMILY_CUDA
  cuda_result%descriptor%route_mask = MIZU_BACKEND_MASK_CUDA
  cuda_result%descriptor%is_available = .true.
  cuda_result%descriptor%backend_name = "cuda"
  cuda_result%descriptor%device_name = "nvidia_test_gpu"
  cuda_result%constraints%planner_version = 90_i64
  call register_backend_probe_result(registry, cuda_result)

  call expect_equal_i32("registry should retain two detected families", registry%backend_count, 2_i32)
  call expect_true("registry mask should include ANE", iand(registry%available_backend_mask, &
    MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64)
  call expect_true("registry mask should include Metal", iand(registry%available_backend_mask, &
    MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64)
  call expect_true("registry mask should include CUDA", iand(registry%available_backend_mask, &
    MIZU_BACKEND_MASK_CUDA) /= 0_i64)

  call initialize_runtime_state(runtime, runtime_config())
  call apply_backend_registry_to_runtime(registry, runtime)

  call expect_equal_i32("runtime should retain detected backend count", runtime%detected_backend_count, 2_i32)
  call expect_equal_i64("runtime should retain detected backend mask", runtime%detected_backend_mask, &
    registry%available_backend_mask)
  call expect_equal_i32("first runtime backend should be apple", runtime%detected_backends(1)%family, &
    MIZU_BACKEND_FAMILY_APPLE)
  call expect_equal_i32("second runtime backend should be cuda", runtime%detected_backends(2)%family, &
    MIZU_BACKEND_FAMILY_CUDA)
  call expect_equal_string("runtime should retain apple device name", trim(runtime%detected_backends(1)%device_name), &
    "mac_mini")
  call expect_equal_string("runtime should retain cuda device name", trim(runtime%detected_backends(2)%device_name), &
    "nvidia_test_gpu")
  call expect_equal_i64("runtime should retain apple planner version", &
    runtime%detected_backends(1)%planner_version, 1_i64)
  call expect_equal_i64("runtime should retain cuda planner version", &
    runtime%detected_backends(2)%planner_version, 90_i64)

  write(*, "(A)") "test_backend_registry: PASS"

contains

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i32), intent(in)     :: actual
    integer(i32), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

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

end program test_backend_registry
