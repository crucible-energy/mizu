module mod_backend_registry
  use mod_kinds,            only: i32, i64
  use mod_status,           only: MIZU_STATUS_OK
  use mod_types,            only: runtime_state, backend_descriptor, MAX_RUNTIME_BACKENDS, &
                                  MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_MASK_APPLE_ANE, &
                                  MIZU_BACKEND_MASK_APPLE_METAL, MIZU_BACKEND_MASK_CUDA
  use mod_backend_contract, only: capability_probe_request, capability_probe_result
  use mod_apple_capability, only: probe_apple_backend
  use mod_cuda_capability,  only: probe_cuda_backend

  implicit none

  private
  public :: runtime_backend_registry
  public :: initialize_runtime_backend_registry, reset_runtime_backend_registry
  public :: register_backend_probe_result, probe_runtime_backend_registry
  public :: apply_backend_registry_to_runtime

  type :: runtime_backend_registry
    integer(i32)              :: backend_count = 0_i32
    integer(i64)              :: available_backend_mask = MIZU_BACKEND_MASK_NONE
    type(backend_descriptor)  :: descriptors(MAX_RUNTIME_BACKENDS)
  end type runtime_backend_registry

contains

  subroutine initialize_runtime_backend_registry(registry)
    type(runtime_backend_registry), intent(out) :: registry

    registry%backend_count = 0_i32
    registry%available_backend_mask = MIZU_BACKEND_MASK_NONE
    registry%descriptors = backend_descriptor()
  end subroutine initialize_runtime_backend_registry

  subroutine reset_runtime_backend_registry(registry)
    type(runtime_backend_registry), intent(inout) :: registry

    call initialize_runtime_backend_registry(registry)
  end subroutine reset_runtime_backend_registry

  subroutine register_backend_probe_result(registry, result)
    type(runtime_backend_registry), intent(inout) :: registry
    type(capability_probe_result), intent(in)     :: result
    type(backend_descriptor)                      :: descriptor

    if (.not. result%descriptor%is_available) return
    if (result%descriptor%route_mask == MIZU_BACKEND_MASK_NONE) return
    if (registry%backend_count >= MAX_RUNTIME_BACKENDS) return

    descriptor = result%descriptor
    descriptor%planner_version = max(0_i64, result%constraints%planner_version)
    registry%backend_count = registry%backend_count + 1_i32
    registry%descriptors(registry%backend_count) = descriptor
    registry%available_backend_mask = ior(registry%available_backend_mask, descriptor%route_mask)
  end subroutine register_backend_probe_result

  subroutine probe_runtime_backend_registry(registry)
    type(runtime_backend_registry), intent(inout) :: registry
    type(capability_probe_request)                :: request
    type(capability_probe_result)                 :: result
    integer(i32)                                  :: status_code

    call reset_runtime_backend_registry(registry)

    request = capability_probe_request()
    request%allowed_backend_mask = ior(ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL), &
      MIZU_BACKEND_MASK_CUDA)

    call probe_apple_backend(request, result, status_code)
    if (status_code == MIZU_STATUS_OK) call register_backend_probe_result(registry, result)

    call probe_cuda_backend(request, result, status_code)
    if (status_code == MIZU_STATUS_OK) call register_backend_probe_result(registry, result)
  end subroutine probe_runtime_backend_registry

  subroutine apply_backend_registry_to_runtime(registry, runtime)
    type(runtime_backend_registry), intent(in) :: registry
    type(runtime_state), intent(inout)         :: runtime

    runtime%detected_backend_mask = registry%available_backend_mask
    runtime%detected_backend_count = registry%backend_count
    runtime%detected_backends = backend_descriptor()
    if (registry%backend_count > 0_i32) then
      runtime%detected_backends(1:registry%backend_count) = &
        registry%descriptors(1:registry%backend_count)
    end if
  end subroutine apply_backend_registry_to_runtime

end module mod_backend_registry
