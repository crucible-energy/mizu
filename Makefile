FC := gfortran
CC ?= gcc
CXX ?= g++
NVCC ?= nvcc
OBJC ?= clang
UNAME_S := $(shell uname -s)

FFLAGS ?= -std=f2018 -Wall -Wextra
CFLAGS ?= -std=c11 -Wall -Wextra
CXXFLAGS ?= -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -Iinclude -Isrc/backends/cuda
OBJCFLAGS ?= -Wall -Wextra -fobjc-arc
DEBUG_FFLAGS ?= $(FFLAGS) -fcheck=all -fbacktrace

HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

BUILD_DIR := build
DEBUG_BUILD_DIR ?= $(BUILD_DIR)-debug
TEST_DIR := $(BUILD_DIR)/tests
CUDA_BRIDGE_OBJ := $(BUILD_DIR)/cuda_bridge.o
APPLE_BRIDGE_OBJ := $(BUILD_DIR)/apple_bridge.o

COMMON_F90 := \
	src/common/mod_kinds.f90 \
	src/common/mod_status.f90 \
	src/common/mod_memory.f90 \
	src/common/mod_types.f90 \
	src/common/mod_errors.f90

MODEL_F90 := \
	src/model/mod_model_manifest.f90 \
	src/model/mod_model_import_layout.f90 \
	src/model/mod_model_loader.f90

CACHE_F90 := \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/cache/mod_plan_cache.f90 \
	src/cache/mod_weight_cache.f90 \
	src/cache/mod_session_cache.f90 \
	src/cache/mod_mm_cache.f90

RUNTIME_F90 := \
	src/runtime/mod_request.f90 \
	src/runtime/mod_workspace.f90 \
	src/runtime/mod_scheduler.f90 \
	src/runtime/mod_runtime.f90 \
	src/runtime/mod_session.f90 \
	src/runtime/mod_optimization_store.f90

BACKEND_F90 := \
	src/backends/mod_backend_contract.f90 \
	src/backends/mod_backend_probe_support.f90 \
	src/backends/apple/mod_apple_bridge.f90 \
	src/backends/apple/mod_apple_capability.f90 \
	src/backends/apple/mod_apple_planner.f90 \
	src/backends/apple/mod_apple_executor.f90 \
	src/backends/cuda/mod_cuda_bridge.f90 \
	src/backends/cuda/mod_cuda_capability.f90 \
	src/backends/cuda/mod_cuda_planner.f90 \
	src/backends/cuda/mod_cuda_executor.f90 \
	src/backends/mod_backend_registry.f90

CAPI_F90 := src/c_api/mod_c_api.f90

UNIT_BINS := \
	$(TEST_DIR)/test_model_manifest_loader \
	$(TEST_DIR)/test_cache_keys \
	$(TEST_DIR)/test_cache_store \
	$(TEST_DIR)/test_plan_cache \
	$(TEST_DIR)/test_weight_cache \
	$(TEST_DIR)/test_session_cache \
	$(TEST_DIR)/test_mm_cache \
	$(TEST_DIR)/test_optimization_store \
	$(TEST_DIR)/test_backend_registry \
	$(TEST_DIR)/test_runtime_workspace \
	$(TEST_DIR)/test_session_staging \
	$(TEST_DIR)/test_apple_planner \
	$(TEST_DIR)/test_apple_executor \
	$(TEST_DIR)/test_cuda_planner \
	$(TEST_DIR)/test_cuda_executor

CONTRACT_SMOKES := \
	$(TEST_DIR)/test_header_c_smoke.o \
	$(TEST_DIR)/test_header_cpp_smoke.o

CONTRACT_BINS := \
	$(TEST_DIR)/test_backend_availability \
	$(TEST_DIR)/test_backend_routing_contracts \
	$(TEST_DIR)/test_handle_arena_capacity \
	$(TEST_DIR)/test_handle_lifecycle \
	$(TEST_DIR)/test_model_open_failures \
	$(TEST_DIR)/test_modal_input_validation \
	$(TEST_DIR)/test_opaque_handles \
	$(TEST_DIR)/test_token_only_execution \
	$(TEST_DIR)/test_decode_terminal_status \
	$(TEST_DIR)/test_session_eviction \
	$(TEST_DIR)/test_session_checkpoint_restore_failures \
	$(TEST_DIR)/test_session_state_guards \
	$(TEST_DIR)/test_struct_sizes \
	$(TEST_DIR)/test_runtime_last_error_propagation \
	$(TEST_DIR)/test_cuda_artifacts \
	$(TEST_DIR)/test_qwench_gguf_cuda_smoke \
	$(TEST_DIR)/test_stage_reports

TOOL_TESTS := \
	tests/tooling/test_format_local.py \
	tests/tooling/test_make_dependencies.py \
	tests/tooling/test_gguf_to_mizu.py \
	tests/tooling/test_hf_safetensors_to_mizu.py

BUILD_RECIPE_INPUTS := Makefile

UNIT_COMMON_F90 := \
	src/common/mod_kinds.f90 \
	src/common/mod_status.f90 \
	src/common/mod_types.f90

UNIT_MEMORY_F90 := \
	$(UNIT_COMMON_F90) \
	src/common/mod_memory.f90

UNIT_MODEL_F90 := \
	src/model/mod_model_manifest.f90 \
	src/model/mod_model_import_layout.f90 \
	src/model/mod_model_loader.f90

MODEL_MANIFEST_LOADER_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	tests/unit/test_model_manifest_loader.f90

CACHE_KEYS_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	src/cache/mod_cache_keys.f90 \
	tests/unit/test_cache_keys.f90

CACHE_STORE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	src/model/mod_model_manifest.f90 \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	tests/unit/test_cache_store.f90

PLAN_CACHE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/cache/mod_plan_cache.f90 \
	tests/unit/test_plan_cache.f90

WEIGHT_CACHE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/cache/mod_weight_cache.f90 \
	tests/unit/test_weight_cache.f90

SESSION_CACHE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/cache/mod_session_cache.f90 \
	src/runtime/mod_session.f90 \
	tests/unit/test_session_cache.f90

MM_CACHE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	$(UNIT_MODEL_F90) \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/cache/mod_mm_cache.f90 \
	tests/unit/test_mm_cache.f90

OPTIMIZATION_STORE_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	src/model/mod_model_manifest.f90 \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90 \
	src/runtime/mod_optimization_store.f90 \
	tests/unit/test_optimization_store.f90

BACKEND_REGISTRY_TEST_SOURCES := \
	$(UNIT_MEMORY_F90) \
	src/runtime/mod_workspace.f90 \
	src/runtime/mod_runtime.f90 \
	src/backends/mod_backend_contract.f90 \
	src/backends/mod_backend_probe_support.f90 \
	src/backends/apple/mod_apple_bridge.f90 \
	src/backends/apple/mod_apple_capability.f90 \
	src/backends/cuda/mod_cuda_bridge.f90 \
	src/backends/cuda/mod_cuda_capability.f90 \
	src/backends/mod_backend_registry.f90 \
	tests/unit/test_backend_registry.f90 \
	$(APPLE_BRIDGE_OBJ) \
	$(CUDA_BRIDGE_OBJ)

RUNTIME_WORKSPACE_TEST_SOURCES := \
	$(UNIT_MEMORY_F90) \
	src/runtime/mod_workspace.f90 \
	src/runtime/mod_runtime.f90 \
	tests/unit/test_runtime_workspace.f90

SESSION_STAGING_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	src/runtime/mod_session.f90 \
	tests/unit/test_session_staging.f90

APPLE_PLANNER_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	src/backends/mod_backend_contract.f90 \
	src/backends/apple/mod_apple_planner.f90 \
	tests/unit/test_apple_planner.f90

APPLE_EXECUTOR_TEST_SOURCES := \
	$(UNIT_MEMORY_F90) \
	src/runtime/mod_workspace.f90 \
	src/model/mod_model_manifest.f90 \
	src/backends/apple/mod_apple_bridge.f90 \
	src/backends/apple/mod_apple_executor.f90 \
	tests/unit/test_apple_executor.f90 \
	$(APPLE_BRIDGE_OBJ)

CUDA_PLANNER_TEST_SOURCES := \
	$(UNIT_COMMON_F90) \
	src/backends/mod_backend_contract.f90 \
	src/backends/cuda/mod_cuda_planner.f90 \
	tests/unit/test_cuda_planner.f90

CUDA_EXECUTOR_TEST_SOURCES := \
	$(UNIT_MEMORY_F90) \
	src/runtime/mod_workspace.f90 \
	src/model/mod_model_manifest.f90 \
	src/backends/cuda/mod_cuda_bridge.f90 \
	src/backends/cuda/mod_cuda_executor.f90 \
	tests/unit/test_cuda_executor.f90 \
	$(CUDA_BRIDGE_OBJ)

define build_fortran_test
$1: $$($2) | $$(TEST_DIR)
	mkdir -p $$(TEST_DIR)/$3
	$$(FC) $$(FFLAGS) -J $$(TEST_DIR)/$3 -o $$@ \
		$$($2) $4
endef

.PHONY: all hooks format format-check check-local check-debug clean-debug test unit-tests contract-tests contract-smokes tool-tests clean

all: test

hooks:
	bash scripts/install-local-hooks.sh

format:
	./scripts/format-local.sh --all --write

format-check:
	./scripts/format-local.sh --all --check

check-local: format-check
	git diff --check
	$(MAKE) test

check-debug: format-check clean-debug
	git diff --check
	$(MAKE) BUILD_DIR=$(DEBUG_BUILD_DIR) FFLAGS='$(DEBUG_FFLAGS)' test

test: unit-tests contract-tests tool-tests

unit-tests: $(UNIT_BINS)
	@set -e; for test_bin in $(UNIT_BINS); do \
		echo "running $$test_bin"; \
		$$test_bin || exit $$?; \
	done

contract-tests: contract-smokes $(CONTRACT_BINS)
	@set -e; for test_bin in $(CONTRACT_BINS); do \
		echo "running $$test_bin"; \
		$$test_bin || exit $$?; \
	done

contract-smokes: $(CONTRACT_SMOKES)

tool-tests:
	@set -e; for test_script in $(TOOL_TESTS); do \
		echo "running $$test_script"; \
		BUILD_DIR=$(BUILD_DIR) python3 $$test_script || exit $$?; \
	done

clean-debug:
	rm -rf $(DEBUG_BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_DIR):
	mkdir -p $(TEST_DIR)

$(UNIT_BINS) $(CONTRACT_BINS) $(CONTRACT_SMOKES) $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ): $(BUILD_RECIPE_INPUTS)

ifeq ($(HAVE_NVCC),1)
CUDA_BRIDGE_SRC := src/backends/cuda/cuda_bridge.cu
CUDA_BRIDGE_LINK_LIBS := -lcudart -lstdc++
CUDA_TEST_CPPFLAGS := -DMIZU_CUDA_BRIDGE_STUB=0

$(CUDA_BRIDGE_OBJ): $(CUDA_BRIDGE_SRC) src/backends/cuda/cuda_bridge.h include/mizu.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
CUDA_BRIDGE_SRC := src/backends/cuda/cuda_bridge_stub.c
CUDA_BRIDGE_LINK_LIBS :=
CUDA_TEST_CPPFLAGS := -DMIZU_CUDA_BRIDGE_STUB=1

$(CUDA_BRIDGE_OBJ): $(CUDA_BRIDGE_SRC) src/backends/cuda/cuda_bridge.h include/mizu.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -Isrc/backends/cuda -c $< -o $@
endif

ifeq ($(UNAME_S),Darwin)
APPLE_BRIDGE_SRC := src/backends/apple/apple_bridge.m
APPLE_BRIDGE_LINK_LIBS := -framework Foundation -framework Metal

$(APPLE_BRIDGE_OBJ): $(APPLE_BRIDGE_SRC) src/backends/apple/apple_bridge.h \
	src/backends/apple/apple_bridge_common.h include/mizu.h | $(BUILD_DIR)
	$(OBJC) $(OBJCFLAGS) -Iinclude -Isrc/backends/apple -c $< -o $@
else
APPLE_BRIDGE_SRC := src/backends/apple/apple_bridge_stub.c
APPLE_BRIDGE_LINK_LIBS :=

$(APPLE_BRIDGE_OBJ): $(APPLE_BRIDGE_SRC) src/backends/apple/apple_bridge.h \
	src/backends/apple/apple_bridge_common.h include/mizu.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -Isrc/backends/apple -c $< -o $@
endif

$(eval $(call build_fortran_test,$(TEST_DIR)/test_model_manifest_loader,MODEL_MANIFEST_LOADER_TEST_SOURCES,loader_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_cache_keys,CACHE_KEYS_TEST_SOURCES,cache_keys_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_cache_store,CACHE_STORE_TEST_SOURCES,cache_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_plan_cache,PLAN_CACHE_TEST_SOURCES,plan_cache_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_weight_cache,WEIGHT_CACHE_TEST_SOURCES,weight_cache_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_session_cache,SESSION_CACHE_TEST_SOURCES,session_cache_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_mm_cache,MM_CACHE_TEST_SOURCES,mm_cache_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_optimization_store,OPTIMIZATION_STORE_TEST_SOURCES,optimization_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_backend_registry,BACKEND_REGISTRY_TEST_SOURCES,backend_registry_mods,$(APPLE_BRIDGE_LINK_LIBS) $(CUDA_BRIDGE_LINK_LIBS)))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_runtime_workspace,RUNTIME_WORKSPACE_TEST_SOURCES,runtime_workspace_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_session_staging,SESSION_STAGING_TEST_SOURCES,session_staging_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_apple_planner,APPLE_PLANNER_TEST_SOURCES,apple_planner_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_apple_executor,APPLE_EXECUTOR_TEST_SOURCES,apple_executor_mods,$(APPLE_BRIDGE_LINK_LIBS)))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_cuda_planner,CUDA_PLANNER_TEST_SOURCES,cuda_planner_mods))
$(eval $(call build_fortran_test,$(TEST_DIR)/test_cuda_executor,CUDA_EXECUTOR_TEST_SOURCES,cuda_executor_mods,$(CUDA_BRIDGE_LINK_LIBS)))

CONTRACT_OBJECT_CPPFLAGS :=
$(TEST_DIR)/test_cuda_artifacts.o: CONTRACT_OBJECT_CPPFLAGS := $(CUDA_TEST_CPPFLAGS)
$(TEST_DIR)/test_qwench_gguf_cuda_smoke.o: CONTRACT_OBJECT_CPPFLAGS := $(CUDA_TEST_CPPFLAGS)

$(TEST_DIR)/%.o: tests/contract/%.c include/mizu.h | $(TEST_DIR)
	$(CC) $(CFLAGS) $(CONTRACT_OBJECT_CPPFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_header_cpp_smoke.o: tests/contract/test_header_cpp_smoke.cpp include/mizu.h | $(TEST_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_opaque_handles: tests/contract/test_opaque_handles.c include/mizu.h | $(TEST_DIR)
	$(CC) $(CFLAGS) -Iinclude tests/contract/test_opaque_handles.c -o $@

$(TEST_DIR)/test_backend_availability: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_backend_availability.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/backend_availability_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/backend_availability_mods -I $(TEST_DIR)/backend_availability_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_backend_availability.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_backend_routing_contracts: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_backend_routing_contracts.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/backend_routing_contracts_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/backend_routing_contracts_mods -I $(TEST_DIR)/backend_routing_contracts_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_backend_routing_contracts.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_handle_lifecycle: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_handle_lifecycle.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/handle_lifecycle_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/handle_lifecycle_mods -I $(TEST_DIR)/handle_lifecycle_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_handle_lifecycle.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_handle_arena_capacity: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_handle_arena_capacity.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/handle_arena_capacity_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/handle_arena_capacity_mods -I $(TEST_DIR)/handle_arena_capacity_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_handle_arena_capacity.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_model_open_failures: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_model_open_failures.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/model_open_failures_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/model_open_failures_mods -I $(TEST_DIR)/model_open_failures_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_model_open_failures.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_modal_input_validation: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_modal_input_validation.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/modal_input_validation_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/modal_input_validation_mods -I $(TEST_DIR)/modal_input_validation_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_modal_input_validation.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_token_only_execution: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_token_only_execution.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/token_only_execution_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/token_only_execution_mods -I $(TEST_DIR)/token_only_execution_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_token_only_execution.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_decode_terminal_status: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_decode_terminal_status.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/decode_terminal_status_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/decode_terminal_status_mods -I $(TEST_DIR)/decode_terminal_status_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_decode_terminal_status.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_session_eviction: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_session_eviction.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/session_eviction_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/session_eviction_mods -I $(TEST_DIR)/session_eviction_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_session_eviction.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_session_state_guards: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_session_state_guards.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/session_state_guards_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/session_state_guards_mods -I $(TEST_DIR)/session_state_guards_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_session_state_guards.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_session_checkpoint_restore_failures: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_session_checkpoint_restore_failures.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/session_checkpoint_restore_failures_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/session_checkpoint_restore_failures_mods -I $(TEST_DIR)/session_checkpoint_restore_failures_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_session_checkpoint_restore_failures.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_runtime_last_error_propagation: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_runtime_last_error_propagation.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/runtime_last_error_propagation_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/runtime_last_error_propagation_mods -I $(TEST_DIR)/runtime_last_error_propagation_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_runtime_last_error_propagation.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_struct_sizes: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_struct_sizes.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/struct_sizes_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/struct_sizes_mods -I $(TEST_DIR)/struct_sizes_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_struct_sizes.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_cuda_artifacts: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_cuda_artifacts.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/cuda_contract_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cuda_contract_mods -I $(TEST_DIR)/cuda_contract_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_cuda_artifacts.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_qwench_gguf_cuda_smoke: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_qwench_gguf_cuda_smoke.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/qwench_gguf_cuda_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/qwench_gguf_cuda_mods -I $(TEST_DIR)/qwench_gguf_cuda_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_qwench_gguf_cuda_smoke.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_stage_reports: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_stage_reports.o $(CUDA_BRIDGE_OBJ) $(APPLE_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/contract_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/contract_mods -I $(TEST_DIR)/contract_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_stage_reports.o \
		$(APPLE_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_OBJ) \
		$(APPLE_BRIDGE_LINK_LIBS) \
		$(CUDA_BRIDGE_LINK_LIBS)

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_BUILD_DIR) ./*.mod
