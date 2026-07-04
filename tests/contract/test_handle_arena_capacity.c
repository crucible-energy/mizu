#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "mizu.h"

enum {
    RETIRED_HANDLE_ARENA_CAPACITY = 4096
};

static mizu_runtime_config_t runtime_config;
static mizu_model_open_config_t model_config;
static mizu_session_config_t session_config;

typedef int (*isolated_test_fn_t)(void);

static int expect_status(const char *label, mizu_status_code_t actual, mizu_status_code_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected status %d, got %d\n", label, (int)expected, (int)actual);
        return 0;
    }
    return 1;
}

static int expect_status_at_iteration(
    const char *label,
    int64_t iteration,
    mizu_status_code_t actual,
    mizu_status_code_t expected
) {
    if (actual != expected) {
        fprintf(
            stderr,
            "%s at iteration %" PRId64 ": expected status %d, got %d\n",
            label,
            iteration,
            (int)expected,
            (int)actual
        );
        return 0;
    }
    return 1;
}

static int set_fixture_backend_env(void) {
    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE=1\n");
        return 0;
    }
    if (setenv("MIZU_FORCE_APPLE_METAL_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_METAL_AVAILABLE=0\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        return 0;
    }
    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE=0\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
        return 0;
    }
    return 1;
}

static void clear_fixture_backend_env(void) {
    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
}

static void init_runtime_config(mizu_runtime_config_t *config) {
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->cache_root_z = NULL;
    config->optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;
    config->exploration_budget = 0;
    config->runtime_flags = MIZU_RUNTIME_FLAG_NONE;
}

static void init_model_config(mizu_model_open_config_t *config) {
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    config->allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    config->model_flags = MIZU_MODEL_FLAG_NONE;
}

static void init_session_config(mizu_session_config_t *config) {
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->max_context_tokens = 4096;
    config->max_decode_tokens = 128;
    config->sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    config->seed = 0;
    config->temperature = 0.0f;
    config->top_k = 0;
    config->top_p = 0.0f;
    config->session_flags = MIZU_SESSION_FLAG_NONE;
}

static int run_isolated_test(const char *label, isolated_test_fn_t test_fn) {
    pid_t child_pid;
    int wait_status;

    child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        return 0;
    }
    if (child_pid == 0) {
        exit(test_fn() ? 0 : 1);
    }
    if (waitpid(child_pid, &wait_status, 0) < 0) {
        perror("waitpid");
        return 0;
    }
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(stderr, "%s subprocess failed\n", label);
        return 0;
    }
    return 1;
}

static int test_session_arena_capacity(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int64_t iteration;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("session capacity runtime create", status, MIZU_STATUS_OK)) return 0;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("session capacity model open", status, MIZU_STATUS_OK)) return 0;

    for (iteration = 0; iteration < RETIRED_HANDLE_ARENA_CAPACITY; ++iteration) {
        status = mizu_session_open(model, &session_config, &session);
        if (!expect_status_at_iteration("session open", iteration, status, MIZU_STATUS_OK)) return 0;
        status = mizu_session_close(session);
        if (!expect_status_at_iteration("session close", iteration, status, MIZU_STATUS_OK)) return 0;
    }

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session arena exhausted", status, MIZU_STATUS_BUSY)) return 0;

    status = mizu_model_close(model);
    if (!expect_status("session capacity model close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("session capacity runtime destroy", status, MIZU_STATUS_OK)) return 0;

    return 1;
}

static int test_model_arena_capacity(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_status_code_t status;
    int64_t iteration;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("model capacity runtime create", status, MIZU_STATUS_OK)) return 0;

    for (iteration = 0; iteration < RETIRED_HANDLE_ARENA_CAPACITY; ++iteration) {
        status = mizu_model_open(runtime, &model_config, &model);
        if (!expect_status_at_iteration("model open", iteration, status, MIZU_STATUS_OK)) return 0;
        status = mizu_model_close(model);
        if (!expect_status_at_iteration("model close", iteration, status, MIZU_STATUS_OK)) return 0;
    }

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model arena exhausted", status, MIZU_STATUS_BUSY)) return 0;

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("model capacity runtime destroy", status, MIZU_STATUS_OK)) return 0;

    return 1;
}

static int test_runtime_arena_capacity(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_status_code_t status;
    int64_t iteration;

    for (iteration = 0; iteration < RETIRED_HANDLE_ARENA_CAPACITY; ++iteration) {
        status = mizu_runtime_create(&runtime_config, &runtime);
        if (!expect_status_at_iteration("runtime create", iteration, status, MIZU_STATUS_OK)) return 0;
        status = mizu_runtime_destroy(runtime);
        if (!expect_status_at_iteration("runtime destroy", iteration, status, MIZU_STATUS_OK)) return 0;
    }

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime arena exhausted", status, MIZU_STATUS_BUSY)) return 0;

    return 1;
}

int main(void) {
    if (!set_fixture_backend_env()) return 1;

    init_runtime_config(&runtime_config);
    init_model_config(&model_config);
    init_session_config(&session_config);

    if (!run_isolated_test("session arena capacity", test_session_arena_capacity)) {
        clear_fixture_backend_env();
        return 1;
    }
    if (!run_isolated_test("model arena capacity", test_model_arena_capacity)) {
        clear_fixture_backend_env();
        return 1;
    }
    if (!run_isolated_test("runtime arena capacity", test_runtime_arena_capacity)) {
        clear_fixture_backend_env();
        return 1;
    }

    clear_fixture_backend_env();
    puts("test_handle_arena_capacity: PASS");
    return 0;
}
