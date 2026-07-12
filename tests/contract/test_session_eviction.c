#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mizu.h"

enum { PARKED_SESSION_COUNT = 17 };

static int expect_status(const char *label, mizu_status_code_t actual, mizu_status_code_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected status %d, got %d\n", label, (int)expected, (int)actual);
        return 0;
    }
    return 1;
}

int main(void) {
    const char *cache_root = "/tmp/mizu_session_eviction_contract";
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *sessions[PARKED_SESSION_COUNT] = {0};
    mizu_runtime_config_t runtime_config = {0};
    mizu_model_open_config_t model_config = {0};
    mizu_session_config_t session_config = {0};
    mizu_status_code_t status;
    int32_t token = 17;
    int index;

    if (system("rm -rf /tmp/mizu_session_eviction_contract") != 0) {
        fprintf(stderr, "failed to reset session-eviction cache root\n");
        return 1;
    }
    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0 ||
        setenv("MIZU_FORCE_APPLE_METAL_AVAILABLE", "0", 1) != 0 ||
        setenv("MIZU_FORCE_CUDA_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to configure deterministic backend availability\n");
        return 1;
    }

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = cache_root;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;

    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;

    for (index = 0; index < PARKED_SESSION_COUNT; ++index) {
        status = mizu_session_open(model, &session_config, &sessions[index]);
        if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;
        status = mizu_session_attach_tokens(sessions[index], &token, 1, MIZU_ATTACH_FLAG_NONE);
        if (!expect_status("attach token", status, MIZU_STATUS_OK)) return 1;
        status = mizu_session_prefill(sessions[index], NULL);
        if (!expect_status("prefill", status, MIZU_STATUS_OK)) return 1;
        status = mizu_session_park(sessions[index], NULL);
        if (!expect_status("park", status, MIZU_STATUS_OK)) return 1;
    }

    status = mizu_session_resume(sessions[0], NULL);
    if (!expect_status("oldest parked session should be evicted", status, MIZU_STATUS_SESSION_EVICTED)) return 1;
    status = mizu_session_resume(sessions[1], NULL);
    if (!expect_status("newer parked session should remain resumable", status, MIZU_STATUS_OK)) return 1;

    for (index = 0; index < PARKED_SESSION_COUNT; ++index) {
        status = mizu_session_close(sessions[index]);
        if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    }
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
    if (system("rm -rf /tmp/mizu_session_eviction_contract") != 0) {
        fprintf(stderr, "failed to clean session-eviction cache root\n");
        return 1;
    }

    puts("test_session_eviction: PASS");
    return 0;
}
