#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mizu.h"

static int expect_status(const char *label, mizu_status_code_t actual, mizu_status_code_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected status %d, got %d\n", label, (int)expected, (int)actual);
        return 0;
    }
    return 1;
}

static int expect_true(const char *label, int condition) {
    if (!condition) {
        fprintf(stderr, "%s\n", label);
        return 0;
    }
    return 1;
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_runtime_t *runtime_reuse = NULL;
    mizu_model_t *failed_model = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *model_reuse = NULL;
    mizu_session_t *failed_session = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_reuse = NULL;
    mizu_status_code_t status;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    mizu_session_info_t session_info_reuse;
    mizu_model_info_t model_info;
    mizu_model_info_t model_info_reuse;
    size_t required_bytes = 0;

    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE=1\n");
        return 1;
    }
    if (setenv("MIZU_FORCE_APPLE_METAL_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_METAL_AVAILABLE=0\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        return 1;
    }
    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE=0\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
        return 1;
    }

    status = mizu_runtime_destroy(NULL);
    if (!expect_status("runtime destroy null", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(NULL);
    if (!expect_status("model close null", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(NULL);
    if (!expect_status("session close null", status, MIZU_STATUS_OK)) return 1;

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = NULL;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;
    runtime_config.exploration_budget = 0;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;
    model_info.struct_size = sizeof(model_info);
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy with live model", status, MIZU_STATUS_BUSY)) return 1;
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("model remains valid after busy runtime destroy", status, MIZU_STATUS_OK)) return 1;

    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.seed = 0;
    session_config.temperature = 0.0f;
    session_config.top_k = 0;
    session_config.top_p = 0.0f;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;
    session_info.struct_size = sizeof(session_info);
    status = mizu_model_close(model);
    if (!expect_status("model close with live session", status, MIZU_STATUS_BUSY)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session remains valid after busy model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_open(model, &session_config, &session_reuse);
    if (!expect_status("session reopen", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("closed session get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    status = mizu_session_close(session);
    if (!expect_status("double session close", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    session_info_reuse.struct_size = sizeof(session_info_reuse);
    status = mizu_session_get_info(session_reuse, &session_info_reuse);
    if (!expect_status("reopened session should remain valid", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session_reuse);
    if (!expect_status("reopened session close", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    failed_session = session_reuse;
    status = mizu_session_open(model, &session_config, &failed_session);
    if (!expect_status("session open should reject closed model", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed session open should clear output handle", failed_session == NULL)) return 1;
    status = mizu_model_open(runtime, &model_config, &model_reuse);
    if (!expect_status("model reopen", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("closed model get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("double model close", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    model_info_reuse.struct_size = sizeof(model_info_reuse);
    status = mizu_model_get_info(model_reuse, &model_info_reuse);
    if (!expect_status("reopened model should remain valid", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_reuse);
    if (!expect_status("reopened model close", status, MIZU_STATUS_OK)) return 1;

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;
    failed_model = model_reuse;
    status = mizu_model_open(runtime, &model_config, &failed_model);
    if (!expect_status("model open should reject destroyed runtime", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed model open should clear output handle", failed_model == NULL)) return 1;
    status = mizu_runtime_create(&runtime_config, &runtime_reuse);
    if (!expect_status("runtime recreate", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_copy_last_error(runtime, NULL, 0, &required_bytes);
    if (!expect_status("destroyed runtime copy last error", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("double runtime destroy", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    status = mizu_runtime_destroy(runtime_reuse);
    if (!expect_status("recreated runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_handle_lifecycle: PASS");
    return 0;
}
