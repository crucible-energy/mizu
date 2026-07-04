#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int set_fixture_env(void) {
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
    if (setenv("MIZU_FORCE_SESSION_EVICTION", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_SESSION_EVICTION=1\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
        unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
        return 0;
    }
    return 1;
}

static void clear_fixture_env(void) {
    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
    unsetenv("MIZU_FORCE_SESSION_EVICTION");
}

static void init_runtime_config(mizu_runtime_config_t *config, const char *cache_root) {
    memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->cache_root_z = cache_root;
    config->optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;
    config->runtime_flags = MIZU_RUNTIME_FLAG_NONE;
}

static void init_model_config(mizu_model_open_config_t *config) {
    memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    config->allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    config->model_flags = MIZU_MODEL_FLAG_NONE;
}

static void init_session_config(mizu_session_config_t *config) {
    memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(*config);
    config->abi_version = mizu_get_abi_version();
    config->max_context_tokens = 4096;
    config->max_decode_tokens = 128;
    config->sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    config->session_flags = MIZU_SESSION_FLAG_NONE;
}

static int reset_cache_root(const char *cache_root) {
    char command[1024];

    if (snprintf(command, sizeof(command), "rm -rf %s && mkdir -p %s", cache_root, cache_root) < 0) {
        fprintf(stderr, "failed to build cache root reset command\n");
        return 0;
    }
    return expect_true("cache root reset should succeed", system(command) == 0);
}

int main(void) {
    const char *cache_root = "/tmp/mizu_session_eviction";
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int32_t tokens[3] = {1, 2, 3};
    mizu_execution_report_t prefill_reports[1];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t resume_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    size_t required_bytes;
    char error_buffer[256];

    if (!set_fixture_env()) return 1;
    if (!reset_cache_root(cache_root)) {
        clear_fixture_env();
        return 1;
    }

    init_runtime_config(&runtime_config, cache_root);
    init_model_config(&model_config);
    init_session_config(&session_config);

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens", status, MIZU_STATUS_OK)) return 1;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(park_reports, 0, sizeof(park_reports));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    memset(&park_buffer, 0, sizeof(park_buffer));
    memset(&resume_buffer, 0, sizeof(resume_buffer));
    memset(&session_info, 0, sizeof(session_info));
    memset(error_buffer, 0, sizeof(error_buffer));

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill", status, MIZU_STATUS_OK)) return 1;

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park", status, MIZU_STATUS_OK)) return 1;

    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after injected eviction", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("evicted session should remain parked",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) return 1;
    if (!expect_true("evicted session should clear live context flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) == 0)) return 1;
    if (!expect_true("evicted session should clear kv token count", session_info.kv_token_count == 0)) return 1;

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume after forced eviction", status, MIZU_STATUS_SESSION_EVICTED)) return 1;

    required_bytes = 0;
    status = mizu_runtime_copy_last_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes);
    if (!expect_status("copy last error after eviction", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("eviction should publish non-empty runtime error", required_bytes > 1)) return 1;
    if (!expect_true("eviction error should mention parked state eviction",
                     strstr(error_buffer, "session parked state was evicted") != NULL)) return 1;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("repeat resume after forced eviction", status, MIZU_STATUS_SESSION_EVICTED)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    clear_fixture_env();
    puts("test_session_eviction: PASS");
    return 0;
}
