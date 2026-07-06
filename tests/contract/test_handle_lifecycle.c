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

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_runtime_t *runtime_reuse = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *model_reuse = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_reuse = NULL;
    mizu_session_t *stale_session_after_model_close = (mizu_session_t *)(uintptr_t)0x2;
    mizu_status_code_t status;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    mizu_session_info_t session_info_reuse;
    mizu_model_info_t model_info;
    mizu_model_info_t model_info_reuse;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_output_buffer_t output_buffer;
    mizu_execution_report_t session_report;
    mizu_execution_report_t model_report;
    mizu_execution_report_t report_storage[1];
    mizu_report_buffer_t report_buffer;
    int32_t decode_token = 4321;
    char error_buffer[16] = "keep";
    size_t required_bytes = 0;
    mizu_model_t *stale_model_after_runtime_destroy = (mizu_model_t *)(uintptr_t)0x1;

    status = mizu_session_close(NULL);
    if (!expect_status("null session close should be a no-op", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(NULL);
    if (!expect_status("null model close should be a no-op", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(NULL);
    if (!expect_status("null runtime destroy should be a no-op", status, MIZU_STATUS_OK)) return 1;

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

    status = mizu_model_close(model);
    if (!expect_status("model close with live session", status, MIZU_STATUS_BUSY)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy with live model", status, MIZU_STATUS_BUSY)) return 1;

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    session_info.session_state_flags = 444;
    session_info.kv_token_count = 555;
    session_info.staged_token_count = 666;
    session_info.staged_modal_count = 777;
    memset(&session_report, 0, sizeof(session_report));
    session_report.struct_size = sizeof(session_report);
    session_report.stage_kind = 123;
    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_open(model, &session_config, &session_reuse);
    if (!expect_status("session reopen", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("closed session get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session info should leave caller output untouched",
                     session_info.struct_size == sizeof(session_info) &&
                     session_info.session_state_flags == 444 &&
                     session_info.kv_token_count == 555 &&
                     session_info.staged_token_count == 666 &&
                     session_info.staged_modal_count == 777)) return 1;
    if (!expect_status("closed session report", mizu_session_get_last_report(session, &session_report), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session report should leave caller output untouched",
                     session_report.stage_kind == 123 && session_report.struct_size == sizeof(session_report))) return 1;

    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 1;
    report_buffer.report_count = 2468;
    if (!expect_status("closed session prefill", mizu_session_prefill(session, &report_buffer), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session prefill should leave report buffer untouched",
                     report_buffer.struct_size == sizeof(report_buffer) &&
                     report_buffer.report_capacity == 1 &&
                     report_buffer.report_count == 2468 &&
                     report_buffer.reports == report_storage)) return 1;
    if (!expect_status("closed session park", mizu_session_park(session, &report_buffer), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session park should leave report buffer untouched",
                     report_buffer.report_count == 2468 && report_buffer.reports == report_storage)) return 1;
    if (!expect_status("closed session resume", mizu_session_resume(session, &report_buffer), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session resume should leave report buffer untouched",
                     report_buffer.report_count == 2468 && report_buffer.reports == report_storage)) return 1;

    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;

    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decode_token;
    decode_result.token_capacity = 1;
    decode_result.token_count = 9753;
    decode_result.stop_reason = MIZU_STOP_REASON_TOKEN_BUDGET;
    decode_result.result_flags = 8642;

    if (!expect_status("closed session decode", mizu_session_decode_step(session, &decode_options, &decode_result, &report_buffer), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session decode should leave result output untouched",
                     decode_result.struct_size == sizeof(decode_result) &&
                     decode_result.token_buffer == &decode_token &&
                     decode_result.token_capacity == 1 &&
                     decode_result.token_count == 9753 &&
                     decode_result.stop_reason == MIZU_STOP_REASON_TOKEN_BUDGET &&
                     decode_result.result_flags == 8642)) return 1;
    if (!expect_true("closed session decode should leave report buffer untouched",
                     report_buffer.report_count == 2468 && report_buffer.reports == report_storage)) return 1;

    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = &decode_token;
    output_buffer.byte_capacity = sizeof(decode_token);
    output_buffer.bytes_written = 1357;
    output_buffer.output_flags = 97531;
    if (!expect_status("closed session read output", mizu_session_read_output(session, &output_buffer), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session read output should leave caller buffer untouched",
                     output_buffer.struct_size == sizeof(output_buffer) &&
                     output_buffer.output_kind == MIZU_OUTPUT_KIND_TOKEN_IDS &&
                     output_buffer.data == &decode_token &&
                     output_buffer.byte_capacity == sizeof(decode_token) &&
                     output_buffer.bytes_written == 1357 &&
                     output_buffer.output_flags == 97531)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("double session close", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    session_info_reuse.struct_size = sizeof(session_info_reuse);
    status = mizu_session_get_info(session_reuse, &session_info_reuse);
    if (!expect_status("reopened session should remain valid", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session_reuse);
    if (!expect_status("reopened session close", status, MIZU_STATUS_OK)) return 1;

    memset(&model_info, 0, sizeof(model_info));
    model_info.struct_size = sizeof(model_info);
    model_info.allowed_backend_mask = 888;
    model_info.model_features = 999;
    model_info.projector_slot_count = 111;
    memset(&model_report, 0, sizeof(model_report));
    model_report.struct_size = sizeof(model_report);
    model_report.stage_kind = 321;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_open(model, &session_config, &stale_session_after_model_close);
    if (!expect_status("closed model session open", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed model session open should clear caller output",
                     stale_session_after_model_close == NULL)) return 1;
    status = mizu_model_open(runtime, &model_config, &model_reuse);
    if (!expect_status("model reopen", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("closed model get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed model info should leave caller output untouched",
                     model_info.struct_size == sizeof(model_info) &&
                     model_info.allowed_backend_mask == 888 &&
                     model_info.model_features == 999 &&
                     model_info.projector_slot_count == 111)) return 1;
    if (!expect_status("closed model report", mizu_model_get_last_report(model, &model_report), MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed model report should leave caller output untouched",
                     model_report.stage_kind == 321 && model_report.struct_size == sizeof(model_report))) return 1;
    status = mizu_model_close(model);
    if (!expect_status("double model close", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    model_info_reuse.struct_size = sizeof(model_info_reuse);
    status = mizu_model_get_info(model_reuse, &model_info_reuse);
    if (!expect_status("reopened model should remain valid", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_reuse);
    if (!expect_status("reopened model close", status, MIZU_STATUS_OK)) return 1;

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime, &model_config, &stale_model_after_runtime_destroy);
    if (!expect_status("destroyed runtime model open", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("destroyed runtime model open should clear caller output",
                     stale_model_after_runtime_destroy == NULL)) return 1;
    status = mizu_runtime_create(&runtime_config, &runtime_reuse);
    if (!expect_status("runtime recreate", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_copy_last_error(runtime, NULL, 0, &required_bytes);
    if (!expect_status("destroyed runtime copy last error", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    required_bytes = 77;
    status = mizu_runtime_copy_last_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes);
    if (!expect_status("destroyed runtime copy last error with outputs", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("destroyed runtime copy last error should preserve required-bytes output", required_bytes == 77)) return 1;
    if (!expect_true("destroyed runtime copy last error should preserve error buffer contents",
                     strcmp(error_buffer, "keep") == 0)) return 1;
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
