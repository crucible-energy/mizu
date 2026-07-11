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
    mizu_runtime_t *bad_runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *bad_model = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *bad_session = NULL;
    mizu_status_code_t status;
    mizu_runtime_config_t runtime_config;
    mizu_runtime_config_t bad_runtime_config;
    mizu_model_open_config_t model_config;
    mizu_model_open_config_t bad_model_config;
    mizu_session_config_t session_config;
    mizu_session_config_t bad_session_config;
    mizu_model_info_t model_info;
    mizu_session_info_t session_info;
    mizu_execution_report_t report;
    mizu_modal_input_desc_t modal_input;
    mizu_report_buffer_t report_buffer;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_output_buffer_t output_buffer;
    mizu_execution_report_t report_storage[2];
    unsigned char expected_model_info_bytes[sizeof(mizu_model_info_t)];
    unsigned char expected_session_info_bytes[sizeof(mizu_session_info_t)];
    unsigned char expected_report_bytes[sizeof(mizu_execution_report_t)];
    unsigned char expected_report_buffer_bytes[sizeof(mizu_report_buffer_t)];
    unsigned char expected_decode_result_bytes[sizeof(mizu_decode_result_t)];
    unsigned char expected_output_buffer_bytes[sizeof(mizu_output_buffer_t)];
    unsigned char expected_report_storage_bytes[sizeof(mizu_execution_report_t) * 2];
    const int32_t prefill_tokens[] = { 17 };
    int32_t decode_token = 7;

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

    memset(&runtime_config, 0, sizeof(runtime_config));
    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, NULL);
    if (!expect_status("runtime create should reject null output", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;

    bad_runtime_config = runtime_config;
    bad_runtime_config.struct_size = sizeof(bad_runtime_config) - 1;
    status = mizu_runtime_create(&bad_runtime_config, &bad_runtime);
    if (!expect_status("runtime create should reject short struct", status, MIZU_STATUS_ABI_MISMATCH)) return 1;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    memset(&model_config, 0, sizeof(model_config));
    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, NULL);
    if (!expect_status("model open should reject null output", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;

    bad_model_config = model_config;
    bad_model_config.struct_size = sizeof(bad_model_config) - 1;
    status = mizu_model_open(runtime, &bad_model_config, &bad_model);
    if (!expect_status("model open should reject short struct", status, MIZU_STATUS_ABI_MISMATCH)) return 1;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;

    memset(&session_config, 0, sizeof(session_config));
    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_model_get_info(model, NULL);
    if (!expect_status("model info should reject null output", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;

    status = mizu_session_open(model, &session_config, NULL);
    if (!expect_status("session open should reject null output", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;

    bad_session_config = session_config;
    bad_session_config.struct_size = sizeof(bad_session_config) - 1;
    status = mizu_session_open(model, &bad_session_config, &bad_session);
    if (!expect_status("session open should reject short struct", status, MIZU_STATUS_ABI_MISMATCH)) return 1;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_get_info(session, NULL);
    if (!expect_status("session info should reject null output", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;

    memset(&model_info, 0xA5, sizeof(model_info));
    model_info.struct_size = sizeof(model_info) - 1;
    memcpy(expected_model_info_bytes, &model_info, sizeof(model_info));
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("model info should reject short struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("model info short-struct failure should preserve caller bytes",
                     memcmp(&model_info, expected_model_info_bytes, sizeof(model_info)) == 0)) return 1;

    memset(&report, 0xA5, sizeof(report));
    report.struct_size = sizeof(report) - 1;
    memcpy(expected_report_bytes, &report, sizeof(report));
    status = mizu_model_get_last_report(model, &report);
    if (!expect_status("model report should reject short struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("model report short-struct failure should preserve caller bytes",
                     memcmp(&report, expected_report_bytes, sizeof(report)) == 0)) return 1;

    memset(&report, 0, sizeof(report));
    report.struct_size = sizeof(report);
    status = mizu_model_get_last_report(model, &report);
    if (!expect_status("model report should accept full struct", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("model report struct_size should survive reuse", report.struct_size == sizeof(report))) return 1;
    status = mizu_model_get_last_report(model, &report);
    if (!expect_status("model report should allow same struct reuse", status, MIZU_STATUS_OK)) return 1;

    memset(&session_info, 0xA5, sizeof(session_info));
    session_info.struct_size = sizeof(session_info) - 1;
    memcpy(expected_session_info_bytes, &session_info, sizeof(session_info));
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info should reject short struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("session info short-struct failure should preserve caller bytes",
                     memcmp(&session_info, expected_session_info_bytes, sizeof(session_info)) == 0)) return 1;

    memset(&modal_input, 0, sizeof(modal_input));
    modal_input.struct_size = sizeof(modal_input) - 1;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("modal input should reject short struct", status, MIZU_STATUS_ABI_MISMATCH)) return 1;

    memset(report_storage, 0xA5, sizeof(report_storage));
    memset(&report_buffer, 0xA5, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer) - 1;
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 2;
    memcpy(expected_report_buffer_bytes, &report_buffer, sizeof(report_buffer));
    memcpy(expected_report_storage_bytes, report_storage, sizeof(report_storage));
    status = mizu_session_prefill(session, &report_buffer);
    if (!expect_status("prefill should reject short report buffer struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("prefill short report-buffer failure should preserve caller bytes",
                     memcmp(&report_buffer, expected_report_buffer_bytes, sizeof(report_buffer)) == 0)) return 1;
    if (!expect_true("prefill short report-buffer failure should preserve staged report payload bytes",
                     memcmp(report_storage, expected_report_storage_bytes, sizeof(report_storage)) == 0)) return 1;

    status = mizu_session_attach_tokens(session, prefill_tokens, 1, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens for prefill", status, MIZU_STATUS_OK)) return 1;

    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 0;
    status = mizu_session_prefill(session, &report_buffer);
    if (!expect_status("prefill should reject undersized report buffer", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("prefill should report required report count", report_buffer.report_count == 1)) return 1;

    memset(report_storage, 0, sizeof(report_storage));
    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 2;
    status = mizu_session_prefill(session, &report_buffer);
    if (!expect_status("prefill should accept zeroed report entries", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("prefill should report one stage", report_buffer.report_count == 1)) return 1;
    if (!expect_true("prefill should stamp buffered report struct_size", report_storage[0].struct_size == sizeof(report_storage[0]))) return 1;
    status = mizu_session_get_last_report(session, &report_storage[0]);
    if (!expect_status("session report should allow buffered report reuse", status, MIZU_STATUS_OK)) return 1;

    memset(report_storage, 0, sizeof(report_storage));
    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 2;
    report_buffer.report_count = 9;
    report_storage[0].struct_size = sizeof(report_storage[0]);
    report_storage[0].stage_kind = MIZU_STAGE_PROJECTOR;
    report_storage[0].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[0].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[0].plan_id = 9;
    report_storage[0].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[0].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[0].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[0].cache_flags = UINT64_C(9);
    report_storage[0].elapsed_us = UINT64_C(9);
    report_storage[1].struct_size = sizeof(report_storage[1]);
    report_storage[1].stage_kind = MIZU_STAGE_DECODE;
    report_storage[1].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[1].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[1].plan_id = 9;
    report_storage[1].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[1].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[1].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[1].cache_flags = UINT64_C(9);
    report_storage[1].elapsed_us = UINT64_C(9);

    memset(&decode_result, 0xA5, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decode_token;
    decode_result.token_capacity = 1;
    decode_result.token_count = 9;
    decode_result.stop_reason = MIZU_STOP_REASON_STOP_SEQUENCE;
    decode_result.result_flags = UINT64_C(9);

    status = mizu_session_decode_step(session, NULL, &decode_result, &report_buffer);
    if (!expect_status("decode should reject null options pointer", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("decode null options failure should preserve result inputs",
                     decode_result.struct_size == sizeof(decode_result) &&
                     decode_result.token_buffer == &decode_token &&
                     decode_result.token_capacity == 1)) {
        return 1;
    }
    if (!expect_true("decode null options failure should clear result outputs",
                     decode_result.token_count == 0 &&
                     decode_result.stop_reason == MIZU_STOP_REASON_NONE &&
                     decode_result.result_flags == 0)) return 1;
    if (!expect_true("decode null options failure should preserve report-buffer inputs",
                     report_buffer.struct_size == sizeof(report_buffer) &&
                     report_buffer.reports == report_storage &&
                     report_buffer.report_capacity == 2)) return 1;
    if (!expect_true("decode null options failure should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("decode null options failure should clear both report payloads",
                     report_storage[0].struct_size == sizeof(report_storage[0]) &&
                     report_storage[0].stage_kind == 0 &&
                     report_storage[0].backend_family == 0 &&
                     report_storage[0].execution_route == 0 &&
                     report_storage[0].plan_id == 0 &&
                     report_storage[0].selection_mode == 0 &&
                     report_storage[0].cold_state == 0 &&
                     report_storage[0].fallback_reason == 0 &&
                     report_storage[0].cache_flags == 0 &&
                     report_storage[0].elapsed_us == 0 &&
                     report_storage[1].struct_size == sizeof(report_storage[1]) &&
                     report_storage[1].stage_kind == 0 &&
                     report_storage[1].backend_family == 0 &&
                     report_storage[1].execution_route == 0 &&
                     report_storage[1].plan_id == 0 &&
                     report_storage[1].selection_mode == 0 &&
                     report_storage[1].cold_state == 0 &&
                     report_storage[1].fallback_reason == 0 &&
                     report_storage[1].cache_flags == 0 &&
                     report_storage[1].elapsed_us == 0)) return 1;

    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.struct_size = sizeof(decode_options) - 1;
    decode_options.token_budget = 1;

    memset(report_storage, 0, sizeof(report_storage));
    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 2;
    report_buffer.report_count = 9;
    report_storage[0].struct_size = sizeof(report_storage[0]);
    report_storage[0].stage_kind = MIZU_STAGE_PROJECTOR;
    report_storage[0].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[0].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[0].plan_id = 9;
    report_storage[0].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[0].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[0].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[0].cache_flags = UINT64_C(9);
    report_storage[0].elapsed_us = UINT64_C(9);
    report_storage[1].struct_size = sizeof(report_storage[1]);
    report_storage[1].stage_kind = MIZU_STAGE_DECODE;
    report_storage[1].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[1].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[1].plan_id = 9;
    report_storage[1].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[1].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[1].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[1].cache_flags = UINT64_C(9);
    report_storage[1].elapsed_us = UINT64_C(9);

    memset(&decode_result, 0xA5, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decode_token;
    decode_result.token_capacity = 1;
    decode_result.token_count = 9;
    decode_result.stop_reason = MIZU_STOP_REASON_STOP_SEQUENCE;
    decode_result.result_flags = UINT64_C(9);
    memcpy(expected_decode_result_bytes, &decode_result, sizeof(decode_result));

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &report_buffer);
    if (!expect_status("decode should reject short options struct", status, MIZU_STATUS_ABI_MISMATCH)) return 1;
    if (!expect_true("decode short options failure should preserve result inputs",
                     decode_result.struct_size == sizeof(decode_result) &&
                     decode_result.token_buffer == &decode_token &&
                     decode_result.token_capacity == 1)) {
        return 1;
    }
    if (!expect_true("decode short options failure should clear result outputs",
                     decode_result.token_count == 0 &&
                     decode_result.stop_reason == MIZU_STOP_REASON_NONE &&
                     decode_result.result_flags == 0)) return 1;
    if (!expect_true("decode short options failure should preserve report-buffer inputs",
                     report_buffer.struct_size == sizeof(report_buffer) &&
                     report_buffer.reports == report_storage &&
                     report_buffer.report_capacity == 2)) return 1;
    if (!expect_true("decode short options failure should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("decode short options failure should clear both report payloads",
                     report_storage[0].struct_size == sizeof(report_storage[0]) &&
                     report_storage[0].stage_kind == 0 &&
                     report_storage[0].backend_family == 0 &&
                     report_storage[0].execution_route == 0 &&
                     report_storage[0].plan_id == 0 &&
                     report_storage[0].selection_mode == 0 &&
                     report_storage[0].cold_state == 0 &&
                     report_storage[0].fallback_reason == 0 &&
                     report_storage[0].cache_flags == 0 &&
                     report_storage[0].elapsed_us == 0 &&
                     report_storage[1].struct_size == sizeof(report_storage[1]) &&
                     report_storage[1].stage_kind == 0 &&
                     report_storage[1].backend_family == 0 &&
                     report_storage[1].execution_route == 0 &&
                     report_storage[1].plan_id == 0 &&
                     report_storage[1].selection_mode == 0 &&
                     report_storage[1].cold_state == 0 &&
                     report_storage[1].fallback_reason == 0 &&
                     report_storage[1].cache_flags == 0 &&
                     report_storage[1].elapsed_us == 0)) return 1;

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 0;
    memset(report_storage, 0, sizeof(report_storage));
    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 2;
    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_capacity = 1;
    decode_result.token_count = 9;
    decode_result.stop_reason = MIZU_STOP_REASON_STOP_SEQUENCE;
    decode_result.result_flags = UINT64_C(9);
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &report_buffer);
    if (!expect_status("decode should reject zero token budget", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed decode should clear output token count", decode_result.token_count == 0)) return 1;
    if (!expect_true("failed decode should clear output stop reason", decode_result.stop_reason == MIZU_STOP_REASON_NONE)) {
        return 1;
    }
    if (!expect_true("failed decode should clear output flags", decode_result.result_flags == 0)) return 1;
    if (!expect_true("failed decode should not publish reports", report_buffer.report_count == 0)) return 1;

    memset(&decode_result, 0xA5, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result) - 1;
    decode_result.token_capacity = 1;
    memcpy(expected_decode_result_bytes, &decode_result, sizeof(decode_result));
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("decode should reject short result struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("decode short result-struct failure should preserve caller bytes",
                     memcmp(&decode_result, expected_decode_result_bytes, sizeof(decode_result)) == 0)) return 1;

    decode_options.token_budget = 1;
    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("decode should reject undersized token buffer", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("decode should report required token count", decode_result.token_count == 1)) return 1;

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after failed decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("failed decode should not advance kv tokens", session_info.kv_token_count == 1)) return 1;

    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.bytes_written = 9;
    output_buffer.output_flags = UINT64_C(9);
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("failed decode should not publish output", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!expect_true("failed decode should clear output bytes", output_buffer.bytes_written == 0)) return 1;
    if (!expect_true("failed decode should clear output flags", output_buffer.output_flags == 0)) return 1;

    {
        int32_t decoded_token = 0;
        memset(&decode_result, 0, sizeof(decode_result));
        decode_result.struct_size = sizeof(decode_result);
        decode_result.token_buffer = &decoded_token;
        decode_result.token_capacity = 1;
        status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
        if (!expect_status("decode should succeed with one token slot", status, MIZU_STATUS_OK)) return 1;
    }

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after successful decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("successful decode should advance kv tokens", session_info.kv_token_count == 2)) return 1;

    memset(&output_buffer, 0xA5, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer) - 1;
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    memcpy(expected_output_buffer_bytes, &output_buffer, sizeof(output_buffer));
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output should reject short struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("read output short-struct failure should preserve caller bytes",
                     memcmp(&output_buffer, expected_output_buffer_bytes, sizeof(output_buffer)) == 0)) return 1;

    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output should reject undersized byte buffer", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("read output should report required byte count", output_buffer.bytes_written == sizeof(int32_t))) return 1;

    memset(&report, 0xA5, sizeof(report));
    report.struct_size = sizeof(report) - 1;
    memcpy(expected_report_bytes, &report, sizeof(report));
    status = mizu_session_get_last_report(session, &report);
    if (!expect_status("session report should reject short struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("session report short-struct failure should preserve caller bytes",
                     memcmp(&report, expected_report_bytes, sizeof(report)) == 0)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_struct_sizes: PASS");
    return 0;
}
