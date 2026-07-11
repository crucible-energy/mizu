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
    mizu_session_config_t bad_session_config;
    mizu_session_info_t session_info;
    mizu_session_info_t session_info_reuse;
    mizu_model_info_t model_info;
    mizu_model_info_t model_info_reuse;
    mizu_modal_input_desc_t modal_input;
    mizu_execution_report_t model_report;
    mizu_execution_report_t session_report;
    mizu_report_buffer_t report_buffer;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_output_buffer_t output_buffer;
    mizu_execution_report_t report_storage[2];
    int32_t decoded_token = 7;
    int32_t output_token = 7;
    uint8_t image_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    char error_buffer[8] = "stale";
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
    status = mizu_session_clear_pending_inputs(NULL);
    if (!expect_status("clear pending inputs null", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    required_bytes = 9;
    memset(error_buffer, 'X', sizeof(error_buffer));
    status = mizu_runtime_copy_last_error(NULL, error_buffer, sizeof(error_buffer), &required_bytes);
    if (!expect_status("copy last error null runtime", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("copy last error null runtime should clear required size", required_bytes == 0)) return 1;
    if (!expect_true("copy last error null runtime should clear error buffer", error_buffer[0] == '\0')) return 1;

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

    failed_model = (mizu_model_t *)(uintptr_t)1;
    status = mizu_model_open(NULL, &model_config, &failed_model);
    if (!expect_status("model open should reject null runtime handle", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed null-runtime model open should clear output handle", failed_model == NULL)) return 1;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;
    model_info.struct_size = sizeof(model_info);
    model_info.model_family = MIZU_MODEL_FAMILY_GEMMA4;
    model_info.allowed_backend_mask = UINT64_C(9);
    model_info.model_features = UINT64_C(9);
    model_info.projector_slot_count = 9;
    model_info.reserved_u32 = 9;
    status = mizu_model_get_info(NULL, &model_info);
    if (!expect_status("null model get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("null model get info should preserve struct size", model_info.struct_size == sizeof(model_info))) {
        return 1;
    }
    if (!expect_true("null model get info should clear outputs",
                     model_info.model_family == 0 &&
                     model_info.allowed_backend_mask == 0 &&
                     model_info.model_features == 0 &&
                     model_info.projector_slot_count == 0 &&
                     model_info.reserved_u32 == 0)) return 1;
    model_report.struct_size = sizeof(model_report);
    model_report.stage_kind = MIZU_STAGE_MODEL_LOAD;
    model_report.backend_family = MIZU_BACKEND_FAMILY_APPLE;
    model_report.execution_route = MIZU_EXEC_ROUTE_ANE;
    model_report.plan_id = 9;
    model_report.selection_mode = MIZU_SELECTION_MODE_DIRECT;
    model_report.cold_state = MIZU_COLD_STATE_WARM;
    model_report.fallback_reason = MIZU_FALLBACK_REASON_NONE;
    model_report.cache_flags = UINT64_C(9);
    model_report.elapsed_us = UINT64_C(9);
    status = mizu_model_get_last_report(NULL, &model_report);
    if (!expect_status("null model get last report", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("null model report should preserve struct size", model_report.struct_size == sizeof(model_report))) {
        return 1;
    }
    if (!expect_true("null model report should clear outputs",
                     model_report.stage_kind == 0 &&
                     model_report.backend_family == 0 &&
                     model_report.execution_route == 0 &&
                     model_report.plan_id == 0 &&
                     model_report.selection_mode == 0 &&
                     model_report.cold_state == 0 &&
                     model_report.fallback_reason == 0 &&
                     model_report.cache_flags == 0 &&
                     model_report.elapsed_us == 0)) return 1;
    model_info.struct_size = sizeof(model_info);
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy with live model", status, MIZU_STATUS_BUSY)) return 1;
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("model remains valid after busy runtime destroy", status, MIZU_STATUS_OK)) return 1;
    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("model report after open", status, MIZU_STATUS_OK)) return 1;

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

    failed_session = (mizu_session_t *)(uintptr_t)1;
    status = mizu_session_open(NULL, &session_config, &failed_session);
    if (!expect_status("session open should reject null model handle", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed null-model session open should clear output handle", failed_session == NULL)) return 1;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;
    session_info.struct_size = sizeof(session_info);
    session_info.session_state_flags = UINT64_C(9);
    session_info.kv_token_count = 9;
    session_info.staged_token_count = 9;
    session_info.staged_modal_count = 9;
    session_info.reserved_u32 = 9;
    status = mizu_session_get_info(NULL, &session_info);
    if (!expect_status("null session get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("null session get info should preserve struct size", session_info.struct_size == sizeof(session_info))) {
        return 1;
    }
    if (!expect_true("null session get info should clear outputs",
                     session_info.session_state_flags == 0 &&
                     session_info.kv_token_count == 0 &&
                     session_info.staged_token_count == 0 &&
                     session_info.staged_modal_count == 0 &&
                     session_info.reserved_u32 == 0)) return 1;
    session_report.struct_size = sizeof(session_report);
    session_report.stage_kind = MIZU_STAGE_DECODE;
    session_report.backend_family = MIZU_BACKEND_FAMILY_APPLE;
    session_report.execution_route = MIZU_EXEC_ROUTE_ANE;
    session_report.plan_id = 9;
    session_report.selection_mode = MIZU_SELECTION_MODE_DIRECT;
    session_report.cold_state = MIZU_COLD_STATE_WARM;
    session_report.fallback_reason = MIZU_FALLBACK_REASON_UNSUPPORTED_OP;
    session_report.cache_flags = UINT64_C(9);
    session_report.elapsed_us = UINT64_C(9);
    status = mizu_session_get_last_report(NULL, &session_report);
    if (!expect_status("null session get last report", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("null session report should preserve struct size", session_report.struct_size == sizeof(session_report))) {
        return 1;
    }
    if (!expect_true("null session report should clear outputs",
                     session_report.stage_kind == 0 &&
                     session_report.backend_family == 0 &&
                     session_report.execution_route == 0 &&
                     session_report.plan_id == 0 &&
                     session_report.selection_mode == 0 &&
                     session_report.cold_state == 0 &&
                     session_report.fallback_reason == 0 &&
                     session_report.cache_flags == 0 &&
                     session_report.elapsed_us == 0)) return 1;
    session_info.struct_size = sizeof(session_info);
    status = mizu_model_close(model);
    if (!expect_status("model close with live session", status, MIZU_STATUS_BUSY)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session remains valid after busy model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_open(model, &session_config, &session_reuse);
    if (!expect_status("session reopen", status, MIZU_STATUS_OK)) return 1;
    session_info.struct_size = sizeof(session_info);
    session_info.session_state_flags = UINT64_C(9);
    session_info.kv_token_count = 9;
    session_info.staged_token_count = 9;
    session_info.staged_modal_count = 9;
    session_info.reserved_u32 = 9;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("closed session get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session get info should preserve struct size", session_info.struct_size == sizeof(session_info))) {
        return 1;
    }
    if (!expect_true("closed session get info should clear outputs",
                     session_info.session_state_flags == 0 &&
                     session_info.kv_token_count == 0 &&
                     session_info.staged_token_count == 0 &&
                     session_info.staged_modal_count == 0 &&
                     session_info.reserved_u32 == 0)) return 1;
    session_report.struct_size = sizeof(session_report);
    session_report.stage_kind = MIZU_STAGE_DECODE;
    session_report.backend_family = MIZU_BACKEND_FAMILY_APPLE;
    session_report.execution_route = MIZU_EXEC_ROUTE_ANE;
    session_report.plan_id = 9;
    session_report.selection_mode = MIZU_SELECTION_MODE_DIRECT;
    session_report.cold_state = MIZU_COLD_STATE_WARM;
    session_report.fallback_reason = MIZU_FALLBACK_REASON_UNSUPPORTED_OP;
    session_report.cache_flags = UINT64_C(9);
    session_report.elapsed_us = UINT64_C(9);
    status = mizu_session_get_last_report(session, &session_report);
    if (!expect_status("closed session get last report", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed session report should preserve struct size", session_report.struct_size == sizeof(session_report))) {
        return 1;
    }
    if (!expect_true("closed session report should clear outputs",
                     session_report.stage_kind == 0 &&
                     session_report.backend_family == 0 &&
                     session_report.execution_route == 0 &&
                     session_report.plan_id == 0 &&
                     session_report.selection_mode == 0 &&
                     session_report.cold_state == 0 &&
                     session_report.fallback_reason == 0 &&
                     session_report.cache_flags == 0 &&
                     session_report.elapsed_us == 0)) return 1;
    status = mizu_session_clear_pending_inputs(session);
    if (!expect_status("closed session clear pending inputs should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    memset(&modal_input, 0, sizeof(modal_input));
    modal_input.struct_size = sizeof(modal_input);
    modal_input.slot_name_z = "image";
    modal_input.placeholder_ordinal = 1;
    modal_input.modality_kind = MIZU_MODALITY_KIND_IMAGE;
    modal_input.storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES;
    modal_input.dtype = MIZU_DTYPE_U8;
    modal_input.data = image_bytes;
    modal_input.byte_count = sizeof(image_bytes);
    modal_input.lifetime_policy = MIZU_LIFETIME_POLICY_COPY;
    modal_input.input_flags = MIZU_INPUT_FLAG_NONE;
    status = mizu_session_attach_tokens(session, &decoded_token, 1, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("closed session attach tokens should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("closed session attach modal input should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decoded_token;
    decode_result.token_capacity = 1;
    decode_result.token_count = 9;
    decode_result.stop_reason = MIZU_STOP_REASON_STOP_SEQUENCE;
    decode_result.result_flags = UINT64_C(9);
    memset(&report_buffer, 0, sizeof(report_buffer));
    report_buffer.struct_size = sizeof(report_buffer);
    report_buffer.reports = report_storage;
    report_buffer.report_capacity = 1;
    report_buffer.report_count = 9;
    report_storage[0].struct_size = sizeof(report_storage[0]);
    report_storage[0].stage_kind = MIZU_STAGE_DECODE;
    report_storage[0].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[0].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[0].plan_id = 9;
    report_storage[0].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[0].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[0].fallback_reason = MIZU_FALLBACK_REASON_UNSUPPORTED_OP;
    report_storage[0].cache_flags = UINT64_C(9);
    report_storage[0].elapsed_us = UINT64_C(9);
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &report_buffer);
    if (!expect_status("closed session decode should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (!expect_true("closed session decode should preserve result inputs",
                     decode_result.struct_size == sizeof(decode_result) &&
                     decode_result.token_buffer == &decoded_token &&
                     decode_result.token_capacity == 1)) return 1;
    if (!expect_true("closed session decode should clear result outputs",
                     decode_result.token_count == 0 &&
                     decode_result.stop_reason == MIZU_STOP_REASON_NONE &&
                     decode_result.result_flags == 0)) return 1;
    if (!expect_true("closed session decode should preserve report buffer inputs",
                     report_buffer.struct_size == sizeof(report_buffer) &&
                     report_buffer.reports == report_storage &&
                     report_buffer.report_capacity == 1)) return 1;
    if (!expect_true("closed session decode should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("closed session decode should clear report payload",
                     report_storage[0].struct_size == sizeof(report_storage[0]) &&
                     report_storage[0].stage_kind == 0 &&
                     report_storage[0].backend_family == 0 &&
                     report_storage[0].execution_route == 0 &&
                     report_storage[0].plan_id == 0 &&
                     report_storage[0].selection_mode == 0 &&
                     report_storage[0].cold_state == 0 &&
                     report_storage[0].fallback_reason == 0 &&
                     report_storage[0].cache_flags == 0 &&
                     report_storage[0].elapsed_us == 0)) return 1;
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
    report_storage[1].stage_kind = MIZU_STAGE_PREFILL;
    report_storage[1].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[1].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[1].plan_id = 9;
    report_storage[1].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[1].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[1].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[1].cache_flags = UINT64_C(9);
    report_storage[1].elapsed_us = UINT64_C(9);
    status = mizu_session_prefill(session, &report_buffer);
    if (!expect_status("closed session prefill should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (!expect_true("closed session prefill should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("closed session prefill should clear both report payloads",
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
    report_buffer.report_capacity = 1;
    report_buffer.report_count = 9;
    report_storage[0].struct_size = sizeof(report_storage[0]);
    report_storage[0].stage_kind = MIZU_STAGE_PARK;
    report_storage[0].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[0].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[0].plan_id = 9;
    report_storage[0].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[0].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[0].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[0].cache_flags = UINT64_C(9);
    report_storage[0].elapsed_us = UINT64_C(9);
    status = mizu_session_park(session, &report_buffer);
    if (!expect_status("closed session park should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (!expect_true("closed session park should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("closed session park should clear report payload",
                     report_storage[0].struct_size == sizeof(report_storage[0]) &&
                     report_storage[0].stage_kind == 0 &&
                     report_storage[0].backend_family == 0 &&
                     report_storage[0].execution_route == 0 &&
                     report_storage[0].plan_id == 0 &&
                     report_storage[0].selection_mode == 0 &&
                     report_storage[0].cold_state == 0 &&
                     report_storage[0].fallback_reason == 0 &&
                     report_storage[0].cache_flags == 0 &&
                     report_storage[0].elapsed_us == 0)) return 1;
    report_buffer.report_count = 9;
    report_storage[0].struct_size = sizeof(report_storage[0]);
    report_storage[0].stage_kind = MIZU_STAGE_RESUME;
    report_storage[0].backend_family = MIZU_BACKEND_FAMILY_APPLE;
    report_storage[0].execution_route = MIZU_EXEC_ROUTE_ANE;
    report_storage[0].plan_id = 9;
    report_storage[0].selection_mode = MIZU_SELECTION_MODE_DIRECT;
    report_storage[0].cold_state = MIZU_COLD_STATE_WARM;
    report_storage[0].fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report_storage[0].cache_flags = UINT64_C(9);
    report_storage[0].elapsed_us = UINT64_C(9);
    status = mizu_session_resume(session, &report_buffer);
    if (!expect_status("closed session resume should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (!expect_true("closed session resume should clear report count", report_buffer.report_count == 0)) return 1;
    if (!expect_true("closed session resume should clear report payload",
                     report_storage[0].struct_size == sizeof(report_storage[0]) &&
                     report_storage[0].stage_kind == 0 &&
                     report_storage[0].backend_family == 0 &&
                     report_storage[0].execution_route == 0 &&
                     report_storage[0].plan_id == 0 &&
                     report_storage[0].selection_mode == 0 &&
                     report_storage[0].cold_state == 0 &&
                     report_storage[0].fallback_reason == 0 &&
                     report_storage[0].cache_flags == 0 &&
                     report_storage[0].elapsed_us == 0)) return 1;
    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = &output_token;
    output_buffer.byte_capacity = sizeof(output_token);
    output_buffer.bytes_written = 9;
    output_buffer.output_flags = UINT64_C(9);
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("closed session read output should reject stale handle", status, MIZU_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (!expect_true("closed session read output should preserve inputs",
                     output_buffer.struct_size == sizeof(output_buffer) &&
                     output_buffer.output_kind == MIZU_OUTPUT_KIND_TOKEN_IDS &&
                     output_buffer.data == &output_token &&
                     output_buffer.byte_capacity == sizeof(output_token))) return 1;
    if (!expect_true("closed session read output should clear outputs",
                     output_buffer.bytes_written == 0 &&
                     output_buffer.output_flags == 0)) return 1;
    status = mizu_session_close(session);
    if (!expect_status("double session close", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    session_info_reuse.struct_size = sizeof(session_info_reuse);
    status = mizu_session_get_info(session_reuse, &session_info_reuse);
    if (!expect_status("reopened session should remain valid", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session_reuse);
    if (!expect_status("reopened session close", status, MIZU_STATUS_OK)) return 1;
    bad_session_config = session_config;
    bad_session_config.abi_version = 0;
    failed_session = session_reuse;
    status = mizu_session_open(model, &bad_session_config, &failed_session);
    if (!expect_status("session open should reject ABI mismatch", status, MIZU_STATUS_ABI_MISMATCH)) return 1;
    if (!expect_true("failed ABI-mismatched session open should clear output handle", failed_session == NULL)) return 1;

    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    failed_session = session_reuse;
    status = mizu_session_open(model, &session_config, &failed_session);
    if (!expect_status("session open should reject closed model", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("failed session open should clear output handle", failed_session == NULL)) return 1;
    status = mizu_model_open(runtime, &model_config, &model_reuse);
    if (!expect_status("model reopen", status, MIZU_STATUS_OK)) return 1;
    model_info.struct_size = sizeof(model_info);
    model_info.model_family = MIZU_MODEL_FAMILY_GEMMA4;
    model_info.allowed_backend_mask = MIZU_BACKEND_MASK_CUDA;
    model_info.model_features = MIZU_MODEL_FEATURE_PROJECTOR;
    model_info.projector_slot_count = 9;
    model_info.reserved_u32 = 9;
    status = mizu_model_get_info(model, &model_info);
    if (!expect_status("closed model get info", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed model get info should preserve struct size", model_info.struct_size == sizeof(model_info))) {
        return 1;
    }
    if (!expect_true("closed model get info should clear outputs",
                     model_info.model_family == 0 &&
                     model_info.allowed_backend_mask == 0 &&
                     model_info.model_features == 0 &&
                     model_info.projector_slot_count == 0 &&
                     model_info.reserved_u32 == 0)) return 1;
    model_report.struct_size = sizeof(model_report);
    model_report.stage_kind = MIZU_STAGE_MODEL_LOAD;
    model_report.backend_family = MIZU_BACKEND_FAMILY_APPLE;
    model_report.execution_route = MIZU_EXEC_ROUTE_ANE;
    model_report.plan_id = 9;
    model_report.selection_mode = MIZU_SELECTION_MODE_DIRECT;
    model_report.cold_state = MIZU_COLD_STATE_COLD;
    model_report.fallback_reason = MIZU_FALLBACK_REASON_BACKEND_UNAVAILABLE;
    model_report.cache_flags = UINT64_C(9);
    model_report.elapsed_us = UINT64_C(9);
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("closed model get last report", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("closed model report should preserve struct size", model_report.struct_size == sizeof(model_report))) {
        return 1;
    }
    if (!expect_true("closed model report should clear outputs",
                     model_report.stage_kind == 0 &&
                     model_report.backend_family == 0 &&
                     model_report.execution_route == 0 &&
                     model_report.plan_id == 0 &&
                     model_report.selection_mode == 0 &&
                     model_report.cold_state == 0 &&
                     model_report.fallback_reason == 0 &&
                     model_report.cache_flags == 0 &&
                     model_report.elapsed_us == 0)) return 1;
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
    required_bytes = 9;
    status = mizu_runtime_copy_last_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes);
    if (!expect_status("destroyed runtime copy last error", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("destroyed runtime copy last error should clear required size", required_bytes == 0)) return 1;
    if (!expect_true("destroyed runtime copy last error should clear error buffer", error_buffer[0] == '\0')) return 1;
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
