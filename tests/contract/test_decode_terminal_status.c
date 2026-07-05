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
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int32_t staged_tokens[3] = {1, 2, 3};
    int32_t decoded_token = -1;
    int32_t output_token = -1;
    mizu_execution_report_t decode_reports[1];
    mizu_report_buffer_t decode_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_output_buffer_t output_buffer;
    mizu_session_info_t session_info;

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

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    memset(&model_config, 0, sizeof(model_config));
    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;

    memset(&session_config, 0, sizeof(session_config));
    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_attach_tokens(session, staged_tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_prefill(session, NULL);
    if (!expect_status("prefill", status, MIZU_STATUS_OK)) return 1;

    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decoded_token;
    decode_result.token_capacity = 1;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(&decode_buffer, 0, sizeof(decode_buffer));
    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("decode at context limit", status, MIZU_STATUS_END_OF_SEQUENCE)) return 1;
    if (!expect_true("terminal decode should still emit one token", decode_result.token_count == 1)) return 1;
    if (!expect_true("terminal decode should publish token-budget stop reason",
                     decode_result.stop_reason == MIZU_STOP_REASON_TOKEN_BUDGET)) return 1;
    if (!expect_true("terminal decode token should be positive", decoded_token > 0)) return 1;
    if (!expect_true("terminal decode should publish one decode report", decode_buffer.report_count == 1)) return 1;
    if (!expect_true("terminal decode report should be decode", decode_reports[0].stage_kind == MIZU_STAGE_DECODE)) {
        return 1;
    }

    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = &output_token;
    output_buffer.byte_capacity = sizeof(output_token);
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read terminal decode output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("read output should return the terminal token", output_token == decoded_token)) return 1;

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after terminal decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("terminal decode should advance kv tokens to the configured limit",
                     session_info.kv_token_count == 4)) return 1;

    decoded_token = -1;
    decode_result.token_count = 99;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    memset(decode_reports, 0, sizeof(decode_reports));
    decode_buffer.report_count = 0;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("repeat decode after exhaustion", status, MIZU_STATUS_END_OF_SEQUENCE)) return 1;
    if (!expect_true("repeat terminal decode should emit no tokens", decode_result.token_count == 0)) return 1;
    if (!expect_true("repeat terminal decode should preserve token-budget stop reason",
                     decode_result.stop_reason == MIZU_STOP_REASON_TOKEN_BUDGET)) return 1;
    if (!expect_true("repeat terminal decode should still publish one decode report", decode_buffer.report_count == 1)) {
        return 1;
    }
    if (!expect_true("repeat terminal decode report should be decode", decode_reports[0].stage_kind == MIZU_STAGE_DECODE)) {
        return 1;
    }

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_decode_terminal_status: PASS");
    return 0;
}
