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
    int32_t initial_tokens[2] = {11, 22};
    int32_t restaged_tokens[1] = {33};
    int32_t decoded_token_a = -1;
    int32_t decoded_token_b = -1;
    int32_t output_token = -1;
    int32_t output_token_repeat = -1;
    mizu_execution_report_t prefill_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_output_buffer_t output_buffer;
    mizu_session_info_t session_info;
    mizu_execution_report_t last_report;

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
    session_config.max_context_tokens = 16;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    memset(&decode_options, 0, sizeof(decode_options));
    memset(&decode_result, 0, sizeof(decode_result));
    memset(&output_buffer, 0, sizeof(output_buffer));
    memset(&session_info, 0, sizeof(session_info));
    memset(&last_report, 0, sizeof(last_report));

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 2;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_capacity = 1;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;

    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.byte_capacity = sizeof(output_token);

    session_info.struct_size = sizeof(session_info);
    last_report.struct_size = sizeof(last_report);

    status = mizu_session_attach_tokens(session, initial_tokens, 2, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach initial tokens", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("token-only prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("token-only prefill should emit one report", prefill_buffer.report_count == 1)) return 1;
    if (!expect_true("token-only prefill should emit prefill stage",
                     prefill_reports[0].stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after token-only prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("token-only prefill should clear pending flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) return 1;
    if (!expect_true("token-only prefill should set live context flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("token-only prefill should clear staged token count", session_info.staged_token_count == 0)) return 1;
    if (!expect_true("token-only prefill should advance kv tokens", session_info.kv_token_count == 2)) return 1;
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after token-only prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("token-only prefill should publish prefill last report",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_attach_tokens(session, restaged_tokens, 1, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach restaged tokens", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info with pending token restage", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("restaged token should set pending flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) != 0)) return 1;
    if (!expect_true("restaged token should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("restaged token should preserve kv tokens", session_info.kv_token_count == 2)) return 1;
    if (!expect_true("restaged token should report staged token count", session_info.staged_token_count == 1)) return 1;

    decode_result.token_buffer = &decoded_token_a;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("decode should reject pending staged tokens", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!expect_true("rejected decode should not publish token count", decode_result.token_count == 0)) return 1;
    if (!expect_true("rejected decode should preserve stop reason none",
                     decode_result.stop_reason == MIZU_STOP_REASON_NONE)) return 1;
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after rejected decode with pending tokens", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("rejected decode should preserve prefill last report",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    prefill_buffer.report_count = 0;
    memset(prefill_reports, 0, sizeof(prefill_reports));
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("second token-only prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("second token-only prefill should emit one report", prefill_buffer.report_count == 1)) return 1;
    if (!expect_true("second token-only prefill should still emit prefill stage",
                     prefill_reports[0].stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after second prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("second prefill should clear pending flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) return 1;
    if (!expect_true("second prefill should extend kv tokens", session_info.kv_token_count == 3)) return 1;

    decode_result.token_buffer = &decoded_token_a;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("first decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("first decode should not exceed caller token budget", decode_result.token_count <= 2)) return 1;
    if (!expect_true("first decode should emit one token", decode_result.token_count == 1)) return 1;
    if (!expect_true("first decode should stay non-terminal", decode_result.stop_reason == MIZU_STOP_REASON_NONE)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after first decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("first decode should advance kv tokens", session_info.kv_token_count == 4)) return 1;

    output_buffer.data = &output_token;
    output_buffer.bytes_written = 0;
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output after first decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("read output should return most recent decode token", output_token == decoded_token_a)) return 1;
    if (!expect_true("read output should report one token worth of bytes",
                     output_buffer.bytes_written == sizeof(int32_t))) return 1;

    output_buffer.data = &output_token_repeat;
    output_buffer.bytes_written = 0;
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("repeat read output after first decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("repeat read output should be stable", output_token_repeat == decoded_token_a)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after repeat read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("read output should not advance kv tokens", session_info.kv_token_count == 4)) return 1;

    decode_result.token_buffer = &decoded_token_b;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("second decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("second decode should not exceed caller token budget", decode_result.token_count <= 2)) return 1;
    if (!expect_true("second decode should emit one token", decode_result.token_count == 1)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after second decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("second decode should extend existing live context", session_info.kv_token_count == 5)) return 1;

    output_buffer.data = &output_token;
    output_buffer.bytes_written = 0;
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output after second decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("read output should update to the newest decode token", output_token == decoded_token_b)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_token_only_execution: PASS");
    return 0;
}
