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
    int32_t tokens[3] = {1, 2, 3};
    uint8_t image_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t decoded_token = -1;
    int32_t output_token = -1;
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t resume_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_modal_input_desc_t modal_input;
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
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(park_reports, 0, sizeof(park_reports));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    memset(&park_buffer, 0, sizeof(park_buffer));
    memset(&resume_buffer, 0, sizeof(resume_buffer));
    memset(&decode_options, 0, sizeof(decode_options));
    memset(&decode_result, 0, sizeof(decode_result));
    memset(&output_buffer, 0, sizeof(output_buffer));
    memset(&session_info, 0, sizeof(session_info));
    memset(&last_report, 0, sizeof(last_report));

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 2;
    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decoded_token;
    decode_result.token_capacity = 1;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;

    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = &output_token;
    output_buffer.byte_capacity = sizeof(output_token);

    session_info.struct_size = sizeof(session_info);

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill without pending inputs", status, MIZU_STATUS_INVALID_STATE)) return 1;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("decode without live context", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!expect_true("invalid decode should not publish token count", decode_result.token_count == 0)) return 1;
    if (!expect_true("invalid decode should not publish stop reason",
                     decode_result.stop_reason == MIZU_STOP_REASON_NONE)) return 1;

    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park without live context", status, MIZU_STATUS_INVALID_STATE)) return 1;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume without parked context", status, MIZU_STATUS_INVALID_STATE)) return 1;

    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output before decode", status, MIZU_STATUS_INVALID_STATE)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info before prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("fresh session should have no state flags", session_info.session_state_flags == 0)) return 1;
    if (!expect_true("fresh session should have zero kv tokens", session_info.kv_token_count == 0)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report on fresh session", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("fresh-session invalid ops should preserve none last report",
                     last_report.stage_kind == MIZU_STAGE_NONE)) return 1;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens", status, MIZU_STATUS_OK)) return 1;

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

    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("attach modal input", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill with pending inputs", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("prefill should publish live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("prefill should clear parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) == 0)) return 1;
    if (!expect_true("prefill should advance kv tokens", session_info.kv_token_count == 3)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("last report after prefill should be prefill",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens on live session", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("attach modal input on live session", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info with restaged inputs", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("restaged inputs should set pending flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) != 0)) return 1;
    if (!expect_true("restaged inputs should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("restaged inputs should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    if (!expect_true("restaged token count should be reported", session_info.staged_token_count == 3)) return 1;
    if (!expect_true("restaged modal count should be reported", session_info.staged_modal_count == 1)) return 1;

    status = mizu_session_clear_pending_inputs(session);
    if (!expect_status("clear pending inputs on live session", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after clearing live-session staging", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("clearing pending inputs should clear pending flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) return 1;
    if (!expect_true("clearing pending inputs should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("clearing pending inputs should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    if (!expect_true("clearing pending inputs should clear staged token count", session_info.staged_token_count == 0)) return 1;
    if (!expect_true("clearing pending inputs should clear staged modal count", session_info.staged_modal_count == 0)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after clearing live-session staging", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("clearing pending inputs should preserve prefill last report",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_prefill(session, NULL);
    if (!expect_status("prefill after clearing restaged inputs", status, MIZU_STATUS_INVALID_STATE)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after cleared-input prefill rejection", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cleared-input prefill rejection should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("cleared-input prefill rejection should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after cleared-input prefill rejection", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cleared-input prefill rejection should preserve prefill last report",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    park_buffer.struct_size = sizeof(park_buffer) - 1;
    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park should reject short report buffer struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = NULL;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;
    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park should reject missing report storage", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("park should still report required count when storage is missing",
                     park_buffer.report_count == 1)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after rejected park outputs", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("rejected park outputs should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("rejected park outputs should not set parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) == 0)) return 1;
    if (!expect_true("rejected park outputs should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after rejected park outputs", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("rejected park outputs should preserve prefill last report",
                     last_report.stage_kind == MIZU_STAGE_PREFILL)) return 1;

    status = mizu_session_park(session, NULL);
    if (!expect_status("park should allow null report buffer", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after null-buffer park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("last report after null-buffer park should be park",
                     last_report.stage_kind == MIZU_STAGE_PARK)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("park should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("park should set parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) return 1;

    status = mizu_session_attach_tokens(session, tokens, 1, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens while parked", status, MIZU_STATUS_INVALID_STATE)) return 1;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill while parked", status, MIZU_STATUS_INVALID_STATE)) return 1;

    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
    if (!expect_status("decode while parked", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!expect_true("parked invalid decode should not publish token count", decode_result.token_count == 0)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after parked invalid ops", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("parked invalid ops should preserve parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) return 1;
    if (!expect_true("parked invalid ops should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after parked invalid ops", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("parked invalid ops should preserve park last report",
                     last_report.stage_kind == MIZU_STAGE_PARK)) return 1;

    resume_buffer.struct_size = sizeof(resume_buffer) - 1;
    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume should reject short report buffer struct", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = NULL;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;
    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume should reject missing report storage", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("resume should still report required count when storage is missing",
                     resume_buffer.report_count == 1)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after rejected resume outputs", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("rejected resume outputs should preserve parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) return 1;
    if (!expect_true("rejected resume outputs should preserve live context flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("rejected resume outputs should preserve kv tokens", session_info.kv_token_count == 3)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after rejected resume outputs", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("rejected resume outputs should preserve park last report",
                     last_report.stage_kind == MIZU_STAGE_PARK)) return 1;

    status = mizu_session_resume(session, NULL);
    if (!expect_status("resume should allow null report buffer", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("last report after null-buffer resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("last report after null-buffer resume should be resume",
                     last_report.stage_kind == MIZU_STAGE_RESUME)) return 1;

    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("resume should clear parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) == 0)) return 1;
    if (!expect_true("resume should preserve live context",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_session_state_guards: PASS");
    return 0;
}
