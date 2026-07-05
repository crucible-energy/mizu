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

static int expect_i64(const char *label, int64_t actual, int64_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %lld, got %lld\n", label,
                (long long)expected, (long long)actual);
        return 0;
    }
    return 1;
}

static int expect_u64(const char *label, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %llu, got %llu\n", label,
                (unsigned long long)expected, (unsigned long long)actual);
        return 0;
    }
    return 1;
}

static int expect_report_equal(const char *label,
                               const mizu_execution_report_t *actual,
                               const mizu_execution_report_t *expected) {
    if (actual->stage_kind != expected->stage_kind ||
        actual->backend_family != expected->backend_family ||
        actual->execution_route != expected->execution_route ||
        actual->plan_id != expected->plan_id ||
        actual->selection_mode != expected->selection_mode ||
        actual->cold_state != expected->cold_state ||
        actual->fallback_reason != expected->fallback_reason ||
        actual->cache_flags != expected->cache_flags ||
        actual->elapsed_us != expected->elapsed_us) {
        fprintf(stderr, "%s\n", label);
        return 0;
    }
    return 1;
}

static void fill_tokens(int32_t *tokens, size_t count) {
    size_t index;

    for (index = 0; index < count; ++index) {
        tokens[index] = (int32_t)(index + 1U);
    }
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model_dual = NULL;
    mizu_model_t *model_ane = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_fail = NULL;
    mizu_status_code_t status;
    int32_t long_tokens[65];
    int32_t short_tokens[3] = {1, 2, 3};
    int32_t decode_tokens[1] = {0};
    uint8_t image_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info_before;
    mizu_session_info_t session_info_after;
    mizu_modal_input_desc_t modal_input;
    mizu_report_buffer_t prefill_buffer_dual;
    mizu_execution_report_t prefill_reports_dual[2];
    mizu_execution_report_t last_report;
    mizu_report_buffer_t prefill_buffer_single;
    mizu_execution_report_t prefill_report_single[1];
    mizu_report_buffer_t decode_buffer;
    mizu_execution_report_t decode_report[1];
    mizu_report_buffer_t park_buffer;
    mizu_execution_report_t park_report[1];
    mizu_report_buffer_t resume_buffer;
    mizu_execution_report_t resume_report[1];
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_report_buffer_t prefill_buffer_fail;
    mizu_execution_report_t prefill_report_fail[1];
    mizu_execution_report_t last_report_before_fail;

    fill_tokens(long_tokens, sizeof(long_tokens) / sizeof(long_tokens[0]));
    memset(prefill_reports_dual, 0, sizeof(prefill_reports_dual));
    memset(prefill_report_single, 0, sizeof(prefill_report_single));
    memset(decode_report, 0, sizeof(decode_report));
    memset(park_report, 0, sizeof(park_report));
    memset(resume_report, 0, sizeof(resume_report));
    memset(prefill_report_fail, 0, sizeof(prefill_report_fail));
    memset(&last_report, 0, sizeof(last_report));
    memset(&last_report_before_fail, 0, sizeof(last_report_before_fail));
    memset(&session_info_before, 0, sizeof(session_info_before));
    memset(&session_info_after, 0, sizeof(session_info_after));

    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE\n");
        return 1;
    }
    if (setenv("MIZU_FORCE_APPLE_METAL_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_METAL_AVAILABLE\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        return 1;
    }
    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE\n");
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
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE | MIZU_BACKEND_MASK_APPLE_METAL;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model_dual);
    if (!expect_status("dual-route model open", status, MIZU_STATUS_OK)) return 1;

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

    status = mizu_session_open(model_dual, &session_config, &session);
    if (!expect_status("dual-route session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session, long_tokens, sizeof(long_tokens) / sizeof(long_tokens[0]),
                                        MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("dual-route attach tokens", status, MIZU_STATUS_OK)) return 1;

    modal_input.struct_size = sizeof(modal_input);
    modal_input.slot_name_z = "image";
    modal_input.placeholder_ordinal = 1;
    modal_input.modality_kind = MIZU_MODALITY_KIND_IMAGE;
    modal_input.storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES;
    modal_input.dtype = MIZU_DTYPE_U8;
    modal_input.rank = 0;
    modal_input.shape = NULL;
    modal_input.data = image_bytes;
    modal_input.byte_count = sizeof(image_bytes);
    modal_input.lifetime_policy = MIZU_LIFETIME_POLICY_COPY;
    modal_input.input_flags = MIZU_INPUT_FLAG_NONE;

    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("dual-route attach modal", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_dual.struct_size = sizeof(prefill_buffer_dual);
    prefill_buffer_dual.reports = prefill_reports_dual;
    prefill_buffer_dual.report_capacity = 2;
    prefill_buffer_dual.report_count = 0;

    status = mizu_session_prefill(session, &prefill_buffer_dual);
    if (!expect_status("dual-route prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("dual-route prefill should emit projector then prefill",
                     prefill_buffer_dual.report_count == 2 &&
                     prefill_reports_dual[0].stage_kind == MIZU_STAGE_PROJECTOR &&
                     prefill_reports_dual[1].stage_kind == MIZU_STAGE_PREFILL)) return 1;
    if (!expect_true("projector should stay on ANE",
                     prefill_reports_dual[0].execution_route == MIZU_EXEC_ROUTE_ANE &&
                     prefill_reports_dual[0].fallback_reason == MIZU_FALLBACK_REASON_NONE)) return 1;
    if (!expect_true("prefill should fall back to Metal with unsupported-shape reason",
                     prefill_reports_dual[1].execution_route == MIZU_EXEC_ROUTE_METAL &&
                     prefill_reports_dual[1].fallback_reason == MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE)) return 1;

    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("dual-route get_last_report after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("last report should mirror the most recent prefill report",
                             &last_report, &prefill_reports_dual[1])) return 1;

    session_info_after.struct_size = sizeof(session_info_after);
    status = mizu_session_get_info(session, &session_info_after);
    if (!expect_status("dual-route session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("dual-route session should expose a live context after prefill",
                     (session_info_after.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 1;
    if (!expect_true("dual-route prefill should clear pending inputs",
                     (session_info_after.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) return 1;
    if (!expect_i64("dual-route prefill should advance kv count",
                    session_info_after.kv_token_count, 65)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("dual-route session close", status, MIZU_STATUS_OK)) return 1;
    session = NULL;
    status = mizu_model_close(model_dual);
    if (!expect_status("dual-route model close", status, MIZU_STATUS_OK)) return 1;
    model_dual = NULL;

    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    status = mizu_model_open(runtime, &model_config, &model_ane);
    if (!expect_status("ANE-only model open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_open(model_ane, &session_config, &session);
    if (!expect_status("ANE-only session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session, short_tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("ANE-only attach short tokens", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_single.struct_size = sizeof(prefill_buffer_single);
    prefill_buffer_single.reports = prefill_report_single;
    prefill_buffer_single.report_capacity = 1;
    prefill_buffer_single.report_count = 0;

    status = mizu_session_prefill(session, &prefill_buffer_single);
    if (!expect_status("ANE-only token prefill", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("get_last_report after token prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("last report should mirror token prefill",
                             &last_report, &prefill_report_single[0])) return 1;

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = decode_tokens;
    decode_result.token_capacity = 1;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result.result_flags = 0;

    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_report;
    decode_buffer.report_capacity = 1;
    decode_buffer.report_count = 0;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("ANE-only decode", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("get_last_report after decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("last report should mirror decode", &last_report, &decode_report[0])) return 1;

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_report;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("ANE-only park", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("get_last_report after park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("last report should mirror park", &last_report, &park_report[0])) return 1;

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_report;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("ANE-only resume", status, MIZU_STATUS_OK)) return 1;
    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session, &last_report);
    if (!expect_status("get_last_report after resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("last report should mirror resume", &last_report, &resume_report[0])) return 1;

    status = mizu_session_close(session);
    if (!expect_status("ANE-only session close", status, MIZU_STATUS_OK)) return 1;
    session = NULL;

    status = mizu_session_open(model_ane, &session_config, &session_fail);
    if (!expect_status("ANE-only failure session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_fail, long_tokens, sizeof(long_tokens) / sizeof(long_tokens[0]),
                                        MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("ANE-only attach long tokens", status, MIZU_STATUS_OK)) return 1;

    session_info_before.struct_size = sizeof(session_info_before);
    status = mizu_session_get_info(session_fail, &session_info_before);
    if (!expect_status("session info before failed prefill", status, MIZU_STATUS_OK)) return 1;
    last_report_before_fail.struct_size = sizeof(last_report_before_fail);
    status = mizu_session_get_last_report(session_fail, &last_report_before_fail);
    if (!expect_status("last report before failed prefill", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_fail.struct_size = sizeof(prefill_buffer_fail);
    prefill_buffer_fail.reports = prefill_report_fail;
    prefill_buffer_fail.report_capacity = 1;
    prefill_buffer_fail.report_count = 0;

    status = mizu_session_prefill(session_fail, &prefill_buffer_fail);
    if (!expect_status("ANE-only oversized prefill should fail", status, MIZU_STATUS_NO_VALID_PLAN)) return 1;

    session_info_after.struct_size = sizeof(session_info_after);
    status = mizu_session_get_info(session_fail, &session_info_after);
    if (!expect_status("session info after failed prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_u64("failed prefill should preserve session flags",
                    session_info_after.session_state_flags, session_info_before.session_state_flags)) return 1;
    if (!expect_i64("failed prefill should preserve kv token count",
                    session_info_after.kv_token_count, session_info_before.kv_token_count)) return 1;
    if (!expect_i64("failed prefill should preserve staged token count",
                    session_info_after.staged_token_count, session_info_before.staged_token_count)) return 1;
    if (!expect_true("failed prefill should preserve staged modal count",
                     session_info_after.staged_modal_count == session_info_before.staged_modal_count)) return 1;

    last_report.struct_size = sizeof(last_report);
    status = mizu_session_get_last_report(session_fail, &last_report);
    if (!expect_status("last report after failed prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_report_equal("failed prefill should not change last report",
                             &last_report, &last_report_before_fail)) return 1;

    status = mizu_session_close(session_fail);
    if (!expect_status("ANE-only failure session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_ane);
    if (!expect_status("ANE-only model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_backend_routing_contracts: PASS");
    return 0;
}
