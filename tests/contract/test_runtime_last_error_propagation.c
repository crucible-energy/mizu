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

static int copy_runtime_error(
    mizu_runtime_t *runtime,
    char *buffer,
    size_t buffer_size,
    size_t *required_bytes
) {
    mizu_status_code_t status;

    memset(buffer, 0, buffer_size);
    *required_bytes = 0;
    status = mizu_runtime_copy_last_error(runtime, buffer, buffer_size, required_bytes);
    return expect_status("copy runtime last error", status, MIZU_STATUS_OK);
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    size_t required_bytes;
    char error_buffer[256];
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_output_buffer_t output_buffer;
    mizu_execution_report_t model_report;
    mizu_execution_report_t resume_reports[1];
    mizu_report_buffer_t resume_buffer;

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
    memset(&model_report, 0, sizeof(model_report));
    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("model report after open", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("model report after open should be model load",
                     model_report.stage_kind == MIZU_STAGE_MODEL_LOAD)) return 1;

    memset(&session_config, 0, sizeof(session_config));
    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_prefill(session, NULL);
    if (!expect_status("prefill without pending inputs", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("prefill failure should explain invalid session state",
                     strstr(error_buffer, "session cannot prefill in current state") != NULL)) return 1;

    {
        mizu_decode_options_t decode_options;
        mizu_decode_result_t decode_result;

        memset(&decode_options, 0, sizeof(decode_options));
        decode_options.struct_size = sizeof(decode_options);
        decode_options.token_budget = 1;

        memset(&decode_result, 0, sizeof(decode_result));
        decode_result.struct_size = sizeof(decode_result);

        status = mizu_session_decode_step(session, &decode_options, &decode_result, NULL);
        if (!expect_status("decode without live context", status, MIZU_STATUS_INVALID_STATE)) return 1;
        if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
        if (!expect_true("decode failure should replace prefill error text",
                         strstr(error_buffer, "session cannot decode in current state") != NULL)) return 1;
    }

    status = mizu_session_park(session, NULL);
    if (!expect_status("park without live context", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("park failure should replace decode error text",
                     strstr(error_buffer, "session cannot park in current state") != NULL)) return 1;

    status = mizu_model_close(model);
    if (!expect_status("model close with live session", status, MIZU_STATUS_BUSY)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("model-close failure should publish non-empty runtime error", required_bytes > 1)) return 1;
    if (!expect_true("model-close failure should explain live sessions",
                     strstr(error_buffer, "model cannot close while sessions are live") != NULL)) return 1;
    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("model report after failed close", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("failed model close should preserve model-load report",
                     model_report.stage_kind == MIZU_STAGE_MODEL_LOAD)) return 1;

    memset(&output_buffer, 0, sizeof(output_buffer));
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("read output before decode", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("read-output failure should explain missing decode output",
                     strstr(error_buffer, "session has no decode output to read") != NULL)) return 1;

    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&resume_buffer, 0, sizeof(resume_buffer));
    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume before park", status, MIZU_STATUS_INVALID_STATE)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("resume failure should replace earlier error text",
                     strstr(error_buffer, "session cannot resume in current state") != NULL)) return 1;
    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("model report after session failures", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("session failures should not mutate model last report",
                     model_report.stage_kind == MIZU_STAGE_MODEL_LOAD)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_runtime_last_error_propagation: PASS");
    return 0;
}
