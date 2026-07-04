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

static int file_contains_substring(const char *path, const char *needle) {
    FILE *file = NULL;
    char line[2048];

    file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        if (strstr(line, needle) != NULL) {
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_runtime_t *runtime_fresh = NULL;
    mizu_runtime_t *runtime_explore = NULL;
    mizu_runtime_t *runtime_persist_a = NULL;
    mizu_runtime_t *runtime_persist_b = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *model_cached = NULL;
    mizu_model_t *model_fresh = NULL;
    mizu_model_t *model_explore_a = NULL;
    mizu_model_t *model_explore_b = NULL;
    mizu_model_t *model_explore_c = NULL;
    mizu_model_t *model_persist_a = NULL;
    mizu_model_t *model_persist_b = NULL;
    mizu_model_t *model_persist_c = NULL;
    mizu_model_t *model_persist_reuse = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_cached = NULL;
    mizu_session_t *session_persist_a = NULL;
    mizu_session_t *session_persist_b = NULL;
    mizu_status_code_t status;
    int command_status;
    int32_t tokens[3] = {1, 2, 3};
    uint8_t image_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t decode_tokens[1] = {0};
    int32_t decode_tokens_cached[1] = {0};
    const char *persist_root = "/tmp/mizu_stage_report_persist";
    const char *persist_artifact_cache_path = "/tmp/mizu_stage_report_persist/artifact_cache_v1.txt";
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t prefill_reports_cached[2];
    mizu_execution_report_t decode_reports[1];
    mizu_execution_report_t decode_reports_cached[1];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t park_reports_cached[1];
    mizu_execution_report_t resume_reports[1];
    mizu_execution_report_t model_reports[2];
    mizu_execution_report_t fresh_model_report;
    mizu_execution_report_t explore_reports[3];
    mizu_execution_report_t persist_reports[4];
    mizu_execution_report_t persist_prefill_reports[2];
    mizu_execution_report_t persist_prefill_reports_reuse[2];
    mizu_execution_report_t persist_decode_reports[1];
    mizu_execution_report_t persist_decode_reports_reuse[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t prefill_buffer_cached;
    mizu_report_buffer_t decode_buffer;
    mizu_report_buffer_t decode_buffer_cached;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t park_buffer_cached;
    mizu_report_buffer_t resume_buffer;
    mizu_report_buffer_t persist_prefill_buffer;
    mizu_report_buffer_t persist_prefill_buffer_reuse;
    mizu_report_buffer_t persist_decode_buffer;
    mizu_report_buffer_t persist_decode_buffer_reuse;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_runtime_config_t persist_runtime_config;
    mizu_model_open_config_t explore_model_config;
    mizu_session_config_t session_config;
    mizu_modal_input_desc_t modal_input;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    size_t report_index;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(prefill_reports_cached, 0, sizeof(prefill_reports_cached));
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(decode_reports_cached, 0, sizeof(decode_reports_cached));
    memset(park_reports, 0, sizeof(park_reports));
    memset(park_reports_cached, 0, sizeof(park_reports_cached));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(model_reports, 0, sizeof(model_reports));
    memset(&fresh_model_report, 0, sizeof(fresh_model_report));
    memset(explore_reports, 0, sizeof(explore_reports));
    memset(persist_reports, 0, sizeof(persist_reports));
    memset(persist_prefill_reports, 0, sizeof(persist_prefill_reports));
    memset(persist_prefill_reports_reuse, 0, sizeof(persist_prefill_reports_reuse));
    memset(persist_decode_reports, 0, sizeof(persist_decode_reports));
    memset(persist_decode_reports_reuse, 0, sizeof(persist_decode_reports_reuse));

    for (report_index = 0; report_index < 2; ++report_index) {
        model_reports[report_index].struct_size = sizeof(model_reports[report_index]);
    }
    fresh_model_report.struct_size = sizeof(fresh_model_report);
    for (report_index = 0; report_index < 3; ++report_index) {
        explore_reports[report_index].struct_size = sizeof(explore_reports[report_index]);
    }
    for (report_index = 0; report_index < 4; ++report_index) {
        persist_reports[report_index].struct_size = sizeof(persist_reports[report_index]);
    }

    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE\n");
        return 1;
    }
    if (setenv("MIZU_FORCE_APPLE_METAL_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_METAL_AVAILABLE\n");
        unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
        return 1;
    }

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = NULL;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_config.exploration_budget = 0;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;
    explore_model_config = model_config;
    explore_model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE | MIZU_BACKEND_MASK_APPLE_METAL;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model, &model_reports[0]);
    if (!expect_status("model report 1", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("first model load should be cold weight miss", (model_reports[0].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) == 0)) return 1;
    if (!expect_true("first model load should be direct", model_reports[0].selection_mode == MIZU_SELECTION_MODE_DIRECT)) return 1;
    if (!expect_true("first model load should report elapsed time", model_reports[0].elapsed_us > 0)) return 1;

    status = mizu_model_open(runtime, &model_config, &model_cached);
    if (!expect_status("model open cached", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_cached, &model_reports[1]);
    if (!expect_status("model report 2", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("second model load should hit weight cache", (model_reports[1].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) return 1;
    if (!expect_true("second model load should reuse winner", (model_reports[1].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("second model load should report reuse selection", model_reports[1].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;
    if (!expect_true("second model load should report elapsed time", model_reports[1].elapsed_us > 0)) return 1;

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

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens", status, MIZU_STATUS_OK)) return 1;

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
    if (!expect_status("attach modal input", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    prefill_buffer.report_count = 0;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill should reject undersized multimodal report buffer", status, MIZU_STATUS_BUFFER_TOO_SMALL)) return 1;
    if (!expect_true("prefill should report two required stages", prefill_buffer.report_count == 2)) return 1;

    prefill_buffer.report_capacity = 2;
    prefill_buffer.report_count = 0;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("prefill report count should be 2", prefill_buffer.report_count == 2)) return 1;
    if (!expect_true("projector report should stamp struct_size", prefill_reports[0].struct_size == sizeof(prefill_reports[0]))) return 1;
    if (!expect_true("prefill report should stamp struct_size", prefill_reports[1].struct_size == sizeof(prefill_reports[1]))) return 1;
    if (!expect_true("first prefill report should be projector", prefill_reports[0].stage_kind == MIZU_STAGE_PROJECTOR)) return 1;
    if (!expect_true("second prefill report should be prefill", prefill_reports[1].stage_kind == MIZU_STAGE_PREFILL)) return 1;
    if (!expect_true("projector plan id should be nonzero", prefill_reports[0].plan_id != 0)) return 1;
    if (!expect_true("prefill plan id should be nonzero", prefill_reports[1].plan_id != 0)) return 1;
    if (!expect_true("projector should report elapsed time", prefill_reports[0].elapsed_us > 0)) return 1;
    if (!expect_true("prefill should report elapsed time", prefill_reports[1].elapsed_us > 0)) return 1;
    if (!expect_true("first projector should be multimodal cache miss", (prefill_reports[0].cache_flags & MIZU_CACHE_FLAG_MM_HIT) == 0)) return 1;
    if (!expect_true("first prefill should be plan cache miss", (prefill_reports[1].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) == 0)) return 1;
    status = mizu_session_get_last_report(session, &prefill_reports[1]);
    if (!expect_status("prefill report entry should be reusable with get_last_report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("reused prefill report entry should stay prefill", prefill_reports[1].stage_kind == MIZU_STAGE_PREFILL)) return 1;

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
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    decode_buffer.report_count = 0;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("decode report count should be 1", decode_buffer.report_count == 1)) return 1;
    if (!expect_true("decode stage kind should be decode", decode_reports[0].stage_kind == MIZU_STAGE_DECODE)) return 1;
    if (!expect_true("decode plan id should be nonzero", decode_reports[0].plan_id != 0)) return 1;
    if (!expect_true("decode should report elapsed time", decode_reports[0].elapsed_us > 0)) return 1;
    if (!expect_true("first decode should be plan cache miss", (decode_reports[0].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) == 0)) return 1;

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("park stage kind should be park", park_reports[0].stage_kind == MIZU_STAGE_PARK)) return 1;
    if (!expect_true("park plan id should be nonzero", park_reports[0].plan_id != 0)) return 1;
    if (!expect_true("park should report elapsed time", park_reports[0].elapsed_us > 0)) return 1;
    if (!expect_true("first park should be session cache miss", (park_reports[0].cache_flags & MIZU_CACHE_FLAG_SESSION_HIT) == 0)) return 1;

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("resume stage kind should be resume", resume_reports[0].stage_kind == MIZU_STAGE_RESUME)) return 1;
    if (!expect_true("resume plan id should be nonzero", resume_reports[0].plan_id != 0)) return 1;
    if (!expect_true("resume should report elapsed time", resume_reports[0].elapsed_us > 0)) return 1;
    if (!expect_true("resume should hit session cache", (resume_reports[0].cache_flags & MIZU_CACHE_FLAG_SESSION_HIT) != 0)) return 1;
    if (!expect_true("resume should reuse winner", (resume_reports[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;

    status = mizu_session_open(model, &session_config, &session_cached);
    if (!expect_status("session open cached", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_attach_tokens(session_cached, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens cached", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_cached, &modal_input);
    if (!expect_status("attach modal input cached", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_cached.struct_size = sizeof(prefill_buffer_cached);
    prefill_buffer_cached.reports = prefill_reports_cached;
    prefill_buffer_cached.report_capacity = 2;
    prefill_buffer_cached.report_count = 0;

    status = mizu_session_prefill(session_cached, &prefill_buffer_cached);
    if (!expect_status("prefill cached", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cached projector should hit multimodal cache", (prefill_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_MM_HIT) != 0)) return 1;
    if (!expect_true("cached prefill should hit plan cache", (prefill_reports_cached[1].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) return 1;
    if (!expect_true("cached projector should reuse winner", (prefill_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("cached prefill should reuse winner", (prefill_reports_cached[1].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("cached projector should report reuse selection", prefill_reports_cached[0].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;
    if (!expect_true("cached prefill should report reuse selection", prefill_reports_cached[1].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;

    decode_result.token_buffer = decode_tokens_cached;
    decode_buffer_cached.struct_size = sizeof(decode_buffer_cached);
    decode_buffer_cached.reports = decode_reports_cached;
    decode_buffer_cached.report_capacity = 1;
    decode_buffer_cached.report_count = 0;

    status = mizu_session_decode_step(session_cached, &decode_options, &decode_result, &decode_buffer_cached);
    if (!expect_status("decode cached", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cached decode should hit plan cache", (decode_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) return 1;
    if (!expect_true("cached decode should reuse winner", (decode_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("cached decode should report reuse selection", decode_reports_cached[0].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;

    park_buffer_cached.struct_size = sizeof(park_buffer_cached);
    park_buffer_cached.reports = park_reports_cached;
    park_buffer_cached.report_capacity = 1;
    park_buffer_cached.report_count = 0;

    status = mizu_session_park(session_cached, &park_buffer_cached);
    if (!expect_status("park cached", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cached park should hit session cache", (park_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_SESSION_HIT) != 0)) return 1;
    if (!expect_true("cached park should reuse winner", (park_reports_cached[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("cached park should report reuse selection", park_reports_cached[0].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_close(session_cached);
    if (!expect_status("session close cached", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_cached);
    if (!expect_status("model close cached", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    status = mizu_runtime_create(&runtime_config, &runtime_fresh);
    if (!expect_status("fresh runtime create", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_fresh, &model_config, &model_fresh);
    if (!expect_status("fresh runtime model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_fresh, &fresh_model_report);
    if (!expect_status("fresh runtime model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("fresh runtime should start with cold weight miss", (fresh_model_report.cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) == 0)) return 1;
    status = mizu_model_close(model_fresh);
    if (!expect_status("fresh runtime model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_fresh);
    if (!expect_status("fresh runtime destroy", status, MIZU_STATUS_OK)) return 1;

    runtime_config.exploration_budget = 2;
    status = mizu_runtime_create(&runtime_config, &runtime_explore);
    if (!expect_status("explore runtime create", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime_explore, &explore_model_config, &model_explore_a);
    if (!expect_status("explore model open 1", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_explore_a, &explore_reports[0]);
    if (!expect_status("explore report 1", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("explore model 1 should be exploratory", explore_reports[0].selection_mode == MIZU_SELECTION_MODE_EXPLORATORY)) return 1;
    if (!expect_true("explore model 1 should be weight miss", (explore_reports[0].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) == 0)) return 1;
    if (!expect_true("explore model 1 should route to ANE first", explore_reports[0].execution_route == MIZU_EXEC_ROUTE_ANE)) return 1;

    status = mizu_model_open(runtime_explore, &explore_model_config, &model_explore_b);
    if (!expect_status("explore model open 2", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_explore_b, &explore_reports[1]);
    if (!expect_status("explore report 2", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("explore model 2 should be exploratory", explore_reports[1].selection_mode == MIZU_SELECTION_MODE_EXPLORATORY)) return 1;
    if (!expect_true("explore model 2 should be weight miss", (explore_reports[1].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) == 0)) return 1;
    if (!expect_true("explore model plans should differ", explore_reports[0].plan_id != explore_reports[1].plan_id)) return 1;
    if (!expect_true("explore model 2 should route to Metal second", explore_reports[1].execution_route == MIZU_EXEC_ROUTE_METAL)) return 1;

    status = mizu_model_open(runtime_explore, &explore_model_config, &model_explore_c);
    if (!expect_status("explore model open 3", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_explore_c, &explore_reports[2]);
    if (!expect_status("explore report 3", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("explore model 3 should reuse winner", explore_reports[2].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;
    if (!expect_true("explore model 3 should hit weight cache", (explore_reports[2].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) return 1;
    if (!expect_true("explore model 3 should mark winner reuse", (explore_reports[2].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("explore model 3 should report elapsed time", explore_reports[2].elapsed_us > 0)) return 1;
    if (!expect_true("explore model 3 should reuse one explored plan", explore_reports[2].plan_id == explore_reports[0].plan_id || explore_reports[2].plan_id == explore_reports[1].plan_id)) return 1;

    status = mizu_model_close(model_explore_a);
    if (!expect_status("explore model close 1", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_explore_b);
    if (!expect_status("explore model close 2", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_explore_c);
    if (!expect_status("explore model close 3", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_explore);
    if (!expect_status("explore runtime destroy", status, MIZU_STATUS_OK)) return 1;

    command_status = system("rm -rf /tmp/mizu_stage_report_persist && mkdir -p /tmp/mizu_stage_report_persist");
    if (!expect_true("persist root setup should succeed", command_status == 0)) return 1;

    persist_runtime_config = runtime_config;
    persist_runtime_config.cache_root_z = persist_root;
    persist_runtime_config.exploration_budget = 2;

    status = mizu_runtime_create(&persist_runtime_config, &runtime_persist_a);
    if (!expect_status("persist runtime create a", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime_persist_a, &explore_model_config, &model_persist_a);
    if (!expect_status("persist model open a", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_persist_a, &persist_reports[0]);
    if (!expect_status("persist report a", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime_persist_a, &explore_model_config, &model_persist_b);
    if (!expect_status("persist model open b", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_persist_b, &persist_reports[1]);
    if (!expect_status("persist report b", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime_persist_a, &explore_model_config, &model_persist_c);
    if (!expect_status("persist model open c", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_persist_c, &persist_reports[2]);
    if (!expect_status("persist report c", status, MIZU_STATUS_OK)) return 1;

    if (!expect_true("persist run should explore ANE first", persist_reports[0].execution_route == MIZU_EXEC_ROUTE_ANE)) return 1;
    if (!expect_true("persist run should explore Metal second", persist_reports[1].execution_route == MIZU_EXEC_ROUTE_METAL)) return 1;
    if (!expect_true("persist run should reuse by third open", persist_reports[2].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;

    status = mizu_session_open(model_persist_c, &session_config, &session_persist_a);
    if (!expect_status("persist session open a", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_persist_a, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("persist attach tokens a", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_persist_a, &modal_input);
    if (!expect_status("persist attach modal a", status, MIZU_STATUS_OK)) return 1;

    persist_prefill_buffer.struct_size = sizeof(persist_prefill_buffer);
    persist_prefill_buffer.reports = persist_prefill_reports;
    persist_prefill_buffer.report_capacity = 2;
    persist_prefill_buffer.report_count = 0;

    status = mizu_session_prefill(session_persist_a, &persist_prefill_buffer);
    if (!expect_status("persist prefill a", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persist prefill a projector should explore ANE first", persist_prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_ANE)) return 1;
    if (!expect_true("persist prefill a main stage should explore ANE first", persist_prefill_reports[1].execution_route == MIZU_EXEC_ROUTE_ANE)) return 1;

    persist_decode_buffer.struct_size = sizeof(persist_decode_buffer);
    persist_decode_buffer.reports = persist_decode_reports;
    persist_decode_buffer.report_capacity = 1;
    persist_decode_buffer.report_count = 0;

    decode_result.token_buffer = decode_tokens;
    status = mizu_session_decode_step(session_persist_a, &decode_options, &decode_result, &persist_decode_buffer);
    if (!expect_status("persist decode a", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persist decode a should explore ANE first", persist_decode_reports[0].execution_route == MIZU_EXEC_ROUTE_ANE)) return 1;

    status = mizu_session_close(session_persist_a);
    if (!expect_status("persist session close a", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_open(model_persist_c, &session_config, &session_persist_a);
    if (!expect_status("persist session open b", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_persist_a, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("persist attach tokens b", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_persist_a, &modal_input);
    if (!expect_status("persist attach modal b", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_prefill(session_persist_a, &persist_prefill_buffer);
    if (!expect_status("persist prefill b", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persist prefill b projector should explore Metal second", persist_prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_METAL)) return 1;
    if (!expect_true("persist prefill b main stage should explore Metal second", persist_prefill_reports[1].execution_route == MIZU_EXEC_ROUTE_METAL)) return 1;

    decode_result.token_buffer = decode_tokens;
    status = mizu_session_decode_step(session_persist_a, &decode_options, &decode_result, &persist_decode_buffer);
    if (!expect_status("persist decode b", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persist decode b should explore Metal second", persist_decode_reports[0].execution_route == MIZU_EXEC_ROUTE_METAL)) return 1;

    status = mizu_session_close(session_persist_a);
    if (!expect_status("persist session close b", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_close(model_persist_a);
    if (!expect_status("persist model close a", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_persist_b);
    if (!expect_status("persist model close b", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_persist_c);
    if (!expect_status("persist model close c", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_persist_a);
    if (!expect_status("persist runtime destroy a", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persisted artifact cache should include weight metadata",
                     file_contains_substring(persist_artifact_cache_path, "meta weight"))) return 1;
    if (!expect_true("persisted artifact cache should include multimodal metadata",
                     file_contains_substring(persist_artifact_cache_path, "meta mm"))) return 1;
    if (!expect_true("persisted artifact cache should include plan metadata",
                     file_contains_substring(persist_artifact_cache_path, "meta plan"))) return 1;
    if (!expect_true("persisted artifact cache should include ANE weight path",
                     file_contains_substring(persist_artifact_cache_path, "artifacts/apple/ane/weights/"))) return 1;
    if (!expect_true("persisted artifact cache should include Metal prefill plan path",
                     file_contains_substring(persist_artifact_cache_path, "artifacts/apple/metal/plans/prefill/"))) return 1;
    if (!expect_true("persisted artifact cache should include the ANE weight-pack format label",
                     file_contains_substring(persist_artifact_cache_path, "apple_ane_bf16_weight_pack_v1"))) return 1;
    if (!expect_true("persisted artifact cache should include the ANE projector format label",
                     file_contains_substring(persist_artifact_cache_path, "apple_ane_u8_bf16_projector_plan_v1"))) return 1;
    if (!expect_true("persisted artifact cache should include the ANE prefill format label",
                     file_contains_substring(persist_artifact_cache_path, "apple_ane_bf16_prefill_plan_v1"))) return 1;
    if (!expect_true("persisted artifact cache should include the Metal prefill format label",
                     file_contains_substring(persist_artifact_cache_path, "apple_metal_bf16_prefill_plan_v1"))) return 1;

    status = mizu_runtime_create(&persist_runtime_config, &runtime_persist_b);
    if (!expect_status("persist runtime create b", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_persist_b, &explore_model_config, &model_persist_reuse);
    if (!expect_status("persist model open reuse", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_persist_reuse, &persist_reports[3]);
    if (!expect_status("persist report reuse", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persisted winner should be reused on next runtime", persist_reports[3].selection_mode == MIZU_SELECTION_MODE_REUSE)) return 1;
    if (!expect_true("persisted weight artifact should hit on next runtime", (persist_reports[3].cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) return 1;
    if (!expect_true("persisted winner should mark winner reuse", (persist_reports[3].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("persisted winner should match one explored plan", persist_reports[3].plan_id == persist_reports[0].plan_id || persist_reports[3].plan_id == persist_reports[1].plan_id)) return 1;
    if (!expect_true("persisted reuse should preserve explored route", persist_reports[3].execution_route == persist_reports[0].execution_route || persist_reports[3].execution_route == persist_reports[1].execution_route)) return 1;

    status = mizu_session_open(model_persist_reuse, &session_config, &session_persist_b);
    if (!expect_status("persist session open reuse", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_persist_b, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("persist attach tokens reuse", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_persist_b, &modal_input);
    if (!expect_status("persist attach modal reuse", status, MIZU_STATUS_OK)) return 1;

    persist_prefill_buffer_reuse.struct_size = sizeof(persist_prefill_buffer_reuse);
    persist_prefill_buffer_reuse.reports = persist_prefill_reports_reuse;
    persist_prefill_buffer_reuse.report_capacity = 2;
    persist_prefill_buffer_reuse.report_count = 0;

    status = mizu_session_prefill(session_persist_b, &persist_prefill_buffer_reuse);
    if (!expect_status("persist prefill reuse", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persisted projector artifact should hit", (persist_prefill_reports_reuse[0].cache_flags & MIZU_CACHE_FLAG_MM_HIT) != 0)) return 1;
    if (!expect_true("persisted prefill artifact should hit", (persist_prefill_reports_reuse[1].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) return 1;
    if (!expect_true("persisted projector should reuse winner", (persist_prefill_reports_reuse[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;
    if (!expect_true("persisted prefill should reuse winner", (persist_prefill_reports_reuse[1].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;

    persist_decode_buffer_reuse.struct_size = sizeof(persist_decode_buffer_reuse);
    persist_decode_buffer_reuse.reports = persist_decode_reports_reuse;
    persist_decode_buffer_reuse.report_capacity = 1;
    persist_decode_buffer_reuse.report_count = 0;

    decode_result.token_buffer = decode_tokens_cached;
    status = mizu_session_decode_step(session_persist_b, &decode_options, &decode_result, &persist_decode_buffer_reuse);
    if (!expect_status("persist decode reuse", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("persisted decode artifact should hit", (persist_decode_reports_reuse[0].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) return 1;
    if (!expect_true("persisted decode should reuse winner", (persist_decode_reports_reuse[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) return 1;

    status = mizu_session_close(session_persist_b);
    if (!expect_status("persist session close reuse", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_persist_reuse);
    if (!expect_status("persist model close reuse", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_persist_b);
    if (!expect_status("persist runtime destroy b", status, MIZU_STATUS_OK)) return 1;
    command_status = system("rm -rf /tmp/mizu_stage_report_persist");
    if (!expect_true("persist root cleanup should succeed", command_status == 0)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    puts("test_stage_reports: PASS");
    return 0;
}
