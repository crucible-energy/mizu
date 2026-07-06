#define _POSIX_C_SOURCE 200809L

#include <glob.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mizu.h"

enum {
    CHECKPOINT_MUTATION_DELETE = 1,
    CHECKPOINT_MUTATION_CORRUPT = 2
};

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

    *required_bytes = 0;
    memset(buffer, 0, buffer_size);
    status = mizu_runtime_copy_last_error(runtime, buffer, buffer_size, required_bytes);
    return expect_status("copy runtime error", status, MIZU_STATUS_OK);
}

static int set_fixture_backend_env(void) {
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
    return 1;
}

static void clear_fixture_backend_env(void) {
    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
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

static int find_single_plan_artifact(
    const char *cache_root,
    const char *stage_name,
    char *artifact_path,
    size_t artifact_path_size
) {
    char pattern[1024];
    glob_t matches;
    int status;
    size_t path_len;

    memset(&matches, 0, sizeof(matches));
    if (snprintf(pattern, sizeof(pattern), "%s/artifacts/apple/*/plans/%s/*.plan",
                 cache_root, stage_name) < 0) {
        fprintf(stderr, "failed to build plan artifact search pattern\n");
        return 0;
    }

    status = glob(pattern, 0, NULL, &matches);
    if (status != 0) {
        globfree(&matches);
        fprintf(stderr, "expected a persisted %s plan under %s\n", stage_name, cache_root);
        return 0;
    }
    if (!expect_true("exactly one plan artifact should exist", matches.gl_pathc == 1)) {
        globfree(&matches);
        return 0;
    }

    path_len = strlen(matches.gl_pathv[0]);
    if (!expect_true("plan artifact path should fit buffer", path_len + 1 <= artifact_path_size)) {
        globfree(&matches);
        return 0;
    }
    memcpy(artifact_path, matches.gl_pathv[0], path_len + 1);
    globfree(&matches);
    return 1;
}

static int find_single_session_artifact(const char *cache_root, char *artifact_path, size_t artifact_path_size) {
    char pattern[1024];
    glob_t matches;
    int status;
    size_t path_len;

    memset(&matches, 0, sizeof(matches));
    if (snprintf(pattern, sizeof(pattern), "%s/artifacts/*/*/sessions/*.session", cache_root) < 0) {
        fprintf(stderr, "failed to build artifact search pattern\n");
        return 0;
    }

    status = glob(pattern, 0, NULL, &matches);
    if (status != 0) {
        globfree(&matches);
        fprintf(stderr, "expected a persisted session checkpoint under %s\n", cache_root);
        return 0;
    }
    if (!expect_true("exactly one session checkpoint should exist", matches.gl_pathc == 1)) {
        globfree(&matches);
        return 0;
    }

    path_len = strlen(matches.gl_pathv[0]);
    if (!expect_true("checkpoint artifact path should fit buffer", path_len + 1 <= artifact_path_size)) {
        globfree(&matches);
        return 0;
    }
    memcpy(artifact_path, matches.gl_pathv[0], path_len + 1);
    globfree(&matches);
    return 1;
}

static int mutate_checkpoint_artifact(const char *artifact_path, int mutation_kind) {
    FILE *file;

    switch (mutation_kind) {
        case CHECKPOINT_MUTATION_DELETE:
            return expect_true("checkpoint delete should succeed", unlink(artifact_path) == 0);
        case CHECKPOINT_MUTATION_CORRUPT:
            file = fopen(artifact_path, "w");
            if (file == NULL) {
                fprintf(stderr, "failed to open checkpoint for corruption: %s\n", artifact_path);
                return 0;
            }
            if (fputs("corrupt checkpoint payload\n", file) == EOF) {
                fclose(file);
                fprintf(stderr, "failed to corrupt checkpoint payload: %s\n", artifact_path);
                return 0;
            }
            if (fclose(file) != 0) {
                fprintf(stderr, "failed to close corrupted checkpoint payload: %s\n", artifact_path);
                return 0;
            }
            return 1;
        default:
            fprintf(stderr, "unknown checkpoint mutation kind %d\n", mutation_kind);
            return 0;
    }
}

static int corrupt_text_artifact(const char *artifact_path, const char *text) {
    FILE *file;

    file = fopen(artifact_path, "w");
    if (file == NULL) {
        fprintf(stderr, "failed to open artifact for corruption: %s\n", artifact_path);
        return 0;
    }
    if (fputs(text, file) == EOF) {
        fclose(file);
        fprintf(stderr, "failed to corrupt artifact payload: %s\n", artifact_path);
        return 0;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "failed to close corrupted artifact payload: %s\n", artifact_path);
        return 0;
    }
    return 1;
}

static int run_restore_failure_case(const char *cache_root, int mutation_kind, const char *label_prefix) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int32_t tokens[3] = {1, 2, 3};
    mizu_execution_report_t prefill_reports[1];
    mizu_execution_report_t park_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_execution_report_t resume_reports[1];
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    size_t required_bytes;
    char error_buffer[256];
    char artifact_path[1024];
    char label[128];

    if (!reset_cache_root(cache_root)) return 0;

    init_runtime_config(&runtime_config, cache_root);
    init_model_config(&model_config);
    init_session_config(&session_config);

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create", status, MIZU_STATUS_OK)) return 0;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("session open", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("attach tokens", status, MIZU_STATUS_OK)) return 0;

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
    if (!expect_status("prefill", status, MIZU_STATUS_OK)) return 0;

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("park", status, MIZU_STATUS_OK)) return 0;

    if (!find_single_session_artifact(cache_root, artifact_path, sizeof(artifact_path))) return 0;
    if (!mutate_checkpoint_artifact(artifact_path, mutation_kind)) return 0;

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status(label_prefix, status, MIZU_STATUS_INVALID_STATE)) return 0;

    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 0;
    if (!expect_true("failed resume should publish non-empty runtime error", required_bytes > 1)) return 0;
    if (!expect_true("failed resume error should mention checkpoint restore",
                     strstr(error_buffer, "checkpoint restore failed") != NULL)) return 0;

    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after failed resume", status, MIZU_STATUS_OK)) return 0;
    if (!expect_true("failed resume should preserve live context flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 0;
    if (!expect_true("failed resume should preserve parked flag",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) return 0;
    if (!expect_true("failed resume should preserve kv token count", session_info.kv_token_count == 3)) return 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (snprintf(label, sizeof(label), "%s repeat", label_prefix) < 0) {
        fprintf(stderr, "failed to build repeated resume label\n");
        return 0;
    }
    if (!expect_status(label, status, MIZU_STATUS_INVALID_STATE)) return 0;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 0;

    return 1;
}

static int run_prefill_plan_failure_case(const char *cache_root) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int32_t tokens[3] = {1, 2, 3};
    mizu_execution_report_t prefill_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    size_t required_bytes;
    char error_buffer[256];
    char artifact_path[1024];

    if (!reset_cache_root(cache_root)) return 0;

    init_runtime_config(&runtime_config, cache_root);
    init_model_config(&model_config);
    init_session_config(&session_config);

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("prefill runtime create", status, MIZU_STATUS_OK)) return 0;
    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("prefill model open", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("prefill session open seed", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("prefill attach tokens seed", status, MIZU_STATUS_OK)) return 0;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill seed run", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_close(session);
    if (!expect_status("prefill seed session close", status, MIZU_STATUS_OK)) return 0;
    session = NULL;

    if (!find_single_plan_artifact(cache_root, "prefill", artifact_path, sizeof(artifact_path))) return 0;
    if (!corrupt_text_artifact(artifact_path, "corrupt prefill plan payload\n")) return 0;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("prefill session open failing run", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("prefill attach tokens failing run", status, MIZU_STATUS_OK)) return 0;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("prefill should fail when plan payload is corrupt", status, MIZU_STATUS_INVALID_STATE)) return 0;

    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 0;
    if (!expect_true("failed prefill should publish non-empty runtime error", required_bytes > 1)) return 0;
    if (!expect_true("failed prefill error should mention prefill execution",
                     strstr(error_buffer, "prefill execution failed") != NULL)) return 0;

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after failed prefill", status, MIZU_STATUS_OK)) return 0;
    if (!expect_true("failed prefill should preserve staged token count", session_info.staged_token_count == 3)) return 0;
    if (!expect_true("failed prefill should not advance kv tokens", session_info.kv_token_count == 0)) return 0;

    status = mizu_session_close(session);
    if (!expect_status("prefill failing session close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_model_close(model);
    if (!expect_status("prefill model close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("prefill runtime destroy", status, MIZU_STATUS_OK)) return 0;

    return 1;
}

static int run_decode_plan_failure_case(const char *cache_root) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int32_t tokens[3] = {1, 2, 3};
    int32_t decoded_token = 0;
    mizu_execution_report_t prefill_reports[1];
    mizu_execution_report_t decode_reports[1];
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t decode_buffer;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    size_t required_bytes;
    char error_buffer[256];
    char artifact_path[1024];

    if (!reset_cache_root(cache_root)) return 0;

    init_runtime_config(&runtime_config, cache_root);
    init_model_config(&model_config);
    init_session_config(&session_config);

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("decode runtime create", status, MIZU_STATUS_OK)) return 0;
    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("decode model open", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("decode session open seed", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("decode attach tokens seed", status, MIZU_STATUS_OK)) return 0;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("decode prefill seed", status, MIZU_STATUS_OK)) return 0;

    memset(decode_reports, 0, sizeof(decode_reports));
    memset(&decode_buffer, 0, sizeof(decode_buffer));
    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;

    memset(&decode_options, 0, sizeof(decode_options));
    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;

    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decoded_token;
    decode_result.token_capacity = 1;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("decode seed run", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_close(session);
    if (!expect_status("decode seed session close", status, MIZU_STATUS_OK)) return 0;
    session = NULL;

    if (!find_single_plan_artifact(cache_root, "decode", artifact_path, sizeof(artifact_path))) return 0;
    if (!corrupt_text_artifact(artifact_path, "corrupt decode plan payload\n")) return 0;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("decode session open failing run", status, MIZU_STATUS_OK)) return 0;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("decode attach tokens failing run", status, MIZU_STATUS_OK)) return 0;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 1;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("decode prefill failing run", status, MIZU_STATUS_OK)) return 0;

    memset(decode_reports, 0, sizeof(decode_reports));
    memset(&decode_buffer, 0, sizeof(decode_buffer));
    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    decoded_token = 0;
    memset(&decode_result, 0, sizeof(decode_result));
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = &decoded_token;
    decode_result.token_capacity = 1;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("decode should fail when plan payload is corrupt", status, MIZU_STATUS_INVALID_STATE)) return 0;

    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 0;
    if (!expect_true("failed decode should publish non-empty runtime error", required_bytes > 1)) return 0;
    if (!expect_true("failed decode error should mention decode execution",
                     strstr(error_buffer, "decode execution failed") != NULL)) return 0;

    memset(&session_info, 0, sizeof(session_info));
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("session info after failed decode", status, MIZU_STATUS_OK)) return 0;
    if (!expect_true("failed decode should preserve live context", (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) return 0;
    if (!expect_true("failed decode should not advance kv tokens", session_info.kv_token_count == 3)) return 0;

    status = mizu_session_close(session);
    if (!expect_status("decode failing session close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_model_close(model);
    if (!expect_status("decode model close", status, MIZU_STATUS_OK)) return 0;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("decode runtime destroy", status, MIZU_STATUS_OK)) return 0;

    return 1;
}

int main(void) {
    if (!set_fixture_backend_env()) return 1;

    if (!run_prefill_plan_failure_case("/tmp/mizu_prefill_plan_failure")) {
        clear_fixture_backend_env();
        return 1;
    }
    if (!run_decode_plan_failure_case("/tmp/mizu_decode_plan_failure")) {
        clear_fixture_backend_env();
        return 1;
    }
    if (!run_restore_failure_case("/tmp/mizu_checkpoint_restore_missing", CHECKPOINT_MUTATION_DELETE,
                                  "resume should fail when checkpoint is missing")) {
        clear_fixture_backend_env();
        return 1;
    }
    if (!run_restore_failure_case("/tmp/mizu_checkpoint_restore_corrupt", CHECKPOINT_MUTATION_CORRUPT,
                                  "resume should fail when checkpoint is corrupt")) {
        clear_fixture_backend_env();
        return 1;
    }

    clear_fixture_backend_env();
    puts("test_session_checkpoint_restore_failures: PASS");
    return 0;
}
