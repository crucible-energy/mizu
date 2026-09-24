#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "mizu.h"

extern char *mkdtemp(char *template);

static int path_exists(const char *path) {
    struct stat info;
    return path != NULL && stat(path, &info) == 0;
}

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

static int run_command(const char *label, const char *command) {
    int status;

    status = system(command);
    if (status != 0) {
        fprintf(stderr, "%s failed with status %d\n", label, status);
        return 0;
    }
    return 1;
}

static int open_model_smoke(mizu_runtime_t *runtime, const char *model_root, mizu_model_t **out_model) {
    mizu_model_open_config_t model_config;
    mizu_execution_report_t model_report;
    mizu_status_code_t status;

    memset(&model_config, 0, sizeof(model_config));
    memset(&model_report, 0, sizeof(model_report));

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = model_root;
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_CUDA;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, out_model);
    if (!expect_status("qwench model open", status, MIZU_STATUS_OK)) return 0;

    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(*out_model, &model_report);
    if (!expect_status("qwench model report", status, MIZU_STATUS_OK)) return 0;
    return expect_true("qwench model load should route to CUDA", model_report.execution_route == MIZU_EXEC_ROUTE_CUDA);
}

static int run_qwen_session_smoke(mizu_model_t *model) {
    mizu_session_t *session = NULL;
    mizu_session_config_t session_config;
    mizu_modal_input_desc_t modal_input;
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t decode_buffer;
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t decode_reports[1];
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    int32_t tokens[3] = {101, 202, 303};
    int32_t decode_tokens[1] = {0};
    uint8_t image_bytes[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    mizu_status_code_t status;
    int ok = 1;

    memset(&session_config, 0, sizeof(session_config));
    memset(&modal_input, 0, sizeof(modal_input));
    memset(&prefill_buffer, 0, sizeof(prefill_buffer));
    memset(&decode_buffer, 0, sizeof(decode_buffer));
    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(&decode_options, 0, sizeof(decode_options));
    memset(&decode_result, 0, sizeof(decode_result));

    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 16;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("qwench session open", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("qwench attach tokens", status, MIZU_STATUS_OK)) ok = 0;

    modal_input.struct_size = sizeof(modal_input);
    modal_input.slot_name_z = "image";
    modal_input.placeholder_ordinal = 1;
    modal_input.modality_kind = MIZU_MODALITY_KIND_IMAGE;
    modal_input.storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES;
    modal_input.dtype = MIZU_DTYPE_U8;
    modal_input.data = image_bytes;
    modal_input.byte_count = sizeof(image_bytes);
    modal_input.lifetime_policy = MIZU_LIFETIME_POLICY_COPY;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("qwench attach modal", status, MIZU_STATUS_OK)) ok = 0;

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 2;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("qwench prefill", status, MIZU_STATUS_OK)) ok = 0;
    if (ok) {
        ok = expect_true("qwench projector should route to CUDA",
                         prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA) && ok;
        ok = expect_true("qwench prefill should route to CUDA",
                         prefill_reports[1].execution_route == MIZU_EXEC_ROUTE_CUDA) && ok;
    }

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = decode_tokens;
    decode_result.token_capacity = 1;
    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("qwench decode", status, MIZU_STATUS_OK)) ok = 0;
    if (ok) {
        ok = expect_true("qwench decode should route to CUDA",
                         decode_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA) && ok;
        ok = expect_true("qwench decode should emit one token", decode_result.token_count == 1U) && ok;
    }

    status = mizu_session_close(session);
    if (!expect_status("qwench session close", status, MIZU_STATUS_OK)) ok = 0;
    return ok;
}

int main(void) {
    const char *home = getenv("HOME");
    const char *importer_path = getenv("MIZU_GGUF_IMPORTER");
    const char *fixture_writer = getenv("MIZU_IMPORTER_TEST_BIN");
    char persist_root[] = "/tmp/mizu_qwench_gguf_cuda_smoke.XXXXXX";
    char cache_root[4096];
    char weights_root[4096];
    char projector_root[4096];
    char qwen_bundle_root[4096];
    char gemma_bundle_root[4096];
    char qwen_import_log[4096];
    char gemma_import_log[4096];
    char qwen_model_path[1024];
    char qwen_projector_path[1024];
    char gemma_model_path[1024];
    char command[4096];
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *qwen_model = NULL;
    mizu_model_t *gemma_model = NULL;
    mizu_runtime_config_t runtime_config;
    mizu_status_code_t status;
    int command_status;
    int ok = 1;

    if (home == NULL || home[0] == '\0') {
        printf("test_qwench_gguf_cuda_smoke: SKIP (HOME is not set)\n");
        return 0;
    }

    snprintf(qwen_model_path, sizeof(qwen_model_path), "%s/.qwench/models/qwen3.5-9b-instruct-q4_k_m.gguf", home);
    snprintf(qwen_projector_path, sizeof(qwen_projector_path), "%s/.qwench/models/mmproj-Qwen_Qwen3.5-9B-f16.gguf", home);
    snprintf(gemma_model_path, sizeof(gemma_model_path), "%s/.qwench/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf", home);

    if (!path_exists(qwen_model_path) || !path_exists(qwen_projector_path) || !path_exists(gemma_model_path)) {
        printf("test_qwench_gguf_cuda_smoke: SKIP (Qwench GGUF assets not found)\n");
        return 0;
    }
    if (importer_path == NULL || importer_path[0] == '\0' || fixture_writer == NULL || fixture_writer[0] == '\0') {
        fprintf(stderr, "MIZU_GGUF_IMPORTER and MIZU_IMPORTER_TEST_BIN are required for the Qwench import smoke\n");
        return 1;
    }

    if (mkdtemp(persist_root) == NULL ||
        snprintf(cache_root, sizeof(cache_root), "%s/cache", persist_root) >= (int)sizeof(cache_root) ||
        snprintf(weights_root, sizeof(weights_root), "%s/artifacts/cuda/cuda/weights", cache_root) >= (int)sizeof(weights_root) ||
        snprintf(projector_root, sizeof(projector_root), "%s/artifacts/cuda/cuda/projector", cache_root) >= (int)sizeof(projector_root) ||
        snprintf(qwen_bundle_root, sizeof(qwen_bundle_root), "%s/qwen35-9b", persist_root) >= (int)sizeof(qwen_bundle_root) ||
        snprintf(gemma_bundle_root, sizeof(gemma_bundle_root), "%s/gemma4-26b", persist_root) >= (int)sizeof(gemma_bundle_root) ||
        snprintf(qwen_import_log, sizeof(qwen_import_log), "%s/qwen_import.log", persist_root) >= (int)sizeof(qwen_import_log) ||
        snprintf(gemma_import_log, sizeof(gemma_import_log), "%s/gemma_import.log", persist_root) >= (int)sizeof(gemma_import_log) ||
        mkdir(cache_root, 0700) != 0) {
        perror("qwench smoke temporary root setup");
        return 1;
    }
    printf("test_qwench_gguf_cuda_smoke: artifacts: %s\n", persist_root);

    snprintf(command, sizeof(command),
             "'%s' '%s' --projector-gguf '%s' "
             "--output-root '%s' --force >'%s'",
             importer_path, qwen_model_path, qwen_projector_path, qwen_bundle_root, qwen_import_log);
    if (!run_command("qwench qwen import", command)) return 1;

    snprintf(command, sizeof(command),
             "'%s' '%s' --output-root '%s' --force >'%s'",
             importer_path, gemma_model_path, gemma_bundle_root, gemma_import_log);
    if (!run_command("qwench gemma import", command)) return 1;

    snprintf(command, sizeof(command),
             "awk -F'|' 'NF==9 && $8 ~ /^[0-9][0-9]*$/ && $8 > 0 { found=1 } END { exit found ? 0 : 1 }' '%s/mizu_import/gguf_tensors.tsv'",
             qwen_bundle_root);
    command_status = system(command);
    if (!expect_true("qwench import should record absolute GGUF source offsets", command_status == 0)) return 1;

    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE\n");
        return 1;
    }

    memset(&runtime_config, 0, sizeof(runtime_config));
    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = cache_root;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("qwench runtime create", status, MIZU_STATUS_OK)) return 1;

    if (!open_model_smoke(runtime, qwen_bundle_root, &qwen_model)) ok = 0;
    if (ok && !run_qwen_session_smoke(qwen_model)) ok = 0;
    if (qwen_model != NULL) {
        status = mizu_model_close(qwen_model);
        if (!expect_status("qwench qwen model close", status, MIZU_STATUS_OK)) ok = 0;
    }

    if (ok && !open_model_smoke(runtime, gemma_bundle_root, &gemma_model)) ok = 0;
    if (gemma_model != NULL) {
        status = mizu_model_close(gemma_model);
        if (!expect_status("qwench gemma model close", status, MIZU_STATUS_OK)) ok = 0;
    }

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("qwench runtime destroy", status, MIZU_STATUS_OK)) ok = 0;
    if (!ok) return 1;

    snprintf(command, sizeof(command), "grep -R \"storage=q4_k\" '%s' >/dev/null", weights_root);
    command_status = system(command);
    if (!expect_true("qwench CUDA weight artifacts should retain q4_k storage", command_status == 0)) return 1;
    snprintf(command, sizeof(command), "grep -R \"storage=iq2_xxs\" '%s' >/dev/null", weights_root);
    command_status = system(command);
    if (!expect_true("qwench CUDA weight artifacts should retain Gemma quantized storage", command_status == 0)) return 1;
    snprintf(command, sizeof(command), "grep -R -E \"source_offset=[1-9][0-9]*\" '%s' >/dev/null", weights_root);
    command_status = system(command);
    if (!expect_true("qwench CUDA weight artifacts should retain per-tensor source offsets", command_status == 0)) return 1;
    snprintf(command, sizeof(command), "'%s' --check-packbuffer-tree '%s'", fixture_writer, weights_root);
    if (!run_command("qwench packbuffer inspection", command)) return 1;
    snprintf(command, sizeof(command), "grep -R \"mm.0.weight\" '%s' >/dev/null", weights_root);
    command_status = system(command);
    if (!expect_true("qwench CUDA decoder weight pack should exclude mmproj tensors", command_status != 0)) return 1;
    snprintf(command, sizeof(command), "grep -R -E \"projector_bytes=[1-9]\" '%s' >/dev/null", projector_root);
    command_status = system(command);
    if (!expect_true("qwench CUDA projector artifact should carry projector byte lineage", command_status == 0)) return 1;

    printf("test_qwench_gguf_cuda_smoke: PASS\n");
    return 0;
}
