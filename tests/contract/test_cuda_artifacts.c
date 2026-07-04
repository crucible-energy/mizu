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

static int file_contains_substring(const char *path, const char *needle) {
    FILE *file = NULL;
    char line[2048];

    file = fopen(path, "r");
    if (file == NULL) return 0;

    while (fgets(line, sizeof(line), file) != NULL) {
      if (strstr(line, needle) != NULL) {
        fclose(file);
        return 1;
      }
    }

    fclose(file);
    return 0;
}

static int read_text_file(const char *path, char *buffer, size_t buffer_size) {
    FILE *file = NULL;
    size_t bytes_read;

    if (buffer == NULL || buffer_size < 2U) return 0;
    file = fopen(path, "rb");
    if (file == NULL) return 0;

    bytes_read = fread(buffer, 1, buffer_size - 1U, file);
    if (ferror(file) != 0) {
        fclose(file);
        return 0;
    }

    buffer[bytes_read] = '\0';
    fclose(file);
    return 1;
}

static int write_text_file(const char *path, const char *text) {
    FILE *file = NULL;
    size_t text_length;
    size_t bytes_written;

    if (path == NULL || text == NULL) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;

    text_length = strlen(text);
    bytes_written = fwrite(text, 1, text_length, file);
    fclose(file);
    return bytes_written == text_length;
}

static int read_binary_file_alloc(const char *path, unsigned char **out_buffer, size_t *out_size) {
    FILE *file = NULL;
    long file_size;
    unsigned char *buffer = NULL;
    size_t bytes_read;

    if (path == NULL || out_buffer == NULL || out_size == NULL) return 0;
    *out_buffer = NULL;
    *out_size = 0U;

    file = fopen(path, "rb");
    if (file == NULL) return 0;
    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    file_size = ftell(file);
    if (file_size <= 0L) {
        fclose(file);
        return 0;
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }

    buffer = (unsigned char *)malloc((size_t)file_size);
    if (buffer == NULL) {
        fclose(file);
        return 0;
    }
    bytes_read = fread(buffer, 1U, (size_t)file_size, file);
    fclose(file);
    if (bytes_read != (size_t)file_size) {
        free(buffer);
        return 0;
    }

    *out_buffer = buffer;
    *out_size = (size_t)file_size;
    return 1;
}

static int read_le_u32_field(const unsigned char *buffer, size_t buffer_size, size_t offset, uint32_t *out_value) {
    if (buffer == NULL || out_value == NULL || offset + 4U > buffer_size) return 0;
    *out_value = ((uint32_t)buffer[offset]) |
                 ((uint32_t)buffer[offset + 1U] << 8U) |
                 ((uint32_t)buffer[offset + 2U] << 16U) |
                 ((uint32_t)buffer[offset + 3U] << 24U);
    return 1;
}

static int read_le_i64_field(const unsigned char *buffer, size_t buffer_size, size_t offset, int64_t *out_value) {
    uint64_t raw_value = 0U;
    size_t byte_index;

    if (buffer == NULL || out_value == NULL || offset + 8U > buffer_size) return 0;
    for (byte_index = 0U; byte_index < 8U; ++byte_index) {
        raw_value |= ((uint64_t)buffer[offset + byte_index]) << (8U * byte_index);
    }
    *out_value = (int64_t)raw_value;
    return 1;
}

static int capture_first_line(const char *command, char *buffer, size_t buffer_size) {
    FILE *pipe = NULL;

    if (command == NULL || buffer == NULL || buffer_size < 2U) return 0;

    pipe = popen(command, "r");
    if (pipe == NULL) return 0;

    if (fgets(buffer, (int)buffer_size, pipe) == NULL) {
        pclose(pipe);
        return 0;
    }

    pclose(pipe);
    buffer[strcspn(buffer, "\r\n")] = '\0';
    return buffer[0] != '\0';
}

static int fragment_has_indexed_prefix(const char *fragment, const char *prefix) {
    size_t prefix_length;

    if (fragment == NULL || prefix == NULL) return 0;
    prefix_length = strlen(prefix);
    if (prefix_length == 0U) return 0;
    if (strncmp(fragment, prefix, prefix_length) != 0) return 0;
    return fragment[prefix_length] >= '0' && fragment[prefix_length] <= '9';
}

static int is_binary_sidecar_redundant_fragment(const char *fragment) {
    if (fragment == NULL || fragment[0] == '\0') return 0;

    if (fragment_has_indexed_prefix(fragment, "pack_use")) return 1;
    if (fragment_has_indexed_prefix(fragment, "pack_dispatch")) return 1;
    if (fragment_has_indexed_prefix(fragment, "pack_span")) return 1;

    if (strncmp(fragment, "pack_use_hash=", 14) == 0) return 1;
    if (strncmp(fragment, "pack_use_kind=", 14) == 0) return 1;
    if (strncmp(fragment, "pack_use_bytes=", 15) == 0) return 1;
    if (strncmp(fragment, "pack_use_count=", 15) == 0) return 1;
    if (strncmp(fragment, "pack_use_first_offset=", 22) == 0) return 1;
    if (strncmp(fragment, "pack_use_last_offset=", 21) == 0) return 1;
    if (strncmp(fragment, "pack_use_last_bytes=", 20) == 0) return 1;
    if (strncmp(fragment, "pack_dispatch_hash=", 19) == 0) return 1;
    if (strncmp(fragment, "pack_dispatch_kind=", 19) == 0) return 1;
    if (strncmp(fragment, "pack_dispatch_count=", 20) == 0) return 1;
    if (strncmp(fragment, "pack_span_root=", 15) == 0) return 1;
    if (strncmp(fragment, "pack_span_cache=", 16) == 0) return 1;
    if (strncmp(fragment, "pack_ref_hash=", 14) == 0) return 1;
    if (strncmp(fragment, "pack_ref_bytes=", 15) == 0) return 1;
    if (strncmp(fragment, "pack_ref_count=", 15) == 0) return 1;
    if (strncmp(fragment, "weight_pack_hash=", 17) == 0) return 1;
    if (strncmp(fragment, "weight_pack_bytes=", 18) == 0) return 1;
    if (strncmp(fragment, "weight_pack_count=", 18) == 0) return 1;
    if (strncmp(fragment, "pack_ref_artifact=", 18) == 0) return 1;
    if (strncmp(fragment, "pack_ref_tile_cache=", 20) == 0) return 1;
    if (strncmp(fragment, "pack_ref_tile_buffer=", 21) == 0) return 1;
    if (strncmp(fragment, "pack_dependency=", 16) == 0) return 1;
    if (strncmp(fragment, "pack_usage_buffer=", 18) == 0) return 1;
    if (strncmp(fragment, "pack_dispatch_buffer=", 21) == 0) return 1;
    if (strncmp(fragment, "pack_span_buffer=", 17) == 0) return 1;
    if (strncmp(fragment, "pack_exec_buffer=", 17) == 0) return 1;

    return 0;
}

static int reduce_plan_to_binary_sidecars(const char *input_text, char *output_text, size_t output_size) {
    const char *cursor;
    const char *separator;
    size_t output_length;
    int wrote_any;

    if (input_text == NULL || output_text == NULL || output_size < 2U) return 0;

    output_text[0] = '\0';
    output_length = 0U;
    wrote_any = 0;
    cursor = input_text;

    while (*cursor != '\0') {
        size_t fragment_length;
        char fragment[4096];
        int keep_fragment;

        separator = strchr(cursor, ';');
        fragment_length = separator == NULL ? strlen(cursor) : (size_t)(separator - cursor);
        if (fragment_length >= sizeof(fragment)) return 0;

        memcpy(fragment, cursor, fragment_length);
        fragment[fragment_length] = '\0';

        keep_fragment = wrote_any == 0 ? 1 : !is_binary_sidecar_redundant_fragment(fragment);
        if (keep_fragment && fragment[0] != '\0') {
            size_t needed = fragment_length + (wrote_any != 0 ? 1U : 0U);
            if (output_length + needed >= output_size) return 0;
            if (wrote_any != 0) output_text[output_length++] = ';';
            memcpy(output_text + output_length, fragment, fragment_length);
            output_length += fragment_length;
            output_text[output_length] = '\0';
            wrote_any = 1;
        }

        if (separator == NULL) break;
        cursor = separator + 1;
    }

    return wrote_any;
}

static int overwrite_file_prefix(const char *path, const uint8_t *bytes, size_t byte_count) {
    FILE *file = fopen(path, "r+b");
    size_t bytes_written;

    if (file == NULL) return 0;
    bytes_written = fwrite(bytes, 1, byte_count, file);
    fclose(file);
    return bytes_written == byte_count;
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_runtime_t *runtime_warm = NULL;
    mizu_runtime_t *runtime_fallback = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *model_warm = NULL;
    mizu_model_t *model_fallback = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_warm = NULL;
    mizu_session_t *session_fallback = NULL;
    mizu_status_code_t status;
    int command_status;
    int32_t tokens[3] = {11, 22, 33};
    uint8_t image_bytes[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    uint8_t mutated_prefix[64];
    int32_t decode_tokens[1] = {0};
    int32_t output_tokens[1] = {0};
    int32_t decode_tokens_warm[1] = {0};
    int32_t output_tokens_warm[1] = {0};
    int32_t decode_tokens_fallback[1] = {0};
    int32_t output_tokens_fallback[1] = {0};
    const char *persist_root = "/tmp/mizu_cuda_artifacts";
    const char *artifact_cache_path = "/tmp/mizu_cuda_artifacts/artifact_cache_v1.txt";
    const char *fixture_source_root = "tests/fixtures/models/fixture_import_bundle_tiny";
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t prefill_reports_warm[2];
    mizu_execution_report_t prefill_reports_fallback[2];
    mizu_execution_report_t decode_reports[1];
    mizu_execution_report_t decode_reports_warm[1];
    mizu_execution_report_t decode_reports_fallback[1];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t resume_reports[1];
    mizu_execution_report_t model_report;
    mizu_execution_report_t model_report_warm;
    mizu_execution_report_t model_report_fallback;
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t prefill_buffer_warm;
    mizu_report_buffer_t prefill_buffer_fallback;
    mizu_report_buffer_t decode_buffer;
    mizu_report_buffer_t decode_buffer_warm;
    mizu_report_buffer_t decode_buffer_fallback;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    mizu_session_info_t session_info_warm;
    mizu_session_info_t session_info_fallback;
    mizu_modal_input_desc_t modal_input;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_decode_result_t decode_result_warm;
    mizu_decode_result_t decode_result_fallback;
    mizu_output_buffer_t output_buffer;
    mizu_output_buffer_t output_buffer_warm;
    mizu_output_buffer_t output_buffer_fallback;
    char command_buffer[2048];
    char fixture_runtime_root[512];
    char fixture_bundle_root[640];
    char decode_plan_path[1024];
    char decode_plan_text[16384];
    char decode_binary_only_text[8192];
    char mutated_weight_path[768];
    char pack_buffer_path[1024];
    unsigned char *pack_buffer_bytes = NULL;
    size_t pack_buffer_size = 0U;
    uint32_t pack_buffer_version = 0U;
    uint32_t pack_buffer_entry_bytes = 0U;
    uint32_t pack_buffer_count = 0U;
    int64_t pack1_span_hash = 0;
    int64_t pack2_span_hash = 0;
    int64_t pack1_source_offset = -1;
    int64_t pack2_source_offset = -1;
    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(prefill_reports_warm, 0, sizeof(prefill_reports_warm));
    memset(prefill_reports_fallback, 0, sizeof(prefill_reports_fallback));
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(decode_reports_warm, 0, sizeof(decode_reports_warm));
    memset(decode_reports_fallback, 0, sizeof(decode_reports_fallback));
    memset(park_reports, 0, sizeof(park_reports));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&model_report, 0, sizeof(model_report));
    memset(&model_report_warm, 0, sizeof(model_report_warm));
    memset(&model_report_fallback, 0, sizeof(model_report_fallback));
    memset(&session_info, 0, sizeof(session_info));
    memset(&session_info_warm, 0, sizeof(session_info_warm));
    memset(&session_info_fallback, 0, sizeof(session_info_fallback));

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts && mkdir -p /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root setup should succeed", command_status == 0)) return 1;

    snprintf(fixture_runtime_root, sizeof(fixture_runtime_root), "%s/runtime_fixture_model", persist_root);
    snprintf(fixture_bundle_root, sizeof(fixture_bundle_root), "%s/mizu_import", fixture_runtime_root);
    snprintf(mutated_weight_path, sizeof(mutated_weight_path), "%s/weights/token_embeddings.bin", fixture_bundle_root);
    snprintf(command_buffer, sizeof(command_buffer), "cp -R %s %s", fixture_source_root, fixture_runtime_root);
    command_status = system(command_buffer);
    if (!expect_true("cuda fixture model copy should succeed", command_status == 0)) return 1;

    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE\n");
        return 1;
    }

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = persist_root;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_config.exploration_budget = 0;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("cuda runtime create", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = fixture_runtime_root;
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_CUDA;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("cuda model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("cuda model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda model load should route to CUDA", model_report.execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }

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
    if (!expect_status("cuda session open", status, MIZU_STATUS_OK)) return 1;
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after open", status, MIZU_STATUS_OK)) return 1;
    if (!expect_u64("cuda session should start with no flags", session_info.session_state_flags, MIZU_SESSION_STATE_NONE)) {
        return 1;
    }
    if (!expect_i64("cuda session should start with zero kv tokens", session_info.kv_token_count, 0)) return 1;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda attach tokens", status, MIZU_STATUS_OK)) return 1;

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
    if (!expect_status("cuda attach modal", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after staging", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should report pending inputs after staging",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) != 0)) {
        return 1;
    }
    if (!expect_i64("cuda session should retain staged token count", session_info.staged_token_count, 3)) return 1;
    if (!expect_true("cuda session should retain one staged modal input", session_info.staged_modal_count == 1U)) {
        return 1;
    }

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 2;
    prefill_buffer.report_count = 0;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("cuda prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda projector should route to CUDA", prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda prefill should route to CUDA", prefill_reports[1].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda multimodal prefill should report projector then prefill",
                     prefill_reports[0].stage_kind == MIZU_STAGE_PROJECTOR &&
                     prefill_reports[1].stage_kind == MIZU_STAGE_PREFILL)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should expose a live context after prefill",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda session should clear pending inputs after prefill",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) {
        return 1;
    }
    if (!expect_i64("cuda session should advance kv count after prefill", session_info.kv_token_count, 3)) return 1;
    if (!expect_i64("cuda session should clear staged token count after prefill", session_info.staged_token_count, 0)) {
        return 1;
    }
    if (!expect_true("cuda session should clear staged modal count after prefill", session_info.staged_modal_count == 0U)) {
        return 1;
    }

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
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = output_tokens;
    output_buffer.byte_capacity = sizeof(output_tokens);
    output_buffer.bytes_written = 0;
    output_buffer.output_flags = 0;

    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    decode_buffer.report_count = 0;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("cuda decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda decode should route to CUDA", decode_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda decode should emit one token", decode_result.token_count == 1U)) return 1;
    if (!expect_true("cuda decode should emit a positive placeholder token", decode_tokens[0] > 0)) {
        return 1;
    }
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("cuda read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda output buffer should report one written token", output_buffer.bytes_written == sizeof(int32_t))) {
        return 1;
    }
    if (!expect_true("cuda read output should match decode result", output_tokens[0] == decode_tokens[0])) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda session should advance kv count after decode", session_info.kv_token_count, 4)) return 1;
    if (!expect_true("cuda session should remain live after decode",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("cuda park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda park should route to CUDA", park_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should report parked after park",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda session should retain live-context flag while parked",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("cuda resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda resume should route to CUDA", resume_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should clear parked flag after resume",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) == 0)) {
        return 1;
    }
    if (!expect_true("cuda session should still expose live context after resume",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    status = mizu_session_close(session);
    if (!expect_status("cuda session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("cuda model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("cuda runtime destroy", status, MIZU_STATUS_OK)) return 1;
    runtime = NULL;
    model = NULL;
    session = NULL;

    if (!expect_true("cuda artifact cache should contain weight format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_weight_pack_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain projector format",
                     file_contains_substring(artifact_cache_path, "cuda_u8_bf16_projector_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain prefill format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_prefill_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain decode format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_decode_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain session metadata",
                     file_contains_substring(artifact_cache_path, "meta session"))) return 1;

    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -type f | grep -q .");
    if (!expect_true("cuda weight artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"weights/token_embeddings.bin\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain imported tensor lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"tensor_bytes=631142400\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain exact imported tensor byte estimates", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_kind=cuda_import_weight_pack_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should materialize an import-driven pack record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the expected packed tensor count", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_total_bytes=621967360\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the expected packed tensor bytes", command_status == 0)) return 1;
    command_status = system("grep -R \"pack1=token_embeddings|embedding_table|weights/token_embeddings.bin|offset=0|bytes=306561024\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the first packed tensor entry", command_status == 0)) return 1;
    command_status = system("grep -R \"pack1=token_embeddings|embedding_table|weights/token_embeddings.bin|offset=0|bytes=306561024|layout=row_major|storage=q4_k\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the first packed tensor storage type", command_status == 0)) return 1;
    command_status = system("grep -R \"source_offset=128\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain per-tensor source offsets", command_status == 0)) return 1;
    command_status = system("grep -R \"pack4=lm_head|token_projection|weights/lm_head.bin|offset=315406336|bytes=306561024\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the final packed tensor entry", command_status == 0)) return 1;
    command_status = system("grep -R \"pack4=lm_head|token_projection|weights/lm_head.bin|offset=315406336|bytes=306561024|layout=row_major|storage=q4_k\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the final packed tensor storage type", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packtiles' | grep -q .");
    if (!expect_true("generated cuda weight artifacts should no longer materialize pack tile-cache indexes", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_payload=artifacts/cuda/cuda/weights/.*\\.packpayload\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("generated cuda weight artifacts should no longer reference pack payload sidecars", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_buffer=artifacts/cuda/cuda/weights/.*\\.packbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should reference the dedicated pack buffer directly", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packpayload' | grep -q .");
    if (!expect_true("generated cuda weight artifacts should no longer materialize pack payload sidecars", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packbuffer' | grep -q .");
    if (!expect_true("cuda weight pack buffer should exist", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packbuffer' -size +0c | grep -q .");
    if (!expect_true("cuda weight pack buffer should store staged binary page and tile bytes", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packbuffer' -exec sh -c 'od -An -t x1 -N 4 \"$1\" | tr -d \" \\n\" | grep -q \"4d5a5042\"' _ {} \\;");
    if (!expect_true("cuda weight pack buffer should begin with the typed buffer magic", command_status == 0)) return 1;
    if (!expect_true("cuda weight pack buffer path should be discoverable",
                     capture_first_line("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packbuffer' | head -n 1",
                                        pack_buffer_path, sizeof(pack_buffer_path)))) return 1;
    if (!expect_true("cuda weight pack buffer should be readable",
                     read_binary_file_alloc(pack_buffer_path, &pack_buffer_bytes, &pack_buffer_size))) return 1;
    if (!expect_true("cuda weight pack buffer should expose version",
                     read_le_u32_field(pack_buffer_bytes, pack_buffer_size, 4U, &pack_buffer_version))) return 1;
    if (!expect_u64("cuda weight pack buffer should use source-offset v2 records",
                    (uint64_t)pack_buffer_version, 2U)) return 1;
    if (!expect_true("cuda weight pack buffer should expose entry bytes",
                     read_le_u32_field(pack_buffer_bytes, pack_buffer_size, 12U, &pack_buffer_entry_bytes))) return 1;
    if (!expect_u64("cuda weight pack buffer entries should include source offsets",
                    (uint64_t)pack_buffer_entry_bytes, 104U)) return 1;
    if (!expect_true("cuda weight pack buffer should expose pack count",
                     read_le_u32_field(pack_buffer_bytes, pack_buffer_size, 16U, &pack_buffer_count))) return 1;
    if (!expect_u64("cuda weight pack buffer should retain four fixture packs",
                    (uint64_t)pack_buffer_count, 4U)) return 1;
    if (!expect_true("cuda weight pack buffer should expose first span hash",
                     read_le_i64_field(pack_buffer_bytes, pack_buffer_size, 32U + 56U, &pack1_span_hash))) return 1;
    if (!expect_true("cuda weight pack buffer should expose second span hash",
                     read_le_i64_field(pack_buffer_bytes, pack_buffer_size,
                                       32U + (size_t)pack_buffer_entry_bytes + 56U, &pack2_span_hash))) return 1;
    if (!expect_true("cuda source offsets should produce distinct span hashes",
                     pack1_span_hash > 0 && pack2_span_hash > 0 && pack1_span_hash != pack2_span_hash)) return 1;
    if (!expect_true("cuda weight pack buffer should expose first source offset",
                     read_le_i64_field(pack_buffer_bytes, pack_buffer_size, 32U + 96U, &pack1_source_offset))) return 1;
    if (!expect_true("cuda weight pack buffer should expose second source offset",
                     read_le_i64_field(pack_buffer_bytes, pack_buffer_size,
                                       32U + (size_t)pack_buffer_entry_bytes + 96U, &pack2_source_offset))) return 1;
    if (!expect_i64("cuda weight pack buffer should retain first fixture source offset",
                    pack1_source_offset, 0)) return 1;
    if (!expect_i64("cuda weight pack buffer should retain second fixture source offset",
                    pack2_source_offset, 128)) return 1;
    free(pack_buffer_bytes);
    pack_buffer_bytes = NULL;
    pack_buffer_size = 0U;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector -type f | grep -q .");
    if (!expect_true("cuda projector artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"stage=2;.*shape0=8\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain staged modal byte count", command_status == 0)) return 1;
    command_status = system("grep -R \"projector/vision_projector.bin\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain imported projector lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"projector_bytes=9175040\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain exact imported projector byte estimates", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dependency=cuda_import_weight_pack_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should depend on the import-driven weight pack", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_ref_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain the packed tensor count dependency", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_cache=artifacts/cuda/cuda/weights/\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should no longer require a weight-pack tile-cache index", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_payload=artifacts/cuda/cuda/weights/.*\\.packpayload\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("generated cuda projector artifacts should no longer require a weight-pack payload sidecar", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_buffer=artifacts/cuda/cuda/weights/.*\\.packbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should reference the weight-pack binary buffer directly", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill -type f | grep -q .");
    if (!expect_true("cuda prefill artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"stage=3;.*format=cuda_bf16_prefill_plan_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain stable stage metadata", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dependency=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a textual pack dependency marker", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_bytes=621967360\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require textual packed-byte dependency hints", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use_kind=cuda_prefill_pack_usage_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual tensor-usage markers", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use_count=3\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual usage-count hints", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use_bytes=315406336\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual usage-byte hints", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_dispatch_count=3\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual dispatch-count hints", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_dispatch1=pack=1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual pack-index dispatch entries", command_status != 0)) return 1;
    snprintf(command_buffer, sizeof(command_buffer),
             "grep -R \"pack_span_root=%s/mizu_import\" %s/artifacts/cuda/cuda/plans/prefill >/dev/null",
             fixture_runtime_root, persist_root);
    command_status = system(command_buffer);
    if (!expect_true("cuda prefill artifact should no longer require a textual span-root hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span1=weights/token_embeddings.bin|sample_bytes=64\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual tensor-span records", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span_cache=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a span-cache sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span_buffer=artifacts/cuda/cuda/plans/prefill/.*\\.spanbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a span-buffer sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_usage_buffer=artifacts/cuda/cuda/plans/prefill/.*\\.usagebuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a usage-buffer sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_cache=artifacts/cuda/cuda/weights/\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a direct weight-pack tile-cache reference", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_buffer=artifacts/cuda/cuda/weights/.*\\.packbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a direct weight-pack binary buffer reference", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_dispatch_buffer=artifacts/cuda/cuda/plans/prefill/.*\\.dispatchbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer require a compact dispatch-buffer hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use1=token_embeddings|embedding_table|offset=0|bytes=306561024\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should no longer retain textual tensor-usage entries", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode -type f | grep -q .");
    if (!expect_true("cuda decode artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"stage=4;.*format=cuda_bf16_decode_plan_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain stable stage metadata", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dependency=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a textual pack dependency marker", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use_kind=cuda_decode_pack_usage_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer retain textual tensor-usage markers", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer retain textual usage-count hints", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_dispatch4=pack=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer retain textual pack-index dispatch entries", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span4=weights/lm_head.bin|sample_bytes=64\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer retain textual tensor-span records", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span_cache=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a span-cache sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_span_buffer=artifacts/cuda/cuda/plans/decode/.*\\.spanbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a span-buffer sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_usage_buffer=artifacts/cuda/cuda/plans/decode/.*\\.usagebuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a usage-buffer sidecar hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_cache=artifacts/cuda/cuda/weights/\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a direct weight-pack tile-cache reference", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_ref_tile_buffer=artifacts/cuda/cuda/weights/.*\\.packbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a direct weight-pack binary buffer reference", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_dispatch_buffer=artifacts/cuda/cuda/plans/decode/.*\\.dispatchbuffer\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer require a compact dispatch-buffer hint", command_status != 0)) return 1;
    command_status = system("grep -R \"pack_use4=lm_head|token_projection|offset=315406336|bytes=306561024\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should no longer retain textual tensor-usage entries", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.dispatchbuffer' | grep -q .");
    if (!expect_true("generated cuda warm plans should no longer materialize dispatch-buffer sidecars", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.usagebuffer' | grep -q .");
    if (!expect_true("generated cuda warm plans should no longer materialize usage-buffer sidecars", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' | grep -q .");
    if (!expect_true("generated cuda warm plans should no longer materialize span-cache sidecars", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spanbuffer' | grep -q .");
    if (!expect_true("generated cuda warm plans should no longer materialize span-buffer sidecars", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.execbuffer' | grep -q .");
    if (!expect_true("cuda exec-buffer sidecar should exist", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.execbuffer' -exec sh -c 'od -An -t x1 -N 4 \"$1\" | tr -d \" \\n\" | grep -q \"4d5a4558\"' _ {} \\;");
    if (!expect_true("cuda exec-buffer sidecar should store the expected binary magic", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.tilecache' | grep -q .");
    if (!expect_true("generated cuda warm plans should no longer materialize tile-cache payloads", command_status != 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/sessions -type f | grep -q .");
    if (!expect_true("cuda session artifact file should exist", command_status == 0)) return 1;

    if (!expect_true("cuda decode artifact plan path should resolve",
                     capture_first_line("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode -type f ! -name '*.execbuffer'",
                                        decode_plan_path, sizeof(decode_plan_path)))) {
        return 1;
    }
    if (!expect_true("cuda decode artifact plan should be readable",
                     read_text_file(decode_plan_path, decode_plan_text, sizeof(decode_plan_text)))) {
        return 1;
    }
    if (!expect_true("cuda decode artifact should reduce cleanly to binary-sidecar replay form",
                     reduce_plan_to_binary_sidecars(decode_plan_text, decode_binary_only_text,
                                                    sizeof(decode_binary_only_text)))) {
        return 1;
    }
    if (!expect_true("cuda decode artifact rewrite to binary refs should succeed",
                     write_text_file(decode_plan_path, decode_binary_only_text))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop direct pack-buffer references",
                     !file_contains_substring(decode_plan_path, "pack_ref_tile_buffer="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual pack dependency markers",
                     !file_contains_substring(decode_plan_path, "pack_dependency="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual weight-pack tile-cache hints",
                     !file_contains_substring(decode_plan_path, "pack_ref_tile_cache="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual weight-pack artifact hints",
                     !file_contains_substring(decode_plan_path, "pack_ref_artifact="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual pack-ref static dependency hints",
                     !file_contains_substring(decode_plan_path, "pack_ref_hash=") &&
                     !file_contains_substring(decode_plan_path, "pack_ref_bytes=") &&
                     !file_contains_substring(decode_plan_path, "pack_ref_count="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual weight-pack lineage dependency hints",
                     !file_contains_substring(decode_plan_path, "weight_pack_hash=") &&
                     !file_contains_substring(decode_plan_path, "weight_pack_bytes=") &&
                     !file_contains_substring(decode_plan_path, "weight_pack_count="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop direct usage-buffer references",
                     !file_contains_substring(decode_plan_path, "pack_usage_buffer="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop direct dispatch-buffer references",
                     !file_contains_substring(decode_plan_path, "pack_dispatch_buffer="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop direct span-buffer references",
                     !file_contains_substring(decode_plan_path, "pack_span_buffer="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should not require direct exec-buffer references",
                     !file_contains_substring(decode_plan_path, "pack_exec_buffer="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual pack-use records",
                     !file_contains_substring(decode_plan_path, "pack_use_kind="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual per-entry usage records",
                     !file_contains_substring(decode_plan_path, "pack_use4="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual usage summary fields",
                     !file_contains_substring(decode_plan_path, "pack_use_count=") &&
                     !file_contains_substring(decode_plan_path, "pack_use_bytes=") &&
                     !file_contains_substring(decode_plan_path, "pack_use_hash=") &&
                     !file_contains_substring(decode_plan_path, "pack_use_first_offset=") &&
                     !file_contains_substring(decode_plan_path, "pack_use_last_offset=") &&
                     !file_contains_substring(decode_plan_path, "pack_use_last_bytes="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual dispatch markers",
                     !file_contains_substring(decode_plan_path, "pack_dispatch_kind="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual dispatch summary fields",
                     !file_contains_substring(decode_plan_path, "pack_dispatch_hash=") &&
                     !file_contains_substring(decode_plan_path, "pack_dispatch_count="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual per-entry dispatch records",
                     !file_contains_substring(decode_plan_path, "pack_dispatch4="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual per-entry span records",
                     !file_contains_substring(decode_plan_path, "pack_span4="))) {
        return 1;
    }
    if (!expect_true("cuda binary-only decode plan should drop textual span-root and span-cache hints",
                     !file_contains_substring(decode_plan_path, "pack_span_root=") &&
                     !file_contains_substring(decode_plan_path, "pack_span_cache="))) {
        return 1;
    }

    for (size_t byte_index = 0; byte_index < sizeof(mutated_prefix); ++byte_index) {
        mutated_prefix[byte_index] = (uint8_t)(0xF0u - (uint8_t)byte_index);
    }
    if (!expect_true("cuda imported tensor span mutation should succeed",
                     overwrite_file_prefix(mutated_weight_path, mutated_prefix, sizeof(mutated_prefix)))) {
        return 1;
    }

    status = mizu_runtime_create(&runtime_config, &runtime_warm);
    if (!expect_status("cuda warm runtime create", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_warm, &model_config, &model_warm);
    if (!expect_status("cuda warm model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_warm, &model_report_warm);
    if (!expect_status("cuda warm model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm model load should hit weight cache",
                     (model_report_warm.cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm model load should reuse winner",
                     (model_report_warm.cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }

    status = mizu_session_open(model_warm, &session_config, &session_warm);
    if (!expect_status("cuda warm session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_warm, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda warm attach tokens", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_warm, &modal_input);
    if (!expect_status("cuda warm attach modal", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_warm.struct_size = sizeof(prefill_buffer_warm);
    prefill_buffer_warm.reports = prefill_reports_warm;
    prefill_buffer_warm.report_capacity = 2;
    prefill_buffer_warm.report_count = 0;

    status = mizu_session_prefill(session_warm, &prefill_buffer_warm);
    if (!expect_status("cuda warm prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm projector should hit multimodal cache",
                     (prefill_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_MM_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm projector should reuse winner",
                     (prefill_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm prefill should hit plan cache",
                     (prefill_reports_warm[1].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm prefill should reuse winner",
                     (prefill_reports_warm[1].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    status = mizu_session_get_info(session_warm, &session_info_warm);
    if (!expect_status("cuda warm session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda warm session should advance kv count after prefill", session_info_warm.kv_token_count, 3)) {
        return 1;
    }

    decode_result_warm.struct_size = sizeof(decode_result_warm);
    decode_result_warm.token_buffer = decode_tokens_warm;
    decode_result_warm.token_capacity = 1;
    decode_result_warm.token_count = 0;
    decode_result_warm.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result_warm.result_flags = 0;

    output_buffer_warm.struct_size = sizeof(output_buffer_warm);
    output_buffer_warm.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer_warm.data = output_tokens_warm;
    output_buffer_warm.byte_capacity = sizeof(output_tokens_warm);
    output_buffer_warm.bytes_written = 0;
    output_buffer_warm.output_flags = 0;

    decode_buffer_warm.struct_size = sizeof(decode_buffer_warm);
    decode_buffer_warm.reports = decode_reports_warm;
    decode_buffer_warm.report_capacity = 1;
    decode_buffer_warm.report_count = 0;

    status = mizu_session_decode_step(session_warm, &decode_options, &decode_result_warm, &decode_buffer_warm);
    if (!expect_status("cuda warm decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm decode should hit plan cache",
                     (decode_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm decode should reuse winner",
                     (decode_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm decode should reproduce the same token for the same multimodal context",
                     decode_tokens_warm[0] == decode_tokens[0])) {
        return 1;
    }
    status = mizu_session_read_output(session_warm, &output_buffer_warm);
    if (!expect_status("cuda warm read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm output should match warm decode token", output_tokens_warm[0] == decode_tokens_warm[0])) {
        return 1;
    }

    status = mizu_session_close(session_warm);
    if (!expect_status("cuda warm session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_warm);
    if (!expect_status("cuda warm model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_warm);
    if (!expect_status("cuda warm runtime destroy", status, MIZU_STATUS_OK)) return 1;
    runtime_warm = NULL;
    model_warm = NULL;
    session_warm = NULL;

    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' -delete");
    if (!expect_true("cuda span-cache sidecar removal should succeed", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.usagebuffer' -delete");
    if (!expect_true("cuda usage-buffer sidecar removal should succeed", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.dispatchbuffer' -delete");
    if (!expect_true("cuda dispatch-buffer sidecar removal should succeed", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spanbuffer' -delete");
    if (!expect_true("cuda span-buffer sidecar removal should succeed", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.tilecache' -delete");
    if (!expect_true("cuda tile-cache payload removal should succeed", command_status == 0)) return 1;

    status = mizu_runtime_create(&runtime_config, &runtime_fallback);
    if (!expect_status("cuda fallback runtime create", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_fallback, &model_config, &model_fallback);
    if (!expect_status("cuda fallback model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_fallback, &model_report_fallback);
    if (!expect_status("cuda fallback model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda fallback model load should still hit weight cache",
                     (model_report_fallback.cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) {
        return 1;
    }
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -name '*.packtiles' -delete");
    if (!expect_true("cuda weight-pack tile cache removal should succeed", command_status == 0)) return 1;

    status = mizu_session_open(model_fallback, &session_config, &session_fallback);
    if (!expect_status("cuda fallback session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_fallback, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda fallback attach tokens", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_fallback, &modal_input);
    if (!expect_status("cuda fallback attach modal", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_fallback.struct_size = sizeof(prefill_buffer_fallback);
    prefill_buffer_fallback.reports = prefill_reports_fallback;
    prefill_buffer_fallback.report_capacity = 2;
    prefill_buffer_fallback.report_count = 0;

    status = mizu_session_prefill(session_fallback, &prefill_buffer_fallback);
    if (!expect_status("cuda fallback prefill", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session_fallback, &session_info_fallback);
    if (!expect_status("cuda fallback session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda fallback session should advance kv count after prefill", session_info_fallback.kv_token_count, 3)) {
        return 1;
    }

    decode_result_fallback.struct_size = sizeof(decode_result_fallback);
    decode_result_fallback.token_buffer = decode_tokens_fallback;
    decode_result_fallback.token_capacity = 1;
    decode_result_fallback.token_count = 0;
    decode_result_fallback.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result_fallback.result_flags = 0;

    output_buffer_fallback.struct_size = sizeof(output_buffer_fallback);
    output_buffer_fallback.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer_fallback.data = output_tokens_fallback;
    output_buffer_fallback.byte_capacity = sizeof(output_tokens_fallback);
    output_buffer_fallback.bytes_written = 0;
    output_buffer_fallback.output_flags = 0;

    decode_buffer_fallback.struct_size = sizeof(decode_buffer_fallback);
    decode_buffer_fallback.reports = decode_reports_fallback;
    decode_buffer_fallback.report_capacity = 1;
    decode_buffer_fallback.report_count = 0;

    status = mizu_session_decode_step(session_fallback, &decode_options, &decode_result_fallback,
                                      &decode_buffer_fallback);
    if (!expect_status("cuda fallback decode", status, MIZU_STATUS_OK)) return 1;
#if MIZU_CUDA_BRIDGE_STUB
    if (!expect_true("cuda fallback decode should still emit a positive token from direct pack buffers",
                     decode_tokens_fallback[0] > 0)) {
#else
    if (!expect_true("cuda fallback decode should reproduce the same token from weight-pack caches",
                     decode_tokens_fallback[0] == decode_tokens[0])) {
#endif
        return 1;
    }
    status = mizu_session_read_output(session_fallback, &output_buffer_fallback);
    if (!expect_status("cuda fallback read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda fallback output should match fallback decode token",
                     output_tokens_fallback[0] == decode_tokens_fallback[0])) {
        return 1;
    }

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session_fallback, &park_buffer);
    if (!expect_status("cuda fallback park", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_close(session_fallback);
    if (!expect_status("cuda fallback session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_fallback);
    if (!expect_status("cuda fallback model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_fallback);
    if (!expect_status("cuda fallback runtime destroy", status, MIZU_STATUS_OK)) return 1;

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root cleanup should succeed", command_status == 0)) return 1;

    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
    puts("test_cuda_artifacts: PASS");
    return 0;
}
