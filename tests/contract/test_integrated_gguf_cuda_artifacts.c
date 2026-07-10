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

static int run_command(const char *label, const char *command) {
    int status = system(command);
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
    if (!expect_status("integrated gguf model open", status, MIZU_STATUS_OK)) return 0;

    model_report.struct_size = sizeof(model_report);
    status = mizu_model_get_last_report(*out_model, &model_report);
    if (!expect_status("integrated gguf model report", status, MIZU_STATUS_OK)) return 0;
    return expect_true("integrated gguf model load should route to CUDA",
                       model_report.execution_route == MIZU_EXEC_ROUTE_CUDA);
}

static int run_session_smoke(mizu_model_t *model) {
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
    if (!expect_status("integrated gguf session open", status, MIZU_STATUS_OK)) return 0;

    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("integrated gguf attach tokens", status, MIZU_STATUS_OK)) ok = 0;

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
    if (!expect_status("integrated gguf attach modal", status, MIZU_STATUS_OK)) ok = 0;

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 2;
    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("integrated gguf prefill", status, MIZU_STATUS_OK)) ok = 0;
    if (ok) {
        ok = expect_true("integrated gguf projector should route to CUDA",
                         prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA) && ok;
        ok = expect_true("integrated gguf prefill should route to CUDA",
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
    if (!expect_status("integrated gguf decode", status, MIZU_STATUS_OK)) ok = 0;
    if (ok) {
        ok = expect_true("integrated gguf decode should route to CUDA",
                         decode_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA) && ok;
        ok = expect_true("integrated gguf decode should emit one token", decode_result.token_count == 1U) && ok;
    }

    status = mizu_session_close(session);
    if (!expect_status("integrated gguf session close", status, MIZU_STATUS_OK)) ok = 0;
    return ok;
}

int main(void) {
    const char *persist_root = "/tmp/mizu_integrated_gguf_cuda_artifacts";
    const char *bundle_root = "/tmp/mizu_integrated_gguf_cuda_artifacts/bundle";
    char command[8192];
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_runtime_config_t runtime_config;
    mizu_status_code_t status;
    int command_status;
    int ok = 1;

    if (!run_command("integrated gguf smoke root setup",
                     "rm -rf /tmp/mizu_integrated_gguf_cuda_artifacts && "
                     "mkdir -p /tmp/mizu_integrated_gguf_cuda_artifacts/cache")) {
        return 1;
    }

    if (!run_command(
            "integrated gguf fixture write",
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "import struct\n"
            "VALUE_TYPES = {'uint32': 4, 'bool': 7, 'string': 8}\n"
            "GGML_TYPES = {'F32': 0, 'F16': 1, 'Q4_K': 12, 'Q5_K': 13}\n"
            "GGML_QUANT_SIZES = {'F32': (1, 4), 'F16': (1, 2), 'Q4_K': (256, 144), 'Q5_K': (256, 176)}\n"
            "path = Path('/tmp/mizu_integrated_gguf_cuda_artifacts/qwen35_integrated.gguf')\n"
            "metadata = {\n"
            "    'general.architecture': ('string', 'qwen35'),\n"
            "    'general.name': ('string', 'Qwen3.5 9B Integrated'),\n"
            "    'general.type': ('string', 'model'),\n"
            "    'general.file_type': ('uint32', 15),\n"
            "    'general.quantization_version': ('uint32', 2),\n"
            "    'clip.has_vision_encoder': ('bool', True),\n"
            "}\n"
            "tensors = [\n"
            "    ('token_embd.weight', [4096, 248320], 'Q4_K', 0),\n"
            "    ('blk.0.attn_qkv.weight', [4096, 8192], 'Q5_K', 128),\n"
            "    ('output_norm.weight', [4096], 'F32', 256),\n"
            "    ('output.weight', [4096, 248320], 'Q4_K', 384),\n"
            "    ('v.blk.0.attn_qkv.weight', [1152, 3456], 'F16', 512),\n"
            "    ('mm.0.weight', [1152, 4096], 'F16', 640),\n"
            "    ('mm.2.bias', [4096], 'F32', 768),\n"
            "]\n"
            "def write_string(handle, text):\n"
            "    encoded = text.encode('utf-8')\n"
            "    handle.write(struct.pack('<Q', len(encoded)))\n"
            "    handle.write(encoded)\n"
            "def tensor_nbytes(shape, ggml_type):\n"
            "    block_elements, block_bytes = GGML_QUANT_SIZES[ggml_type]\n"
            "    row_elements = shape[0]\n"
            "    row_bytes = (row_elements // block_elements) * block_bytes\n"
            "    total_rows = 1\n"
            "    for dim in shape[1:]:\n"
            "        total_rows *= dim\n"
            "    return total_rows * row_bytes\n"
            "with path.open('wb') as handle:\n"
            "    handle.write(b'GGUF')\n"
            "    handle.write(struct.pack('<I', 3))\n"
            "    handle.write(struct.pack('<Q', len(tensors)))\n"
            "    handle.write(struct.pack('<Q', len(metadata)))\n"
            "    for key, (value_type, value) in metadata.items():\n"
            "        write_string(handle, key)\n"
            "        handle.write(struct.pack('<I', VALUE_TYPES[value_type]))\n"
            "        if value_type == 'string':\n"
            "            write_string(handle, str(value))\n"
            "        elif value_type == 'bool':\n"
            "            handle.write(struct.pack('<?', bool(value)))\n"
            "        elif value_type == 'uint32':\n"
            "            handle.write(struct.pack('<I', int(value)))\n"
            "        else:\n"
            "            raise RuntimeError(value_type)\n"
            "    for name, shape, ggml_type, offset in tensors:\n"
            "        write_string(handle, name)\n"
            "        handle.write(struct.pack('<I', len(shape)))\n"
            "        for dim in shape:\n"
            "            handle.write(struct.pack('<Q', dim))\n"
            "        handle.write(struct.pack('<I', GGML_TYPES[ggml_type]))\n"
            "        handle.write(struct.pack('<Q', offset))\n"
            "    alignment = 32\n"
            "    padding = (-handle.tell()) % alignment\n"
            "    if padding:\n"
            "        handle.write(b'\\0' * padding)\n"
            "    payload_bytes = 0\n"
            "    for _, shape, ggml_type, offset in tensors:\n"
            "        payload_bytes = max(payload_bytes, offset + tensor_nbytes(shape, ggml_type))\n"
            "    handle.write(b'\\0' * payload_bytes)\n"
            "PY")) {
        return 1;
    }

    snprintf(command, sizeof(command),
             "python3 tools/import/gguf_to_mizu.py '%s/qwen35_integrated.gguf' "
             "--output-root '%s' --link-mode copy --force >/tmp/mizu_integrated_gguf_cuda_artifacts/import.log",
             persist_root, bundle_root);
    if (!run_command("integrated gguf import", command)) return 1;

    command_status = system("grep -R \"projector_present = true\" "
                            "/tmp/mizu_integrated_gguf_cuda_artifacts/bundle/mizu_import/layout.mizu >/dev/null");
    if (!expect_true("integrated gguf import should report projector presence", command_status == 0)) return 1;

    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE\n");
        return 1;
    }

    memset(&runtime_config, 0, sizeof(runtime_config));
    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = "/tmp/mizu_integrated_gguf_cuda_artifacts/cache";
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("integrated gguf runtime create", status, MIZU_STATUS_OK)) return 1;

    if (!open_model_smoke(runtime, bundle_root, &model)) ok = 0;
    if (ok && !run_session_smoke(model)) ok = 0;

    if (model != NULL) {
        status = mizu_model_close(model);
        if (!expect_status("integrated gguf model close", status, MIZU_STATUS_OK)) ok = 0;
    }

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("integrated gguf runtime destroy", status, MIZU_STATUS_OK)) ok = 0;
    if (!ok) return 1;

    command_status = system("grep -R \"pack_count=4\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("integrated gguf CUDA decoder weight pack should retain four non-projector tensors",
                     command_status == 0)) return 1;
    command_status = system("grep -R \"v.blk.0.attn_qkv.weight\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("integrated gguf CUDA decoder weight pack should exclude vision tensors from the shared GGUF file",
                     command_status != 0)) return 1;
    command_status = system("grep -R \"mm.0.weight\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("integrated gguf CUDA decoder weight pack should exclude projector tensors from the shared GGUF file",
                     command_status != 0)) return 1;
    command_status = system("grep -R -E \"projector_bytes=[1-9]\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("integrated gguf CUDA projector artifact should carry projector byte lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"v.blk.0.attn_qkv.weight\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("integrated gguf CUDA projector artifact should retain shared-file vision lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"mm.0.weight\" /tmp/mizu_integrated_gguf_cuda_artifacts/cache/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("integrated gguf CUDA projector artifact should retain shared-file projector lineage", command_status == 0)) return 1;

    printf("test_integrated_gguf_cuda_artifacts: PASS\n");
    return 0;
}
