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

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_modal_input_desc_t modal_input;
    uint8_t image_bytes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int64_t shape[1] = {8};

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

    memset(&modal_input, 0, sizeof(modal_input));
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
    if (!expect_status("valid modal input", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_clear_pending_inputs(session);
    if (!expect_status("clear pending after valid input", status, MIZU_STATUS_OK)) return 1;

    modal_input.slot_name_z = "audio";
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("unknown slot should be rejected", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    modal_input.slot_name_z = "image";

    modal_input.placeholder_ordinal = 2;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("placeholder ordinal should be rejected", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    modal_input.placeholder_ordinal = 1;

    modal_input.storage_kind = MIZU_STORAGE_KIND_HOST_TENSOR;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("unsupported storage kind should be rejected", status, MIZU_STATUS_UNSUPPORTED_MODALITY)) return 1;
    modal_input.storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES;

    modal_input.modality_kind = MIZU_MODALITY_KIND_TENSOR;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("unsupported modality should be rejected", status, MIZU_STATUS_UNSUPPORTED_MODALITY)) return 1;
    modal_input.modality_kind = MIZU_MODALITY_KIND_IMAGE;

    modal_input.rank = 1;
    modal_input.shape = shape;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("ranked encoded bytes should be rejected", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    modal_input.rank = 0;
    modal_input.shape = NULL;

    modal_input.lifetime_policy = 99;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("unknown lifetime policy should be rejected", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    modal_input.lifetime_policy = MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL;
    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("borrow-until-prefill should be accepted", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_clear_pending_inputs(session);
    if (!expect_status("clear pending after borrow input", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_close(session);
    if (!expect_status("session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_modal_input_validation: PASS");
    return 0;
}
