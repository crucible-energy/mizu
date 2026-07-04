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

    *required_bytes = 0;
    memset(buffer, 0, buffer_size);
    status = mizu_runtime_copy_last_error(runtime, buffer, buffer_size, required_bytes);
    return expect_status("copy runtime error", status, MIZU_STATUS_OK);
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = (mizu_model_t *)(uintptr_t)0x1;
    mizu_model_t *good_model = NULL;
    mizu_status_code_t status;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    size_t required_bytes;
    char error_buffer[256];

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
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    model_config.model_root_z = "tests/fixtures/models/fixture_bad_manifest";
    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open should reject malformed manifest", status, MIZU_STATUS_INVALID_ARGUMENT)) return 1;
    if (!expect_true("bad manifest should not leave a partial model handle", model == NULL)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("bad manifest should publish runtime error text", required_bytes > 1)) return 1;
    if (!expect_true("bad manifest error should mention model manifest load",
                     strstr(error_buffer, "model manifest load failed") != NULL)) return 1;

    model = (mizu_model_t *)(uintptr_t)0x2;
    model_config.model_root_z = "tests/fixtures/models/fixture_bad_import_bundle";
    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open should reject broken import bundle", status, MIZU_STATUS_IO_ERROR)) return 1;
    if (!expect_true("bad import bundle should not leave a partial model handle", model == NULL)) return 1;
    if (!copy_runtime_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes)) return 1;
    if (!expect_true("bad import bundle should publish runtime error text", required_bytes > 1)) return 1;
    if (!expect_true("bad import bundle error should mention model manifest load",
                     strstr(error_buffer, "model manifest load failed") != NULL)) return 1;

    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    status = mizu_model_open(runtime, &model_config, &good_model);
    if (!expect_status("good model should still open after earlier failures", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_close(good_model);
    if (!expect_status("good model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("runtime destroy", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_model_open_failures: PASS");
    return 0;
}
