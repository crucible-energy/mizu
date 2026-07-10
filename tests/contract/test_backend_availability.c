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
    mizu_runtime_t *runtime_apple = NULL;
    mizu_model_t *model = NULL;
    mizu_status_code_t status;
    size_t required_bytes = 0;
    size_t required_bytes_only = 0;
    size_t truncated_required_bytes = 0;
    char error_buffer[256];
    char truncated_error_buffer[8];
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;

    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "0", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE=0\n");
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

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = NULL;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_DISABLED;
    runtime_config.exploration_budget = 0;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("runtime create without backends", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("model open should fail when Apple is unavailable", status, MIZU_STATUS_NO_VALID_PLAN)) return 1;

    memset(error_buffer, 0, sizeof(error_buffer));
    status = mizu_runtime_copy_last_error(runtime, error_buffer, sizeof(error_buffer), &required_bytes);
    if (!expect_status("copy last error", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("copy last error should report full required size",
                     required_bytes == strlen(error_buffer) + 1)) return 1;
    if (!expect_true("error text should mention unavailable backend",
                     strstr(error_buffer, "no requested backend is available on this runtime") != NULL)) return 1;
    status = mizu_runtime_copy_last_error(runtime, NULL, 0, &required_bytes_only);
    if (!expect_status("copy last error size only", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("size-only error copy should match required size",
                     required_bytes_only == required_bytes)) return 1;
    memset(truncated_error_buffer, 'X', sizeof(truncated_error_buffer));
    status = mizu_runtime_copy_last_error(runtime, truncated_error_buffer, sizeof(truncated_error_buffer),
                                          &truncated_required_bytes);
    if (!expect_status("copy last error short buffer", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("short-buffer error copy should match required size",
                     truncated_required_bytes == required_bytes)) return 1;
    if (!expect_true("short-buffer error copy should stay null terminated",
                     truncated_error_buffer[sizeof(truncated_error_buffer) - 1] == '\0')) return 1;
    if (!expect_true("short-buffer error copy should preserve prefix",
                     strncmp(error_buffer, truncated_error_buffer, sizeof(truncated_error_buffer) - 1) == 0)) return 1;

    status = mizu_runtime_destroy(runtime);
    if (!expect_status("destroy unavailable runtime", status, MIZU_STATUS_OK)) return 1;
    runtime = NULL;

    if (setenv("MIZU_FORCE_APPLE_ANE_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_APPLE_ANE_AVAILABLE=1\n");
        return 1;
    }

    status = mizu_runtime_create(&runtime_config, &runtime_apple);
    if (!expect_status("runtime create with forced Apple ANE", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_open(runtime_apple, &model_config, &model);
    if (!expect_status("model open should succeed when Apple is forced available", status, MIZU_STATUS_OK)) return 1;

    status = mizu_model_close(model);
    if (!expect_status("close forced-Apple model", status, MIZU_STATUS_OK)) return 1;

    status = mizu_runtime_destroy(runtime_apple);
    if (!expect_status("destroy forced-Apple runtime", status, MIZU_STATUS_OK)) return 1;

    unsetenv("MIZU_FORCE_APPLE_ANE_AVAILABLE");
    unsetenv("MIZU_FORCE_APPLE_METAL_AVAILABLE");
    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");

    puts("test_backend_availability: PASS");
    return 0;
}
