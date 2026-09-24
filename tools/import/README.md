# Import Tooling

Importer and conversion tooling for `mizu` lives here.

The intended job of this directory is:

- read source model assets from upstream layouts
- normalize them into the `manifest.mizu` plus `mizu_import/` bundle shape
- emit stable relative paths and inventory files
- fail loudly when imported assets are incomplete or ambiguous

The first concrete target layout is documented in:

- `docs/IMPORTER_LAYOUT.md`

## HuggingFace Safetensors Smoke Importer

`hf_safetensors_to_mizu.zig` converts a local HuggingFace-style safetensors
directory into the Mizu bundle shape. It uses Zig 0.16 and has no Python or
third-party runtime dependency.

Example:

```sh
zig run tools/import/hf_safetensors_to_mizu.zig -- /models/qwen-vl \
  --family qwen3_5 \
  --link-mode symlink
```

The tool writes:

- `<model_root>/manifest.mizu`
- `<model_root>/mizu_import/layout.mizu`
- `<model_root>/mizu_import/tensors.tsv`
- `<model_root>/mizu_import/modalities.tsv`
- `<model_root>/mizu_import/projector.mizu`

By default, safetensors shards are symlinked into `mizu_import/weights/` so the
Fortran loader can keep enforcing safe import-relative paths without copying
large model files. Use `--link-mode copy` for small fixtures or portable test
bundles, and `--force` when intentionally regenerating an existing bundle.

This is an asset-layout smoke importer, not a real inference converter. It
classifies common Qwen/Gemma tensor-name patterns into Mizu tensor roles and
creates stable model/projector identity, giving us a concrete way to start
testing real local assets before backend math is complete.

## GGUF Smoke Importer

`gguf_to_mizu.zig` converts local GGUF model assets into the same Mizu bundle
shape. It reads only GGUF metadata and tensor-info headers, so it can inspect
large quantized model files without loading the weight payload into memory.

Example with a paired Qwen model and mmproj file:

```sh
zig run tools/import/gguf_to_mizu.zig -- ~/.qwench/models/qwen3.5-9b-instruct-q4_k_m.gguf \
  --projector-gguf ~/.qwench/models/mmproj-Qwen_Qwen3.5-9B-f16.gguf \
  --output-root build/import-smoke/qwen35-9b \
  --link-mode symlink
```

Example with a single Gemma GGUF:

```sh
zig run tools/import/gguf_to_mizu.zig -- ~/.qwench/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf \
  --output-root build/import-smoke/gemma4-26b \
  --link-mode symlink
```

The tool writes the standard bundle files plus:

- `<output_root>/mizu_import/gguf_tensors.tsv`

`tensors.tsv` keeps the loader-compatible Mizu view. Quantized GGUF tensors use
Mizu staging dtypes such as `f16` in the `dtype` column and preserve the exact
GGUF type, such as `q4_k` or `iq2_xxs`, in the core `storage_type` column.
Runtime byte accounting uses recognized GGUF block sizes from `storage_type`.
`gguf_tensors.tsv` additionally preserves both the GGUF tensor-info
`data_offset` and the absolute file `source_offset` used by runtime span
sampling. That lets CUDA pack identity distinguish individual tensors that
share one containing GGUF file.

On this machine, the current `~/.qwench/models` smoke assets import as:

- Qwen3.5 9B plus `mmproj-Qwen_Qwen3.5-9B-f16.gguf`: 761 tensor records,
  projector present
- `gemma-4-26B-A4B-it-UD-IQ2_M.gguf`: 658 tensor records, no projector tensors
  present in that local GGUF file
