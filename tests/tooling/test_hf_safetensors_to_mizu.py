#!/usr/bin/env python3
"""Smoke-test the dependency-free HuggingFace safetensors importer."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORTER = REPO_ROOT / "tools" / "import" / "hf_safetensors_to_mizu.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_hf_importer_") as temp_root:
        source_root = Path(temp_root) / "Qwen-3.5-VL-9B"
        source_root.mkdir(parents=True)
        write_json(
            source_root / "config.json",
            {
                "_name_or_path": "Qwen/Qwen-3.5-VL-9B",
                "_commit_hash": "fixture-qwen",
                "model_type": "qwen3_5_vl",
                "torch_dtype": "bfloat16",
                "vision_config": {"hidden_size": 1280},
            },
        )
        write_json(source_root / "tokenizer_config.json", {"tokenizer_class": "QwenTokenizer"})
        write_safetensors(
            source_root / "model-00001-of-00002.safetensors",
            {
                "model.embed_tokens.weight": ("BF16", [64, 32]),
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [32, 32]),
            },
        )
        write_safetensors(
            source_root / "model-00002-of-00002.safetensors",
            {
                "model.norm.weight": ("F32", [32]),
                "lm_head.weight": ("BF16", [32, 64]),
                "visual.position_embedding.weight": ("F16", [32, 16]),
                "vision_tower.vision_model.embeddings.class_embedding": ("F16", [16]),
                "visual.merger.mlp.0.weight": ("F16", [16, 32]),
            },
        )
        write_json(
            source_root / "model.safetensors.index.json",
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                    "visual.position_embedding.weight": "model-00002-of-00002.safetensors",
                    "vision_tower.vision_model.embeddings.class_embedding": "model-00002-of-00002.safetensors",
                    "visual.merger.mlp.0.weight": "model-00002-of-00002.safetensors",
                }
            },
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(source_root),
                "--link-mode",
                "copy",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode

        expect_file_contains(source_root / "manifest.mizu", "family = qwen3_5")
        expect_file_contains(source_root / "mizu_import" / "layout.mizu", "projector_present = true")
        expect_file_contains(source_root / "mizu_import" / "modalities.tsv", "1|image|image|encoded_bytes|u8")
        expect_file_contains(
            source_root / "mizu_import" / "projector.mizu",
            "artifact_path = projector/projector_assets.mizu",
        )
        tensor_inventory = (source_root / "mizu_import" / "tensors.tsv").read_text(encoding="utf-8")
        projector_inventory = (source_root / "mizu_import" / "projector" / "projector_assets.mizu").read_text(
            encoding="utf-8"
        )
        expect_contains(tensor_inventory, "model.embed_tokens.weight|embedding_table|bf16|row_major")
        expect_contains(
            tensor_inventory,
            "model.layers.0.self_attn.q_proj.weight|decoder_stack|bf16|packed|weights/model-00001-of-00002.safetensors|32x32|bf16",
        )
        expect_contains(tensor_inventory, "lm_head.weight|token_projection|bf16|row_major")
        expect_contains(tensor_inventory, "visual.position_embedding.weight|vision_encoder|f16|packed")
        expect_contains(
            tensor_inventory,
            "vision_tower.vision_model.embeddings.class_embedding|vision_encoder|f16|vector",
        )
        expect_contains(tensor_inventory, "visual.merger.mlp.0.weight|multimodal_projector|f16|packed")
        expect_contains(projector_inventory, "visual.position_embedding.weight|weights/model-00002-of-00002.safetensors")
        expect_contains(
            projector_inventory,
            "vision_tower.vision_model.embeddings.class_embedding|weights/model-00002-of-00002.safetensors",
        )
        expect_contains(projector_inventory, "visual.merger.mlp.0.weight|weights/model-00002-of-00002.safetensors")
        expect_path_exists(source_root / "mizu_import" / "weights" / "model-00001-of-00002.safetensors")
        expect_path_exists(source_root / "mizu_import" / "weights" / "model-00002-of-00002.safetensors")

        gemma_root = Path(temp_root) / "Gemma4-21B"
        gemma_root.mkdir(parents=True)
        write_json(
            gemma_root / "config.json",
            {
                "_name_or_path": "Google/Gemma4-21B",
                "model_type": "gemma4",
                "vision_config": {"hidden_size": 1152},
            },
        )
        write_safetensors(
            gemma_root / "model.safetensors",
            {
                "embed_tokens.weight": ("BF16", [128, 64]),
                "decoder.layers.0.mlp.up_proj.weight": ("BF16", [64, 256]),
                "mm_projector.weight": ("F16", [16, 64]),
            },
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(gemma_root),
                "--link-mode",
                "copy",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode

        expect_file_contains(gemma_root / "manifest.mizu", "family = gemma4")
        expect_file_contains(gemma_root / "mizu_import" / "tensors.tsv", "mm_projector.weight|multimodal_projector")

        broken_root = Path(temp_root) / "Broken-Qwen"
        broken_root.mkdir(parents=True)
        write_json(
            broken_root / "config.json",
            {
                "_name_or_path": "Broken/Qwen",
                "model_type": "qwen3_5_vl",
            },
        )
        write_invalid_safetensors(broken_root / "model.safetensors")
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(broken_root),
                "--link-mode",
                "copy",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        expect_failed_run(
            "safetensors importer should reject tensors whose data range is shorter than dtype/shape",
            completed,
            "expected 512 bytes from dtype/shape",
        )

    print("test_hf_safetensors_to_mizu: PASS")
    return 0


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header: dict[str, object] = {}
    data_offset = 0
    for name, (dtype, shape) in tensors.items():
        byte_count = max(1, product(shape) * dtype_size(dtype))
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [data_offset, data_offset + byte_count],
        }
        data_offset += byte_count

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(b"\0" * data_offset)


def write_invalid_safetensors(path: Path) -> None:
    header = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [16, 16],
            "data_offsets": [0, 64],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(b"\0" * 64)


def dtype_size(dtype: str) -> int:
    return {"U8": 1, "I32": 4, "F16": 2, "BF16": 2, "F32": 4}[dtype]


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def expect_file_contains(path: Path, needle: str) -> None:
    expect_path_exists(path)
    expect_contains(path.read_text(encoding="utf-8"), needle)


def expect_contains(haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"missing expected text: {needle}")


def expect_path_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"missing expected path: {path}")


def expect_failed_run(label: str, completed: subprocess.CompletedProcess[str], stderr_needle: str) -> None:
    if completed.returncode == 0:
        raise AssertionError(f"{label}: command unexpectedly succeeded")
    if stderr_needle not in completed.stderr:
        raise AssertionError(f"{label}: missing stderr text {stderr_needle!r}\n{completed.stderr}")


if __name__ == "__main__":
    raise SystemExit(main())
