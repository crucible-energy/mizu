#!/usr/bin/env python3
"""Smoke-test the dependency-free GGUF importer."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORTER = REPO_ROOT / "tools" / "import" / "gguf_to_mizu.py"

VALUE_TYPES = {
    "uint32": 4,
    "bool": 7,
    "string": 8,
}

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_K": 12,
    "Q5_K": 13,
}

GGML_QUANT_SIZES = {
    "F32": (1, 4),
    "F16": (1, 2),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_gguf_importer_") as temp_root:
        temp_path = Path(temp_root)
        qwen_model = temp_path / "qwen35.gguf"
        qwen_projector = temp_path / "mmproj-qwen35.gguf"
        qwen_output = temp_path / "qwen35_mizu"

        write_gguf(
            qwen_model,
            {
                "general.architecture": ("string", "qwen35"),
                "general.name": ("string", "Qwen3.5 9B"),
                "general.type": ("string", "model"),
                "general.file_type": ("uint32", 15),
                "general.quantization_version": ("uint32", 2),
                "tokenizer.ggml.model": ("string", "gpt2"),
            },
            [
                ("token_embd.weight", [4096, 248320], "Q4_K", 0),
                ("blk.0.attn_qkv.weight", [4096, 8192], "Q5_K", 128),
                ("output_norm.weight", [4096], "F32", 256),
                ("output.weight", [4096, 248320], "Q4_K", 384),
            ],
        )
        write_gguf(
            qwen_projector,
            {
                "general.architecture": ("string", "clip"),
                "general.name": ("string", "Qwen3.5 9B mmproj"),
                "general.type": ("string", "mmproj"),
                "clip.has_vision_encoder": ("bool", True),
                "general.file_type": ("uint32", 1),
                "general.quantization_version": ("uint32", 2),
            },
            [
                ("v.blk.0.attn_qkv.weight", [1152, 3456], "F16", 0),
                ("mm.0.weight", [1152, 4096], "F16", 128),
                ("mm.2.bias", [4096], "F32", 256),
            ],
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(qwen_model),
                "--projector-gguf",
                str(qwen_projector),
                "--output-root",
                str(qwen_output),
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

        expect_file_contains(qwen_output / "manifest.mizu", "family = qwen3_5")
        expect_file_contains(qwen_output / "mizu_import" / "layout.mizu", "gguf_inventory = gguf_tensors.tsv")
        expect_file_contains(qwen_output / "mizu_import" / "layout.mizu", "projector_present = true")
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "token_embd.weight|embedding_table|f16|row_major|weights/qwen35.gguf|4096x248320|q4_k",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "blk.0.attn_qkv.weight|decoder_stack|f16|packed|weights/qwen35.gguf|4096x8192|q5_k",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "output.weight|token_projection|f16|row_major|weights/qwen35.gguf|4096x248320|q4_k",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "mm.0.weight|multimodal_projector|f16|packed|weights/mmproj-qwen35.gguf|1152x4096|f16",
        )
        gguf_inventory = qwen_output / "mizu_import" / "gguf_tensors.tsv"
        expect_file_contains(gguf_inventory, "data_offset|source_offset|shape")
        token_row = find_tsv_row(gguf_inventory, "token_embd.weight")
        if len(token_row) != 9:
            raise AssertionError(f"expected 9 GGUF inventory fields, got {len(token_row)}: {token_row}")
        expect_equal_list(
            "token_embd GGUF inventory identity",
            token_row[:7],
            ["token_embd.weight", "model", "q4_k", "f16", "row_major", "weights/qwen35.gguf", "0"],
        )
        if int(token_row[7]) <= int(token_row[6]):
            raise AssertionError(f"expected absolute source_offset to exceed data_offset: {token_row}")
        if token_row[8] != "4096x248320":
            raise AssertionError(f"unexpected token_embd shape in GGUF inventory: {token_row[8]}")
        expect_file_contains(
            qwen_output / "mizu_import" / "projector" / "projector_assets.mizu",
            "mm.0.weight|weights/mmproj-qwen35.gguf|offset=128|ggml_type=f16",
        )
        expect_path_exists(qwen_output / "mizu_import" / "weights" / "qwen35.gguf")
        expect_path_exists(qwen_output / "mizu_import" / "weights" / "mmproj-qwen35.gguf")

        gemma_model = temp_path / "gemma4.gguf"
        gemma_output = temp_path / "gemma4_mizu"
        write_gguf(
            gemma_model,
            {
                "general.architecture": ("string", "gemma4"),
                "general.name": ("string", "Gemma-4-26B-A4B-It"),
                "general.type": ("string", "model"),
                "general.file_type": ("uint32", 29),
                "general.quantization_version": ("uint32", 2),
                "tokenizer.ggml.model": ("string", "gemma4"),
            },
            [
                ("token_embd.weight", [2816, 262144], "Q5_K", 0),
                ("blk.0.ffn_gate.weight", [2816, 2112], "Q5_K", 128),
                ("output_norm.weight", [2816], "F32", 256),
            ],
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(gemma_model),
                "--output-root",
                str(gemma_output),
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

        expect_file_contains(gemma_output / "manifest.mizu", "family = gemma4")
        expect_file_contains(gemma_output / "manifest.mizu", "projector_present = false")
        expect_file_contains(
            gemma_output / "mizu_import" / "tensors.tsv",
            "token_embd.weight|embedding_table|f16|row_major|weights/gemma4.gguf|2816x262144|q5_k",
        )

        broken_model = temp_path / "broken-offset.gguf"
        write_gguf(
            broken_model,
            {
                "general.architecture": ("string", "qwen35"),
                "general.name": ("string", "Broken Qwen3.5"),
                "general.type": ("string", "model"),
            },
            [("token_embd.weight", [4096, 248320], "Q4_K", 4096)],
            payload_bytes=128,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(broken_model),
                "--output-root",
                str(temp_path / "broken_mizu"),
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
            "GGUF importer should reject tensors that point beyond EOF",
            completed,
            "points beyond EOF",
        )

        extent_broken_model = temp_path / "broken-extent.gguf"
        write_gguf(
            extent_broken_model,
            {
                "general.architecture": ("string", "qwen35"),
                "general.name": ("string", "Broken Extent Qwen3.5"),
                "general.type": ("string", "model"),
            },
            [("token_embd.weight", [16, 16], "F32", 0)],
            payload_bytes=8,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(extent_broken_model),
                "--output-root",
                str(temp_path / "broken_extent_mizu"),
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
            "GGUF importer should reject tensors whose full encoded extent runs beyond EOF",
            completed,
            "points beyond EOF",
        )

    print("test_gguf_to_mizu: PASS")
    return 0


def write_gguf(
    path: Path,
    metadata: dict[str, tuple[str, object]],
    tensors: list[tuple[str, list[int], str, int]],
    payload_bytes: int | None = None,
) -> None:
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", len(tensors)))
        handle.write(struct.pack("<Q", len(metadata)))
        for key, (value_type, value) in metadata.items():
            write_string(handle, key)
            handle.write(struct.pack("<I", VALUE_TYPES[value_type]))
            if value_type == "string":
                write_string(handle, str(value))
            elif value_type == "bool":
                handle.write(struct.pack("<?", bool(value)))
            elif value_type == "uint32":
                handle.write(struct.pack("<I", int(value)))
            else:
                raise AssertionError(f"unsupported metadata type in fixture: {value_type}")

        for name, shape, ggml_type, offset in tensors:
            write_string(handle, name)
            handle.write(struct.pack("<I", len(shape)))
            for dim in shape:
                handle.write(struct.pack("<Q", dim))
            handle.write(struct.pack("<I", GGML_TYPES[ggml_type]))
            handle.write(struct.pack("<Q", offset))

        if payload_bytes is None:
            alignment = int(metadata.get("general.alignment", ("uint32", 32))[1])
            header_end = handle.tell()
            padding = (alignment - (header_end % alignment)) % alignment
            payload_bytes = padding + max(
                offset + tensor_byte_size(shape, ggml_type)
                for _, shape, ggml_type, offset in tensors
            )
        if payload_bytes > 0:
            handle.seek(payload_bytes - 1, 1)
            handle.write(b"\0")


def write_string(handle: object, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def tensor_byte_size(shape: list[int], ggml_type: str) -> int:
    block_elements, block_bytes = GGML_QUANT_SIZES[ggml_type]
    row_elements = shape[0]
    if row_elements % block_elements != 0:
        raise AssertionError(f"fixture shape {shape} is incompatible with {ggml_type}")
    row_count = 1
    for dim in shape[1:]:
        row_count *= dim
    return row_count * ((row_elements // block_elements) * block_bytes)


def expect_file_contains(path: Path, needle: str) -> None:
    expect_path_exists(path)
    expect_contains(path.read_text(encoding="utf-8"), needle)


def expect_contains(haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"missing expected text: {needle}")


def find_tsv_row(path: Path, first_field: str) -> list[str]:
    expect_path_exists(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if fields and fields[0] == first_field:
            return fields
    raise AssertionError(f"missing TSV row for {first_field} in {path}")


def expect_equal_list(label: str, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {actual} != {expected}")


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
