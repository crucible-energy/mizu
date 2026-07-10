#!/usr/bin/env python3
"""Create a Mizu import bundle from local GGUF model assets.

This is a header-only smoke importer. It reads GGUF metadata and tensor-info
records directly, normalizes common Qwen/Gemma/CLIP tensor names into Mizu's
import bundle dialect, and symlinks or copies the source GGUF files under
`mizu_import/weights/`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any


IMPORT_LAYOUT_VERSION = 1
MAX_SAFE_I64 = (1 << 63) - 1
MAX_REASONABLE_COUNT = 10_000_000

GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGML_TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    31: "q4_0_4_4",
    32: "q4_0_4_8",
    33: "q4_0_8_8",
    34: "tq1_0",
    35: "tq2_0",
}

DIRECT_DTYPE_MAP = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "bf16",
    "i32": "i32",
}

GGML_QUANT_SIZES = {
    "f32": (1, 4),
    "f16": (1, 2),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q8_1": (32, 36),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "q8_k": (256, 292),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq3_xxs": (256, 98),
    "iq1_s": (256, 50),
    "iq4_nl": (32, 18),
    "iq3_s": (256, 110),
    "iq2_s": (256, 82),
    "iq4_xs": (256, 136),
    "i8": (1, 1),
    "i16": (1, 2),
    "i32": (1, 4),
    "i64": (1, 8),
    "f64": (1, 8),
    "iq1_m": (256, 56),
    "bf16": (1, 2),
    # These ARM-repacked formats retain q4_0 storage byte counts.
    "q4_0_4_4": (32, 18),
    "q4_0_4_8": (32, 18),
    "q4_0_8_8": (32, 18),
    "tq1_0": (256, 54),
    "tq2_0": (256, 66),
}


@dataclass(frozen=True)
class MetadataValue:
    value_type: str
    value: Any


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: list[int]
    ggml_type: str
    data_offset: int
    source_offset: int
    source_kind: str
    source_path: Path
    bundle_rel: Path


@dataclass(frozen=True)
class GgufFile:
    path: Path
    source_kind: str
    version: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, MetadataValue]
    tensors: list[GgufTensor]
    file_size: int


class GgufImportError(Exception):
    pass


def main() -> int:
    args = parse_args()
    model_gguf = args.model_gguf.resolve()
    output_root = (args.output_root or default_output_root(model_gguf)).resolve()

    if not model_gguf.exists() or not model_gguf.is_file():
        print(f"model GGUF does not exist or is not a file: {model_gguf}", file=sys.stderr)
        return 2

    projector_gguf = args.projector_gguf.resolve() if args.projector_gguf else None
    if projector_gguf is not None and (not projector_gguf.exists() or not projector_gguf.is_file()):
        print(f"projector GGUF does not exist or is not a file: {projector_gguf}", file=sys.stderr)
        return 2

    try:
        bundle = build_bundle(model_gguf, projector_gguf, output_root, args)
    except GgufImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print_summary(bundle)
        return 0

    try:
        write_bundle(bundle, force=args.force, link_mode=args.link_mode)
    except GgufImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print_summary(bundle)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_gguf", type=Path, help="local GGUF model file")
    parser.add_argument("--projector-gguf", type=Path, default=None, help="optional GGUF mmproj/vision file")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="where to write manifest.mizu and mizu_import/ (defaults beside the model GGUF)",
    )
    parser.add_argument(
        "--family",
        choices=("auto", "qwen3_5", "gemma4"),
        default="auto",
        help="target Mizu model family",
    )
    parser.add_argument("--source-model-id", default="", help="override source_model_id")
    parser.add_argument("--source-revision", default="", help="override source_revision")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="how to materialize GGUF files under mizu_import/weights",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing generated files")
    parser.add_argument("--dry-run", action="store_true", help="print detected mapping without writing")
    return parser.parse_args()


def default_output_root(model_gguf: Path) -> Path:
    return model_gguf.with_suffix("").with_name(f"{model_gguf.stem}.mizu")


def build_bundle(model_gguf: Path, projector_gguf: Path | None, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    gguf_files = [parse_gguf_file(model_gguf, source_kind="model")]
    if projector_gguf is not None:
        gguf_files.append(parse_gguf_file(projector_gguf, source_kind="projector"))
    ensure_unique_bundle_filenames(gguf_files)

    tensors: list[dict[str, Any]] = []
    for gguf_file in gguf_files:
        general_type = metadata_text(gguf_file.metadata, "general.type")
        for tensor in gguf_file.tensors:
            role = classify_tensor_role(tensor.name, tensor.source_kind, general_type)
            dtype = normalize_ggml_dtype(tensor.ggml_type)
            tensors.append(
                {
                    "name": tensor.name,
                    "role": role,
                    "dtype": dtype,
                    "ggml_type": tensor.ggml_type,
                    "layout": infer_layout_name(role, tensor.shape),
                    "source_kind": tensor.source_kind,
                    "source_path": tensor.source_path,
                    "bundle_rel": tensor.bundle_rel,
                    "shape": tensor.shape,
                    "data_offset": tensor.data_offset,
                    "source_offset": tensor.source_offset,
                }
            )

    if not tensors:
        raise GgufImportError(f"no tensors found in GGUF header for {model_gguf}")

    family = resolve_family(args.family, gguf_files[0])
    source_model_id = resolve_source_model_id(args.source_model_id, gguf_files[0], model_gguf)
    source_revision = resolve_source_revision(args.source_revision, gguf_files)
    tokenizer_name = resolve_tokenizer_name(gguf_files[0], family)
    source_hash_text = build_source_hash_text(source_model_id, source_revision, gguf_files, tensors)
    projector_tensors = [tensor for tensor in tensors if is_projector_side_role(tensor["role"])]
    has_projector = bool(projector_tensors)
    projector_revision = stable_positive_i64(source_hash_text + ":projector")

    return {
        "model_gguf": model_gguf,
        "output_root": output_root,
        "import_root": output_root / "mizu_import",
        "family": family,
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_hash_text": source_hash_text,
        "tokenizer_name": tokenizer_name,
        "gguf_files": gguf_files,
        "tensors": tensors,
        "has_projector": has_projector,
        "projector_revision": projector_revision,
        "projector_tensors": projector_tensors,
        "link_mode": args.link_mode,
    }


def ensure_unique_bundle_filenames(gguf_files: list[GgufFile]) -> None:
    seen_names: set[str] = set()
    for gguf_file in gguf_files:
        if gguf_file.path.name in seen_names:
            raise GgufImportError(f"GGUF basename would collide under mizu_import/weights: {gguf_file.path.name}")
        seen_names.add(gguf_file.path.name)


def parse_gguf_file(path: Path, source_kind: str) -> GgufFile:
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic != b"GGUF":
            raise GgufImportError(f"invalid GGUF magic in {path}")
        version = read_u32(handle)
        if version not in (2, 3):
            raise GgufImportError(f"unsupported GGUF version in {path}: {version}")
        tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)
        if tensor_count <= 0 or tensor_count > MAX_REASONABLE_COUNT:
            raise GgufImportError(f"unreasonable tensor count in {path}: {tensor_count}")
        if metadata_count > MAX_REASONABLE_COUNT:
            raise GgufImportError(f"unreasonable metadata count in {path}: {metadata_count}")

        metadata: dict[str, MetadataValue] = {}
        for _ in range(metadata_count):
            key = read_gguf_string(handle)
            metadata[key] = read_metadata_value(handle)

        bundle_rel = Path("weights") / path.name
        tensor_records: list[tuple[str, list[int], str, int]] = []
        for _ in range(tensor_count):
            name = read_gguf_string(handle)
            rank = read_u32(handle)
            if rank <= 0 or rank > 8:
                raise GgufImportError(f"tensor {name} in {path} has invalid rank {rank}")
            shape = [read_u64(handle) for _ in range(rank)]
            if any(dim <= 0 or dim > MAX_REASONABLE_COUNT for dim in shape):
                raise GgufImportError(f"tensor {name} in {path} has invalid shape {shape}")
            ggml_type_id = read_u32(handle)
            ggml_type = GGML_TYPES.get(ggml_type_id)
            if ggml_type is None:
                raise GgufImportError(f"tensor {name} in {path} has unsupported GGML type id {ggml_type_id}")
            data_offset = read_u64(handle)
            if data_offset > MAX_SAFE_I64:
                raise GgufImportError(f"tensor {name} in {path} has unreasonable data offset {data_offset}")
            tensor_records.append((name, shape, ggml_type, data_offset))

        alignment = metadata_int(metadata, "general.alignment", 32)
        if alignment <= 0:
            alignment = 32
        tensor_data_start = align_offset(handle.tell(), alignment)
        file_size = path.stat().st_size
        if tensor_data_start >= file_size:
            raise GgufImportError(f"GGUF tensor data starts beyond EOF in {path}")
        tensors: list[GgufTensor] = []
        for name, shape, ggml_type, data_offset in tensor_records:
            source_offset = tensor_data_start + data_offset
            if source_offset > MAX_SAFE_I64:
                raise GgufImportError(f"tensor {name} in {path} has unreasonable source offset {source_offset}")
            if source_offset >= file_size:
                raise GgufImportError(f"tensor {name} in {path} points beyond EOF at offset {source_offset}")
            encoded_tensor_bytes = ggml_tensor_nbytes(path, name, shape, ggml_type)
            if encoded_tensor_bytes > file_size - source_offset:
                raise GgufImportError(
                    f"tensor {name} in {path} points beyond EOF at offset {source_offset} "
                    f"with byte size {encoded_tensor_bytes}"
                )
            tensors.append(
                GgufTensor(
                    name=name,
                    shape=shape,
                    ggml_type=ggml_type,
                    data_offset=data_offset,
                    source_offset=source_offset,
                    source_kind=source_kind,
                    source_path=path,
                    bundle_rel=bundle_rel,
                )
            )

    return GgufFile(
        path=path,
        source_kind=source_kind,
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        metadata=metadata,
        tensors=tensors,
        file_size=file_size,
    )


def read_metadata_value(handle: BinaryIO) -> MetadataValue:
    value_type_id = read_u32(handle)
    value_type = GGUF_VALUE_TYPES.get(value_type_id)
    if value_type is None:
        raise GgufImportError(f"unsupported GGUF metadata value type id {value_type_id}")
    if value_type_id == 9:
        element_type_id = read_u32(handle)
        element_type = GGUF_VALUE_TYPES.get(element_type_id)
        if element_type is None or element_type == "array":
            raise GgufImportError(f"unsupported GGUF metadata array element type id {element_type_id}")
        count = read_u64(handle)
        if count > MAX_REASONABLE_COUNT:
            raise GgufImportError(f"unreasonable GGUF metadata array length: {count}")
        sample = []
        for index in range(count):
            value = read_scalar_value(handle, element_type_id)
            if index < 8:
                sample.append(value)
        return MetadataValue("array", {"element_type": element_type, "count": count, "sample": sample})
    return MetadataValue(value_type, read_scalar_value(handle, value_type_id))


def read_scalar_value(handle: BinaryIO, value_type_id: int) -> Any:
    if value_type_id == 0:
        return struct.unpack("<B", read_exact(handle, 1))[0]
    if value_type_id == 1:
        return struct.unpack("<b", read_exact(handle, 1))[0]
    if value_type_id == 2:
        return struct.unpack("<H", read_exact(handle, 2))[0]
    if value_type_id == 3:
        return struct.unpack("<h", read_exact(handle, 2))[0]
    if value_type_id == 4:
        return read_u32(handle)
    if value_type_id == 5:
        return struct.unpack("<i", read_exact(handle, 4))[0]
    if value_type_id == 6:
        return struct.unpack("<f", read_exact(handle, 4))[0]
    if value_type_id == 7:
        return struct.unpack("<?", read_exact(handle, 1))[0]
    if value_type_id == 8:
        return read_gguf_string(handle)
    if value_type_id == 10:
        return read_u64(handle)
    if value_type_id == 11:
        return struct.unpack("<q", read_exact(handle, 8))[0]
    if value_type_id == 12:
        return struct.unpack("<d", read_exact(handle, 8))[0]
    raise GgufImportError(f"unsupported scalar value type id {value_type_id}")


def read_gguf_string(handle: BinaryIO) -> str:
    byte_count = read_u64(handle)
    if byte_count > 256 * 1024 * 1024:
        raise GgufImportError(f"unreasonable GGUF string length: {byte_count}")
    return read_exact(handle, byte_count).decode("utf-8", "replace")


def read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_exact(handle: BinaryIO, byte_count: int) -> bytes:
    data = handle.read(byte_count)
    if len(data) != byte_count:
        raise GgufImportError("truncated GGUF header")
    return data


def metadata_text(metadata: dict[str, MetadataValue], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    if isinstance(value.value, str):
        return value.value
    return str(value.value)


def metadata_int(metadata: dict[str, MetadataValue], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return int(value.value)
    except (TypeError, ValueError):
        return default


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 1:
        return offset
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def classify_tensor_role(name: str, source_kind: str, general_type: str) -> str:
    lowered = name.lower()
    if source_kind == "projector" or general_type == "mmproj":
        if is_projector_tensor_name(lowered):
            return "multimodal_projector"
        if is_vision_tensor_name(lowered, allow_broad_match=True):
            return "vision_encoder"
        return "multimodal_projector"
    if is_projector_tensor_name(lowered):
        return "multimodal_projector"
    if is_vision_tensor_name(lowered):
        return "vision_encoder"
    if "token_embd" in lowered or "embed_tokens" in lowered or "embedding" in lowered:
        return "embedding_table"
    if lowered == "output.weight" or "lm_head" in lowered or "output_projection" in lowered:
        return "token_projection"
    if "norm" in lowered:
        return "normalization"
    if lowered.startswith("blk.") or ".blk." in lowered or "decoder" in lowered:
        return "decoder_stack"
    if "attn_" in lowered or "ffn_" in lowered or "ssm_" in lowered:
        return "decoder_stack"
    return "model_tensor"


def is_projector_tensor_name(lowered_name: str) -> bool:
    return lowered_name.startswith("mm.") or "projector" in lowered_name or "merger" in lowered_name


def is_vision_tensor_name(lowered_name: str, *, allow_broad_match: bool = False) -> bool:
    if lowered_name.startswith("v.") or "vision" in lowered_name:
        return True
    if not allow_broad_match:
        return False
    return "patch" in lowered_name or "position" in lowered_name


def is_projector_side_role(role_name: str) -> bool:
    return role_name in {"multimodal_projector", "vision_encoder"}


def normalize_ggml_dtype(ggml_type: str) -> str:
    if ggml_type in DIRECT_DTYPE_MAP:
        return DIRECT_DTYPE_MAP[ggml_type]
    if ggml_type == "i8":
        return "u8"
    if ggml_type in {"i16", "i64", "f64"}:
        return "f32"
    return "f16"


def ggml_tensor_nbytes(path: Path, tensor_name: str, shape: list[int], ggml_type: str) -> int:
    block_elements, block_bytes = GGML_QUANT_SIZES.get(ggml_type, (0, 0))
    if block_elements <= 0 or block_bytes <= 0:
        raise GgufImportError(f"tensor {tensor_name} in {path} has unsupported GGML type {ggml_type}")

    row_elements = shape[0]
    if row_elements % block_elements != 0:
        raise GgufImportError(
            f"tensor {tensor_name} in {path} has shape {shape} incompatible with GGML type {ggml_type}"
        )

    row_bytes = (row_elements // block_elements) * block_bytes
    total_rows = 1
    for dim in shape[1:]:
        if total_rows > MAX_SAFE_I64 // dim:
            raise GgufImportError(f"tensor {tensor_name} in {path} has unreasonable row count for shape {shape}")
        total_rows *= dim

    if row_bytes > MAX_SAFE_I64 // total_rows:
        raise GgufImportError(f"tensor {tensor_name} in {path} has unreasonable byte size for shape {shape}")
    return total_rows * row_bytes


def infer_layout_name(role: str, shape: list[int]) -> str:
    if len(shape) == 1:
        return "vector"
    if role in {"decoder_stack", "multimodal_projector", "vision_encoder"}:
        return "packed"
    if len(shape) == 2:
        return "row_major"
    return "tensor"


def resolve_family(requested_family: str, model_file: GgufFile) -> str:
    if requested_family != "auto":
        return requested_family

    identity = " ".join(
        [
            metadata_text(model_file.metadata, "general.architecture"),
            metadata_text(model_file.metadata, "general.name"),
            metadata_text(model_file.metadata, "general.basename"),
            model_file.path.name,
        ]
    ).lower()
    if "qwen" in identity:
        return "qwen3_5"
    if "gemma" in identity:
        return "gemma4"
    raise GgufImportError("could not infer model family; pass --family qwen3_5 or --family gemma4")


def resolve_source_model_id(override: str, model_file: GgufFile, model_gguf: Path) -> str:
    if override:
        return override
    for key in ("general.name", "general.basename", "general.architecture"):
        value = metadata_text(model_file.metadata, key).strip()
        if value:
            return value
    return model_gguf.stem


def resolve_source_revision(override: str, gguf_files: list[GgufFile]) -> str:
    if override:
        return override
    model_file = gguf_files[0]
    file_type = metadata_text(model_file.metadata, "general.file_type").strip()
    quantization_version = metadata_text(model_file.metadata, "general.quantization_version").strip()
    parts = [f"gguf-v{model_file.version}"]
    if file_type:
        parts.append(f"filetype-{file_type}")
    if quantization_version:
        parts.append(f"quantv-{quantization_version}")
    if len(gguf_files) > 1:
        parts.append(f"projector-gguf-v{gguf_files[1].version}")
    return ":".join(parts)


def resolve_tokenizer_name(model_file: GgufFile, family: str) -> str:
    value = metadata_text(model_file.metadata, "tokenizer.ggml.model").strip()
    if value:
        return value
    return family


def build_source_hash_text(
    source_model_id: str,
    source_revision: str,
    gguf_files: list[GgufFile],
    tensors: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(source_model_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_revision.encode("utf-8"))
    for gguf_file in gguf_files:
        digest.update(b"\0file|")
        digest.update(gguf_file.source_kind.encode("utf-8"))
        digest.update(b"|")
        digest.update(gguf_file.path.name.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(gguf_file.file_size).encode("utf-8"))
        digest.update(b"|")
        digest.update(metadata_text(gguf_file.metadata, "general.architecture").encode("utf-8"))
        digest.update(b"|")
        digest.update(metadata_text(gguf_file.metadata, "general.type").encode("utf-8"))
    for tensor in sorted(tensors, key=lambda item: (item["source_kind"], item["name"])):
        digest.update(b"\0tensor|")
        digest.update(tensor["source_kind"].encode("utf-8"))
        digest.update(b"|")
        digest.update(tensor["name"].encode("utf-8"))
        digest.update(b"|")
        digest.update(tensor["ggml_type"].encode("utf-8"))
        digest.update(b"|")
        digest.update("x".join(str(dim) for dim in tensor["shape"]).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(tensor["data_offset"]).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(tensor["source_offset"]).encode("utf-8"))
    return digest.hexdigest()


def stable_positive_i64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") & MAX_SAFE_I64
    return value or 1


def write_bundle(bundle: dict[str, Any], force: bool, link_mode: str) -> None:
    output_root: Path = bundle["output_root"]
    import_root: Path = bundle["import_root"]
    ensure_can_write(output_root / "manifest.mizu", force)
    ensure_can_write(import_root / "layout.mizu", force)
    ensure_can_write(import_root / "tensors.tsv", force)
    ensure_can_write(import_root / "gguf_tensors.tsv", force)
    ensure_can_write(import_root / "modalities.tsv", force)
    ensure_can_write(import_root / "projector.mizu", force)
    ensure_can_write(import_root / "projector" / "projector_assets.mizu", force)

    (import_root / "weights").mkdir(parents=True, exist_ok=True)
    (import_root / "projector").mkdir(parents=True, exist_ok=True)

    write_text(output_root / "manifest.mizu", render_root_manifest(bundle))
    write_text(import_root / "layout.mizu", render_layout(bundle))
    write_text(import_root / "tensors.tsv", render_tensors(bundle))
    write_text(import_root / "gguf_tensors.tsv", render_gguf_tensors(bundle))
    write_text(import_root / "modalities.tsv", render_modalities(bundle))
    write_text(import_root / "projector.mizu", render_projector(bundle))
    write_text(import_root / "projector" / "projector_assets.mizu", render_projector_assets(bundle))
    materialize_gguf_links(bundle, link_mode, force)


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise GgufImportError(f"refusing to overwrite {path}; pass --force")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_root_manifest(bundle: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Generated by tools/import/gguf_to_mizu.py",
            f"family = {bundle['family']}",
            f"source_model_id = {bundle['source_model_id']}",
            f"source_revision = {bundle['source_revision']}",
            f"source_hash_text = {bundle['source_hash_text']}",
            f"tokenizer = {bundle['tokenizer_name']}",
            "model_features = multimodal,projector" if bundle["has_projector"] else "model_features = none",
            f"projector_present = {str(bundle['has_projector']).lower()}",
            "",
        ]
    )


def render_layout(bundle: dict[str, Any]) -> str:
    lines = [
        "# Generated by tools/import/gguf_to_mizu.py",
        f"layout_version = {IMPORT_LAYOUT_VERSION}",
        f"family = {bundle['family']}",
        f"source_model_id = {bundle['source_model_id']}",
        f"source_revision = {bundle['source_revision']}",
        f"source_hash_text = {bundle['source_hash_text']}",
        f"tokenizer = {bundle['tokenizer_name']}",
        "tensor_inventory = tensors.tsv",
        "gguf_inventory = gguf_tensors.tsv",
        "modality_inventory = modalities.tsv" if bundle["has_projector"] else "modality_inventory = -",
        "projector_inventory = projector.mizu" if bundle["has_projector"] else "projector_inventory = -",
        f"model_features = {'multimodal,projector' if bundle['has_projector'] else 'none'}",
        f"projector_present = {str(bundle['has_projector']).lower()}",
    ]
    if bundle["has_projector"]:
        lines.extend(
            [
                "projector_slot = image",
                "projector_placeholder_count = 1",
                "projector_input_dtype = u8",
                "projector_embedding_dtype = bf16",
                f"projector_revision = {bundle['projector_revision']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_tensors(bundle: dict[str, Any]) -> str:
    lines = ["# tensor_name|tensor_role|dtype|layout_name|relative_path|shape|storage_type"]
    for tensor in sorted(bundle["tensors"], key=lambda item: (item["source_kind"], item["name"])):
        lines.append(
            "|".join(
                [
                    tensor["name"],
                    tensor["role"],
                    tensor["dtype"],
                    tensor["layout"],
                    tensor["bundle_rel"].as_posix(),
                    shape_text(tensor["shape"]),
                    tensor["ggml_type"],
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_gguf_tensors(bundle: dict[str, Any]) -> str:
    lines = [
        "# tensor_name|source_kind|ggml_type|normalized_dtype|layout_name|relative_path|data_offset|source_offset|shape"
    ]
    for tensor in sorted(bundle["tensors"], key=lambda item: (item["source_kind"], item["name"])):
        lines.append(
            "|".join(
                [
                    tensor["name"],
                    tensor["source_kind"],
                    tensor["ggml_type"],
                    tensor["dtype"],
                    tensor["layout"],
                    tensor["bundle_rel"].as_posix(),
                    str(tensor["data_offset"]),
                    str(tensor["source_offset"]),
                    shape_text(tensor["shape"]),
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_modalities(bundle: dict[str, Any]) -> str:
    if not bundle["has_projector"]:
        return "# no multimodal projector detected\n"
    return "# placeholder_ordinal|slot_name|modality_kind|storage_kind|dtype\n1|image|image|encoded_bytes|u8\n"


def render_projector(bundle: dict[str, Any]) -> str:
    if not bundle["has_projector"]:
        return "present = false\n"
    return "\n".join(
        [
            "present = true",
            "slot = image",
            "placeholder_count = 1",
            "input_modality_kind = image",
            "input_dtype = u8",
            "embedding_dtype = bf16",
            f"revision_identity = {bundle['projector_revision']}",
            "artifact_path = projector/projector_assets.mizu",
            "",
        ]
    )


def render_projector_assets(bundle: dict[str, Any]) -> str:
    lines = ["# projector tensor inventory"]
    for tensor in sorted(bundle["projector_tensors"], key=lambda item: item["name"]):
        lines.append(
            f"{tensor['name']}|{tensor['bundle_rel'].as_posix()}|offset={tensor['data_offset']}|"
            f"ggml_type={tensor['ggml_type']}|source_offset={tensor['source_offset']}"
        )
    if len(lines) == 1:
        lines.append("# no projector-like tensors were detected")
    lines.append("")
    return "\n".join(lines)


def materialize_gguf_links(bundle: dict[str, Any], link_mode: str, force: bool) -> None:
    import_root: Path = bundle["import_root"]
    for gguf_file in bundle["gguf_files"]:
        target_path = import_root / "weights" / gguf_file.path.name
        if target_path.exists() or target_path.is_symlink():
            if not force:
                raise GgufImportError(f"refusing to replace existing GGUF link {target_path}; pass --force")
            if target_path.is_dir() and not target_path.is_symlink():
                raise GgufImportError(f"cannot replace directory with GGUF link: {target_path}")
            target_path.unlink()
        if link_mode == "copy":
            shutil.copy2(gguf_file.path, target_path)
        else:
            relative_target = os.path.relpath(gguf_file.path, start=target_path.parent)
            target_path.symlink_to(relative_target)


def shape_text(shape: list[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def print_summary(bundle: dict[str, Any]) -> None:
    print(f"family: {bundle['family']}")
    print(f"source_model_id: {bundle['source_model_id']}")
    print(f"source_revision: {bundle['source_revision']}")
    print(f"tensor_count: {len(bundle['tensors'])}")
    print(f"gguf_file_count: {len(bundle['gguf_files'])}")
    print(f"projector_present: {str(bundle['has_projector']).lower()}")
    print(f"output_root: {bundle['output_root']}")


if __name__ == "__main__":
    raise SystemExit(main())
