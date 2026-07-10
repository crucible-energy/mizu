#!/usr/bin/env python3
"""Create a Mizu import bundle from a local HuggingFace safetensors model.

The importer intentionally uses only Python's standard library. It reads
safetensors headers directly, classifies common Qwen/Gemma tensor names into
Mizu's manifest dialect, and writes a small `mizu_import/` bundle that the
Fortran loader can validate before backend math exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any


IMPORT_LAYOUT_VERSION = 1
MAX_SAFE_I64 = (1 << 63) - 1

DTYPE_MAP = {
    "U8": "u8",
    "I32": "i32",
    "F16": "f16",
    "BF16": "bf16",
    "F32": "f32",
}

DTYPE_SIZES = {
    "U8": 1,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
}


def main() -> int:
    args = parse_args()
    model_root = args.model_root.resolve()
    output_root = (args.output_root or args.model_root).resolve()

    if not model_root.exists() or not model_root.is_dir():
        print(f"model root does not exist or is not a directory: {model_root}", file=sys.stderr)
        return 2

    try:
        bundle = build_bundle(model_root, output_root, args)
    except MizuImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print_summary(bundle)
        return 0

    try:
        write_bundle(bundle, force=args.force, link_mode=args.link_mode)
    except MizuImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print_summary(bundle)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path, help="local HuggingFace model directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="where to write manifest.mizu and mizu_import/ (defaults to model_root)",
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
        help="how to materialize safetensors shards under mizu_import/weights",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing generated files")
    parser.add_argument("--dry-run", action="store_true", help="print detected mapping without writing")
    return parser.parse_args()


def build_bundle(model_root: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json_if_exists(model_root / "config.json")
    tokenizer_config = read_json_if_exists(model_root / "tokenizer_config.json")
    shard_paths = discover_safetensor_shards(model_root)
    if not shard_paths:
        raise MizuImportError(f"no .safetensors files found under {model_root}")

    tensors: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        for tensor_name, metadata in read_safetensors_header(shard_path).items():
            if tensor_name == "__metadata__":
                continue
            tensors.append(normalize_tensor_record(model_root, shard_path, tensor_name, metadata))

    if not tensors:
        raise MizuImportError(f"no tensors found in safetensors headers under {model_root}")

    family = resolve_family(args.family, config, model_root)
    source_model_id = resolve_source_model_id(args.source_model_id, config, model_root)
    source_revision = resolve_source_revision(args.source_revision, config)
    tokenizer_name = resolve_tokenizer_name(tokenizer_config, config, family)
    source_hash_text = build_source_hash_text(source_model_id, source_revision, tensors)
    projector_tensors = [tensor for tensor in tensors if tensor["role"] == "multimodal_projector"]
    has_projector = bool(projector_tensors) or config_indicates_projector(config)
    projector_revision = stable_positive_i64(source_hash_text + ":projector")

    return {
        "model_root": model_root,
        "output_root": output_root,
        "import_root": output_root / "mizu_import",
        "family": family,
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_hash_text": source_hash_text,
        "tokenizer_name": tokenizer_name,
        "tensors": tensors,
        "shard_paths": shard_paths,
        "has_projector": has_projector,
        "projector_revision": projector_revision,
        "projector_tensors": projector_tensors,
        "link_mode": args.link_mode,
    }


def discover_safetensor_shards(model_root: Path) -> list[Path]:
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        index_data = read_json(index_path)
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise MizuImportError(f"expected non-empty weight_map in {index_path}")
        shard_names = sorted({str(value) for value in weight_map.values()})
        missing_shards = [model_root / shard_name for shard_name in shard_names if not (model_root / shard_name).exists()]
        if missing_shards:
            missing_text = ", ".join(path.name for path in missing_shards)
            raise MizuImportError(f"safetensors index references missing shard(s): {missing_text}")
        return [model_root / shard_name for shard_name in shard_names]

    return sorted(model_root.glob("*.safetensors"))


def read_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise MizuImportError(f"invalid safetensors header in {path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise MizuImportError(f"unreasonable safetensors header length in {path}: {header_len}")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise MizuImportError(f"truncated safetensors header in {path}")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MizuImportError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    validate_safetensors_header(path, header, header_len, file_size)
    return header


def validate_safetensors_header(path: Path, header: Any, header_len: int, file_size: int) -> None:
    if not isinstance(header, dict):
        raise MizuImportError(f"expected safetensors header object in {path}")

    payload_size = file_size - 8 - header_len
    if payload_size < 0:
        raise MizuImportError(f"safetensors payload starts beyond EOF in {path}")

    for tensor_name, metadata in header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise MizuImportError(f"tensor metadata for {tensor_name} in {path} is not an object")

        data_offsets = metadata.get("data_offsets")
        if not isinstance(data_offsets, list) or len(data_offsets) != 2:
            raise MizuImportError(f"tensor metadata for {tensor_name} in {path} is missing data_offsets")
        start_offset, end_offset = data_offsets
        if not isinstance(start_offset, int) or not isinstance(end_offset, int):
            raise MizuImportError(f"tensor {tensor_name} in {path} has non-integer data_offsets")
        if start_offset < 0 or end_offset <= start_offset:
            raise MizuImportError(f"tensor {tensor_name} in {path} has invalid data_offsets {data_offsets}")
        if end_offset > payload_size:
            raise MizuImportError(
                f"tensor {tensor_name} in {path} points beyond EOF with data_offsets {data_offsets}"
            )
        _, _, expected_size = parse_tensor_shape_and_dtype(path, tensor_name, metadata)
        if end_offset - start_offset != expected_size:
            raise MizuImportError(
                f"tensor {tensor_name} in {path} has data_offsets {data_offsets} "
                f"but expected {expected_size} bytes from dtype/shape"
            )


def normalize_tensor_record(model_root: Path, shard_path: Path, name: str, metadata: Any) -> dict[str, Any]:
    dtype, shape, _ = parse_tensor_shape_and_dtype(shard_path, name, metadata)

    role = classify_tensor_role(name)
    return {
        "name": name,
        "role": role,
        "dtype": dtype,
        "layout": infer_layout_name(role, shape),
        "source_shard": shard_path,
        "source_rel": safe_relative_to(shard_path, model_root),
        "bundle_rel": Path("weights") / shard_path.name,
        "shape": shape,
    }


def parse_tensor_shape_and_dtype(path: Path, tensor_name: str, metadata: Any) -> tuple[str, list[int], int]:
    if not isinstance(metadata, dict):
        raise MizuImportError(f"tensor metadata for {tensor_name} in {path} is not an object")

    dtype_name = str(metadata.get("dtype", "")).upper()
    dtype = DTYPE_MAP.get(dtype_name, "unknown")
    shape = metadata.get("shape", [])
    if dtype == "unknown" or not isinstance(shape, list) or not shape:
        raise MizuImportError(f"tensor metadata for {tensor_name} in {path} is missing dtype/shape")
    if any((not isinstance(dim, int) or dim <= 0) for dim in shape):
        raise MizuImportError(f"tensor {tensor_name} in {path} has invalid shape {shape}")

    element_size = DTYPE_SIZES[dtype_name]
    element_count = product(shape)
    if element_count > MAX_SAFE_I64 // element_size:
        raise MizuImportError(f"tensor {tensor_name} in {path} has unreasonable shape {shape}")
    return dtype, shape, element_count * element_size


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def classify_tensor_role(name: str) -> str:
    lowered = name.lower()
    if "mm_projector" in lowered or "multi_modal_projector" in lowered:
        return "multimodal_projector"
    if "projector" in lowered or "visual.merger" in lowered or "vision_projector" in lowered:
        return "multimodal_projector"
    if is_vision_tensor_name(lowered):
        return "vision_encoder"
    if "embed_tokens" in lowered or "token_embedding" in lowered or "token_embd" in lowered:
        return "embedding_table"
    if lowered.endswith("lm_head.weight") or lowered == "output.weight" or "output_projection" in lowered:
        return "token_projection"
    if "norm" in lowered:
        return "normalization"
    if ".layers." in lowered or ".blocks." in lowered or lowered.startswith("blk.") or ".blk." in lowered:
        return "decoder_stack"
    if "decoder" in lowered or "self_attn" in lowered or ".mlp." in lowered or "attn_" in lowered or "ffn_" in lowered:
        return "decoder_stack"
    return "model_tensor"


def is_vision_tensor_name(lowered_name: str) -> bool:
    return (
        "vision" in lowered_name
        or lowered_name.startswith("visual.")
        or ".visual." in lowered_name
        or lowered_name.startswith("vision_tower.")
        or ".vision_tower." in lowered_name
        or lowered_name.startswith("vision_model.")
        or ".vision_model." in lowered_name
        or lowered_name.startswith("image_tower.")
        or ".image_tower." in lowered_name
    )


def infer_layout_name(role: str, shape: list[int]) -> str:
    if len(shape) == 1:
        return "vector"
    if role in {"decoder_stack", "multimodal_projector", "vision_encoder"}:
        return "packed"
    if len(shape) == 2:
        return "row_major"
    return "tensor"


def resolve_family(requested_family: str, config: dict[str, Any], model_root: Path) -> str:
    if requested_family != "auto":
        return requested_family

    identity = " ".join(
        str(value)
        for value in (
            config.get("model_type", ""),
            config.get("_name_or_path", ""),
            model_root.name,
            model_root.as_posix(),
        )
    ).lower()
    if "qwen" in identity:
        return "qwen3_5"
    if "gemma" in identity:
        return "gemma4"
    raise MizuImportError("could not infer model family; pass --family qwen3_5 or --family gemma4")


def resolve_source_model_id(override: str, config: dict[str, Any], model_root: Path) -> str:
    if override:
        return override
    for key in ("_name_or_path", "name_or_path", "model_type"):
        value = str(config.get(key, "")).strip()
        if value:
            return value
    return model_root.name


def resolve_source_revision(override: str, config: dict[str, Any]) -> str:
    if override:
        return override
    for key in ("_commit_hash", "revision", "transformers_version"):
        value = str(config.get(key, "")).strip()
        if value:
            return value
    return "imported-local"


def resolve_tokenizer_name(tokenizer_config: dict[str, Any], config: dict[str, Any], family: str) -> str:
    for source in (tokenizer_config, config):
        for key in ("tokenizer_class", "model_type"):
            value = str(source.get(key, "")).strip()
            if value:
                return value
    if family == "qwen3_5":
        return "qwen3_5"
    if family == "gemma4":
        return "gemma4"
    return "unknown"


def config_indicates_projector(config: dict[str, Any]) -> bool:
    lowered_keys = " ".join(str(key).lower() for key in config.keys())
    if "vision" in lowered_keys or "projector" in lowered_keys:
        return True
    for key in ("vision_config", "visual", "mm_vision_tower"):
        if key in config:
            return True
    return False


def build_source_hash_text(source_model_id: str, source_revision: str, tensors: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(source_model_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_revision.encode("utf-8"))
    for tensor in sorted(tensors, key=lambda item: item["name"]):
        digest.update(b"\0")
        digest.update(tensor["name"].encode("utf-8"))
        digest.update(b"|")
        digest.update(tensor["dtype"].encode("utf-8"))
        digest.update(b"|")
        digest.update("x".join(str(dim) for dim in tensor["shape"]).encode("utf-8"))
        digest.update(b"|")
        digest.update(tensor["source_rel"].as_posix().encode("utf-8"))
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
    ensure_can_write(import_root / "modalities.tsv", force)
    ensure_can_write(import_root / "projector.mizu", force)
    ensure_can_write(import_root / "projector" / "projector_assets.mizu", force)

    (import_root / "weights").mkdir(parents=True, exist_ok=True)
    (import_root / "projector").mkdir(parents=True, exist_ok=True)

    write_text(output_root / "manifest.mizu", render_root_manifest(bundle))
    write_text(import_root / "layout.mizu", render_layout(bundle))
    write_text(import_root / "tensors.tsv", render_tensors(bundle))
    write_text(import_root / "modalities.tsv", render_modalities(bundle))
    write_text(import_root / "projector.mizu", render_projector(bundle))
    write_text(import_root / "projector" / "projector_assets.mizu", render_projector_assets(bundle))
    materialize_shard_links(bundle, link_mode, force)


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise MizuImportError(f"refusing to overwrite {path}; pass --force")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_root_manifest(bundle: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Generated by tools/import/hf_safetensors_to_mizu.py",
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
        "# Generated by tools/import/hf_safetensors_to_mizu.py",
        f"layout_version = {IMPORT_LAYOUT_VERSION}",
        f"family = {bundle['family']}",
        f"source_model_id = {bundle['source_model_id']}",
        f"source_revision = {bundle['source_revision']}",
        f"source_hash_text = {bundle['source_hash_text']}",
        f"tokenizer = {bundle['tokenizer_name']}",
        "tensor_inventory = tensors.tsv",
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
    for tensor in sorted(bundle["tensors"], key=lambda item: item["name"]):
        shape_text = "x".join(str(dim) for dim in tensor["shape"])
        lines.append(
            "|".join(
                [
                    tensor["name"],
                    tensor["role"],
                    tensor["dtype"],
                    tensor["layout"],
                    tensor["bundle_rel"].as_posix(),
                    shape_text,
                    tensor["dtype"],
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
        lines.append(f"{tensor['name']}|{tensor['bundle_rel'].as_posix()}")
    if len(lines) == 1:
        lines.append("# no projector-like tensors were detected; config requested projector presence")
    lines.append("")
    return "\n".join(lines)


def materialize_shard_links(bundle: dict[str, Any], link_mode: str, force: bool) -> None:
    import_root: Path = bundle["import_root"]
    for shard_path in bundle["shard_paths"]:
        target_path = import_root / "weights" / shard_path.name
        if target_path.exists() or target_path.is_symlink():
            if not force:
                raise MizuImportError(f"refusing to replace existing shard link {target_path}; pass --force")
            if target_path.is_dir() and not target_path.is_symlink():
                raise MizuImportError(f"cannot replace directory with shard link: {target_path}")
            target_path.unlink()
        if link_mode == "copy":
            shutil.copy2(shard_path, target_path)
        else:
            relative_target = os.path.relpath(shard_path, start=target_path.parent)
            target_path.symlink_to(relative_target)


def print_summary(bundle: dict[str, Any]) -> None:
    print(f"family: {bundle['family']}")
    print(f"source_model_id: {bundle['source_model_id']}")
    print(f"source_revision: {bundle['source_revision']}")
    print(f"tensor_count: {len(bundle['tensors'])}")
    print(f"shard_count: {len(bundle['shard_paths'])}")
    print(f"projector_present: {str(bundle['has_projector']).lower()}")
    print(f"output_root: {bundle['output_root']}")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MizuImportError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MizuImportError(f"expected JSON object in {path}")
    return data


def safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


class MizuImportError(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
