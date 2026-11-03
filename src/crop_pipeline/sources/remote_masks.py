"""Read Sen4AgriNet reference masks with strict, bounded HTTP byte ranges.

A server that ignores Range must never cause a full imagery payload download.
Only small mask-only NetCDFs are cached; old source NetCDFs remain untouched.
"""

import hashlib
import io
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from crop_pipeline.common import save_json
from crop_pipeline.windows.seasons import file_sha256

CACHE_VERSION = 1
BLOCK_BYTES = 64 * 1024
CACHE_BLOCKS = 32
VARIABLE_ATTRIBUTES = (
    "crs",
    "transform",
    "missing_value",
    "scale_factor",
    "add_offset",
    "valid_min",
    "valid_max",
    "valid_range",
    "_Unsigned",
)


class RangeReader(io.RawIOBase):
    """A seekable HDF5 input backed by an LRU of small, validated remote ranges."""

    def __init__(
        self,
        session,
        url,
        size,
        headers=None,
        block_bytes=BLOCK_BYTES,
        cache_blocks=CACHE_BLOCKS,
    ):
        super().__init__()
        if size <= 0 or block_bytes <= 0 or cache_blocks <= 0:
            raise ValueError("Remote file and range-cache sizes must be positive.")
        self.session, self.url, self.size = session, url, int(size)
        self.headers = dict(headers or {})
        self.block_bytes, self.cache_blocks = block_bytes, cache_blocks
        self.position = 0
        self.blocks = OrderedDict()
        self.bytes_received, self.range_requests = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origins or origins[whence] + offset < 0:
            raise ValueError("Invalid remote seek")
        self.position = origins[whence] + offset
        return self.position

    def readinto(self, buffer):
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError(
                "Unbounded remote reads are disabled; no full-file fallback is allowed."
            )
        end = min(self.size, self.position + size)
        pieces = []
        while self.position < end:
            block_number, offset = divmod(self.position, self.block_bytes)
            block = self.read_block(block_number)
            amount = min(end - self.position, len(block) - offset)
            pieces.append(block[offset : offset + amount])
            self.position += amount
        return b"".join(pieces)

    def read_block(self, block_number):
        if block_number in self.blocks:
            self.blocks.move_to_end(block_number)
            return self.blocks[block_number]
        start = block_number * self.block_bytes
        end = min(self.size, start + self.block_bytes)
        content = self.fetch_range(start, end)
        self.blocks[block_number] = content
        if len(self.blocks) > self.cache_blocks:
            self.blocks.popitem(last=False)
        return content

    def fetch_range(self, start, end):
        headers = {
            **self.headers,
            "Range": f"bytes={start}-{end-1}",
            "Accept-Encoding": "identity",
        }
        with self.session.get(
            self.url, headers=headers, stream=True, timeout=(30, 180)
        ) as response:
            validate_range_response(response, start, end, self.size)
            content = read_bounded_response(response, end - start)
        self.bytes_received += len(content)
        self.range_requests += 1
        return content


def validate_range_response(response, start, end, size):
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(
            "Server did not honor HTTP Range. Stopped before reading its body; full NetCDF downloads are disabled."
        )
    expected = f"bytes {start}-{end-1}/{size}"
    if response.headers.get("Content-Range") != expected:
        raise ValueError(
            "Remote Content-Range differs from the pinned file's offsets or size."
        )
    if response.headers.get("Content-Encoding", "identity").lower() not in {
        "identity",
        "",
    }:
        raise ValueError(
            "Encoded range responses cannot establish exact source offsets."
        )


def read_bounded_response(response, expected_bytes):
    pieces, received = [], 0
    for block in response.iter_content(chunk_size=min(BLOCK_BYTES, expected_bytes + 1)):
        received += len(block)
        if received > expected_bytes:
            raise ValueError("Remote range response exceeded its requested byte count.")
        pieces.append(block)
    if received != expected_bytes:
        raise ValueError("Remote range response was truncated.")
    return b"".join(pieces)


def cache_paths(folder, record, crop_codes):
    identity = {
        "source": record,
        "crop_codes": list(crop_codes),
        "cache_version": CACHE_VERSION,
    }
    identifier = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    # Masks survive feature-extractor changes, but are isolated by upstream release and identity.
    root = Path(folder).parents[1] / "reference_masks"
    return root / f"{identifier}.nc", root / f"{identifier}.json", identity


def cached_reference_source(folder, record, crop_codes):
    mask_path, receipt_path, identity = cache_paths(folder, record, crop_codes)
    if not mask_path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("identity") != identity or receipt.get("mask_sha256") != file_sha256(
        mask_path
    ):
        return None
    return mask_path, {
        **receipt,
        "reused_mask_cache": True,
        "network_bytes_this_run": 0,
    }


def obtain_reference_masks(record, manifest, folder, crop_codes, enabled=True):
    cached = cached_reference_source(folder, record, crop_codes)
    if cached is not None:
        return cached
    if not enabled:
        raise FileNotFoundError(
            f"Missing cached reference masks for {record['filename']}; enable reference retrieval."
        )
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(
        repo_id=manifest["dataset"],
        repo_type="dataset",
        revision=manifest["revision"],
        filename=record["filename"],
    )
    with requests.Session() as session:
        reader = RangeReader(session, url, record["expected_bytes"], build_hf_headers())
        return publish_reference_masks(
            reader, record, folder, crop_codes, manifest["revision"]
        )


def publish_reference_masks(reader, record, folder, crop_codes, revision):
    mask_path, receipt_path, identity = cache_paths(folder, record, crop_codes)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".masks_", suffix=".nc", dir=mask_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        has_targets = write_mask_netcdf(reader, temporary, crop_codes)
        receipt = {
            "identity": identity,
            "upstream_revision": revision,
            "mask_sha256": file_sha256(temporary),
            "saved_mask_bytes": temporary.stat().st_size,
            "source_size_bytes": record["expected_bytes"],
            "network_bytes": reader.bytes_received,
            "range_requests": reader.range_requests,
            "network_bytes_this_run": reader.bytes_received,
            "target_labels_present": has_targets,
            "retrieval": "http_range_masks_only",
            "full_source_checksum_verified": False,
        }
        temporary.replace(mask_path)
        save_json(receipt_path, receipt)
        return mask_path, receipt
    finally:
        temporary.unlink(missing_ok=True)


def write_mask_netcdf(reader, destination, crop_codes):
    """Copy raw masks and CF attributes so the existing netCDF4 decoder keeps its semantics."""
    import h5py

    with h5py.File(reader, "r") as source, Dataset(destination, "w") as target:
        copy_patch_attributes(source, target)
        masks = find_masks(source, h5py.Dataset)
        labels = copy_mask(masks["labels"], target)
        has_targets = target_labels_present(labels, crop_codes)
        if has_targets:
            copy_parcel_mask(masks, target)
        return has_targets


def copy_patch_attributes(source, target):
    for name in ("patch_year", "patch_country_code", "patch_tile"):
        if name in source.attrs:
            target.setncattr(name, source.attrs[name])


def copy_parcel_mask(masks, target):
    if "parcels" not in masks:
        raise ValueError("Target-crop patch is missing its parcel mask.")
    copy_mask(masks["parcels"], target)


def find_masks(source, dataset_type):
    masks = {}

    def inspect(name, node):
        basename = name.rsplit("/", 1)[-1].lower()
        if isinstance(node, dataset_type) and basename in {"labels", "parcels"}:
            masks.setdefault(basename, []).append(node)

    source.visititems(inspect)
    if len(masks.get("labels", [])) != 1 or len(masks.get("parcels", [])) > 1:
        raise ValueError(
            "Remote patch requires unambiguous label and parcel mask names."
        )
    return {name: nodes[0] for name, nodes in masks.items()}


def target_labels_present(variable, crop_codes):
    values = np.ma.asarray(variable[:]).compressed()
    values = values[np.isfinite(values)]
    if variable.ndim != 2 or np.any(values != np.floor(values)):
        raise ValueError("Reference labels must be a two-dimensional integer mask.")
    return bool(np.isin(values, crop_codes).any())


def output_group(target, name):
    group = target
    for component in [part for part in name.split("/") if part]:
        group = (
            group.groups[component]
            if component in group.groups
            else group.createGroup(component)
        )
    return group


def copy_mask(variable, target):
    group = output_group(target, variable.parent.name)
    for axis in ("x", "y"):
        coordinate = variable.parent.get(axis)
        if coordinate is None or coordinate.ndim != 1:
            raise ValueError("Mask group lacks one-dimensional x/y coordinates.")
        copy_coordinate(coordinate, group, axis)
    return copy_variable(variable, group, ("y", "x"))


def copy_coordinate(coordinate, group, name):
    if name not in group.dimensions:
        group.createDimension(name, len(coordinate))
    if name not in group.variables:
        copy_variable(coordinate, group, (name,))


def copy_variable(source, target, dimensions):
    name = source.name.rsplit("/", 1)[-1]
    fill_value = source.attrs.get("_FillValue")
    output = target.createVariable(
        name,
        source.dtype.newbyteorder("="),
        dimensions,
        fill_value=fill_value,
        zlib=True,
        complevel=4,
    )
    # Write raw values before scale/offset attributes to avoid scaling twice.
    output[:] = source[...]
    for attribute in VARIABLE_ATTRIBUTES:
        if attribute in source.attrs:
            output.setncattr(attribute, source.attrs[attribute])
    return output
