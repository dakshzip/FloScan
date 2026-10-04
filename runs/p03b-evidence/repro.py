"""Reproduce the two P03A review findings against the committed contracts."""

import copy
import runpy

from pydantic import ValidationError

from floscan.contracts.result import PropertyResult, ReconstructionBundle

fx = runpy.run_path("tests/contract/test_records.py")


def probe(name, model, data):
    try:
        model.model_validate(data)
        print(f"ACCEPTED  {name}")
    except ValidationError as error:
        print(f"REJECTED  {name}: {error.errors()[0]['msg'][:100]}")


depth = {
    "id": "depth-1",
    "provenance": fx["PROVENANCE"],
    "frame_id": "frame-1",
    "camera_id": "cam-1",
    "timestamp_s": 0.0,
    "depth": {
        "uri": "d.npy",
        "sha256": fx["HASH"],
        "byte_count": 4,
        "mime_type": "x",
        "shape": [192, 256],
        "dtype": "<f4",
    },
    "depth_kind": "optical_z",
    "valid_mask": {
        "uri": "m.npy",
        "sha256": fx["HASH"],
        "byte_count": 4,
        "mime_type": "x",
        "shape": [192, 256],
        "dtype": "|b1",
    },
    "source_unit": "mm",
    "scale_to_m": 0.001,
    "alignment": "aligned_rgb",
    "sync_residual_s": 0.0,
}
first = fx["_frame"](depth="depth-1")
second = {**fx["_frame"](depth="depth-1"), "id": "frame-2", "source_index": 1}
probe(
    "frame-2 consumes depth-1 owned by frame-1",
    ReconstructionBundle,
    fx["_bundle"](
        frames=[first, second], cameras=[fx["_camera"]()], depth_frames=[depth]
    ),
)

complete = fx["_complete_result"]()
complete["property_graph"]["nodes"].append(
    copy.deepcopy(complete["property_graph"]["nodes"][0])
)
probe("duplicate graph node node-1 with status ok", PropertyResult, complete)

edge = {
    "id": "edge-1",
    "node_ids": ["node-1", "node-2"],
    "type": "portal",
    "residual_unit": "m",
    "observability_rank": 6,
    "robust_kernel": {"name": "huber", "scale": 0.05},
    "source_ids": ["obs-1"],
    "evidence_group": "group-1",
    "inlier_count": 12,
    "spatial_spread_m": 0.8,
    "accepted": True,
}
graph_edges = fx["_complete_result"]()
graph_edges["property_graph"]["nodes"].append(
    {"id": "node-2", "frame_id": "P", "scale_state": "metric"}
)
graph_edges["property_graph"]["edges"] = [edge, copy.deepcopy(edge)]
probe("duplicate graph edge edge-1", PropertyResult, graph_edges)
