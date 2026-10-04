"""Reproduce the four P03 review findings against the committed contracts.

Each probe starts from a fresh fixture built by tests/contract/test_records.py.
A probe prints ACCEPTED when the contract wrongly accepts the malformed input
and REJECTED when validation raises.
"""

import copy
import math
import runpy

import numpy as np
from pydantic import ValidationError

from floscan.contracts.geometry import Pose
from floscan.contracts.result import SECTIONS, PropertyResult

fixtures = runpy.run_path("tests/contract/test_records.py")
result, pose = fixtures["_result"], fixtures["_pose"]


def probe(name, build, model=PropertyResult):
    try:
        model.model_validate(build())
        print(f"ACCEPTED  {name}")
    except ValidationError as error:
        first = error.errors()[0]["msg"].replace("\n", " ")[:90]
        print(f"REJECTED  {name}: {first}")


def empty_success():
    d = result()
    for key in (
        "rooms",
        "surfaces",
        "walls",
        "openings",
        "measurements",
        "coordinate_frames",
    ):
        d[key] = []
    d["capture_id"] = None
    d["coverage"] = {name: {"status": "available"} for name in SECTIONS}
    d["status"], d["status_reason"] = "ok", None
    return d


def mutated(path, value):
    def build():
        d = result()
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return d

    return build


def duplicate_frame():
    d = result()
    d["coordinate_frames"].append(copy.deepcopy(d["coordinate_frames"][1]))
    return d


def indefinite_covariance():
    p = pose("p1", "m")
    c = np.eye(6)
    c[0, 1] = c[1, 0] = 2.0
    p["covariance_status"], p["covariance"] = "known", c.tolist()
    print(f"          (smallest eigenvalue {np.linalg.eigvalsh(c).min():+.1f})")
    return p


print("## Finding 1: empty result claiming complete success")
probe("empty result, every section available, status ok", empty_success)
print("## Finding 2: references")
probe(
    "opening height_measurement_id nonexistent",
    mutated(("openings", 0, "height_measurement_id"), "nonexistent"),
)
probe(
    "wall thickness_measurement_id nonexistent",
    mutated(("walls", 0, "thickness_measurement_id"), "nonexistent"),
)
probe(
    "opening width bound to the wall-length measurement",
    mutated(("openings", 0, "width_measurement_id"), "m-wall"),
)
probe("duplicate coordinate frame R1", duplicate_frame)
print("## Finding 3: covariance")
probe("indefinite covariance as known", indefinite_covariance, model=Pose)
print("## Finding 4: nested mutation")
r = PropertyResult.model_validate(result())
try:
    r.surfaces[0].origin[0] = math.nan
    print("ACCEPTED  nested list assignment of NaN")
except TypeError as error:
    print(f"REJECTED  nested list assignment: {error}")
try:
    text = r.to_json()
    origin = text.split('"origin": [')[1].split("]")[0].split()
    print(
        f"          serialized origin starts {origin[0]} (silent NaN -> null)"
        if "null" in origin[0]
        else f"          serialized origin starts {origin[0]}"
    )
except (ValidationError, ValueError) as error:
    print(f"REJECTED  serialization: {type(error).__name__}")
try:
    r.coverage["measurements"] = r.coverage["capture"]
    print("ACCEPTED  nested dict assignment")
except TypeError as error:
    print(f"REJECTED  nested dict assignment: {error}")
