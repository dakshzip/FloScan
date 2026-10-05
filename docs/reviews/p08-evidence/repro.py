"""P08 review probes using analytic fixtures and read-only local bundles."""

import runpy
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from floscan.geometry.rooms import SceneEvidence, build_rooms, load_bundle


def main():
    helpers = runpy.run_path("tests/synthetic/test_rooms.py")
    horizontal = helpers["horizontal"]
    wall_scene = helpers["room_walls"]
    scene = helpers["scene"]
    table = Polygon([(1, 1), (3.5, 1), (3.5, 2.5), (1, 2.5)])
    model = build_rooms(scene([horizontal("table", table, 0.75, "up")], [(2, 1.5)]))
    print("1. Table only, actual floor unseen")
    print("   floor diagnostic", model.diagnostics["floor"])
    print("   surfaces", [(s.kind, s.status) for s in model.surfaces])

    a = [(0, 0), (4, 0), (4, 3), (0, 3)]
    b = [(6, 0), (10, 0), (10, 3), (6, 3)]
    planes = wall_scene(a, (2, 1.5), prefix="a")
    planes += wall_scene(b, (8, 1.5), prefix="b")
    planes += [
        horizontal("floor-a", Polygon(a), 0, "up"),
        horizontal("floor-b", Polygon(b), 0, "up"),
        horizontal("ceiling-a", Polygon(a), 2.5, "down"),
    ]
    model = build_rooms(scene(planes, [(2, 1.5), (8, 1.5)]))
    print("2. Ceiling observed only over first of two disconnected rooms")
    print("   model status", model.status)
    for room in model.rooms:
        print("   room", room.id, Polygon(room.boundary.outer).centroid.coords[:])
    for surface in model.surfaces:
        if surface.kind == "ceiling":
            print(
                "   ceiling", surface.room_id, surface.observation_ids, surface.status
            )
    model = build_rooms(helpers["rectangle_scene"](ceiling=False))
    print("   absent ceiling model/room statuses", model.status, model.rooms[0].status)

    evidence = helpers["rectangle_scene"]()
    transform = np.array([[1, 0, 0], [0, 1, 0], [0.05, 0, 1]])
    sloped = []
    for plane in evidence.planes:
        points = plane.points @ transform.T
        normal = np.linalg.inv(transform).T @ plane.normal
        normal /= np.linalg.norm(normal)
        sloped.append(helpers["_plane"](plane.plane_id, points, normal))
    cameras = evidence.cameras @ transform.T
    model = build_rooms(SceneEvidence(sloped, cameras, "synthetic sloped floor"))
    surface = next(s for s in model.surfaces if s.kind == "floor")
    error = max(abs(surface.origin[2] - 0.05 * x) for x, _ in surface.boundary_uv.outer)
    print("3. Analytic floor z=0.05*x (about 2.86 degrees)")
    print(
        "   status",
        model.status,
        "floor normal",
        surface.normal,
        "max_z_error_m",
        error,
    )

    path = Path("runs/p07-evidence/run-1a8384c3f6/reconstruction")
    if path.is_dir():
        import json

        bundle = json.loads((path / "bundle.json").read_text())
        loaded = load_bundle(path)
        print(
            "4. Tracking segments from supplied development bundle", bundle["segments"]
        )
        print("   fields retained by SceneEvidence", list(loaded.__dict__))


if __name__ == "__main__":
    main()
