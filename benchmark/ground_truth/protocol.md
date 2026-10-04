# Benchmark collection work order and measurement sheet

This is the collection plan for the FloScan benchmark: what to capture, what to measure with the tape, and how to hand everything over.
The capture steps themselves are in `docs/capture_protocol.md`; follow that card for every capture below.
Collection status is tracked in `benchmark/manifests/*.json`; run `uv run python -m benchmark.cases.manifest status` at any time to see what is still missing.

Equipment confirmed by the operator on 2026-10-04: iPhone 16 Pro or Pro Max, Stray Scanner for LiDAR, the native Camera app for photos and video, magicplan as the incumbent app, and a tape measure for ground truth.

## Rules

- Never write a value you did not read off the tape. Leave a cell empty if you could not measure it, and say why in the notes.
- Never measure from the FloScan output, the incumbent app or a floor plan; ground truth comes only from the tape.
- Do not place markers, rulers or reference objects in view for capture; the captures must look like any ordinary room.
- Every capture is a new recording. Copying, re-exporting, renaming or trimming a recording is not a new capture; registration compares the content of every photo and video file, whatever its name, and refuses copies.
- Keep the original files exactly as the phone produced them.

## What to collect (minimum, from the assignment)

| # | Item | Case | Status column in the manifest |
|---|---|---|---|
| 1 | One property with at least 3 rooms plus a connector (hallway or corridor) | `dev_property` | `rooms` |
| 2 | One furnished room in it with staged damage of two different classes (a stain and a crack) | `dev_property` | `damage` |
| 3 | Photos, video and LiDAR of the whole property (one capture of each) | `dev_property` | `captures` |
| 4 | One room captured twice at the same tier, as two separate recordings (LiDAR at least; photo and video too if time allows) | `dev_property` | `captures` (purpose `repeat`) |
| 5 | Tape measurements of everything below | `dev_property` | `ground_truth` |
| 6 | magicplan scans of two of the rooms, with the app's own export | `dev_property` | `incumbent` |
| 7 | If time allows: a second property for calibration, a third for held-out evaluation, and a never-seen space for a walk-in rehearsal | `calibration_set`, `heldout_property`, `walkin_property` | all |

Also capture any mirror, glass door, glossy or wet-looking floor and dim corner you have, and note them on the sheet; failures on them are reported, not hidden.

## Order of work (about 2 to 3 hours for one property)

1. **Prepare (15 min).** Open doors, turn on lights, stage the damage (below), and print this sheet.
2. **Sketch and label (15 min).** Draw each room from above. Number the rooms `R01`, `R02` ... in walking order, with the hallway as its own room. In each room number the walls clockwise starting from the wall with the entrance door: `R01-W1`, `R01-W2` ... Label doors `R01-D1` ..., windows `R01-N1` ..., damage `R02-X1` ...
3. **Measure (60 to 90 min)** with the sheet below.
4. **Capture (30 to 45 min)** following the card: photos, then video, then LiDAR of the whole property, then the second LiDAR recording of one room (start it from a different doorway and walk the room in the other direction).
   Every whole-property capture must include every room and the connector on the sketch; a room left out is recorded as missing for that tier.
   Write each photo album's name next to its room on the sketch (for example `room_001 = R01`); this mapping, not the album name, says which room the photos show.
   Note which sketch rooms the video and each LiDAR recording passed through, and the date of each recording.
5. **Incumbent (15 min).** In magicplan, scan the two rooms you chose for the head-to-head (they must be among the rooms you measured).
6. **Hand over (10 min).** Copy the files as described at the end.

## Staging the damage (removable, no harm to the property)

Use one furnished room (furniture stays where it is).
- **Stain:** a sheet of paper with a brown tea or coffee stain at least 10 cm across, dried, then taped flat on a wall at a height between 0.5 m and 1.8 m.
- **Crack:** a strip of light masking tape at least 30 cm long on a wall or the ceiling, with a thin, irregular dark line drawn along it with a fine marker.
- Put the two on different walls. Measure them (sheet D) before capturing, and photograph each one close up with a tape laid alongside for the record (these close-ups are evidence, not capture input).
- List the walls of that room that have no damage (sheet D, last row); these are the negative surfaces.

## Measurement sheet

Record every reading in metres to the millimetre (for example `3.412`), twice.
If the two readings differ by more than 5 mm, measure a third time and record all three.
Do not write an uncertainty you cannot justify.
If a reading was hard to place (an obstructed corner, a jamb you could not reach), say so in the notes; the uncertainty is worked out later from the stated tolerance and these notes, or recorded as unknown.

### Sheet A: instrument and operator (fill once)

| Field | Value |
|---|---|
| Tape brand and model | |
| Tape length and accuracy class (printed on the tape, for example "EC class II") | |
| Smallest marking (resolution) | |
| Stated accuracy or tolerance (from the tape or its manual; write "unknown" if not stated) | |
| Operator name | |
| Date and start time | |
| Property address or nickname (for `property_id`) | |

### Sheet B: rooms

Ceiling height is measured from the finished floor to the finished ceiling at the **centre of the room**; mark the spot on your sketch.
Diagonals run between opposite inside corners at 1.0 m above the floor.

| Room ID | Name (living, kitchen ...) | Ceiling height reading 1 | Reading 2 | Diagonal 1 (corners) | Reading 2 | Diagonal 2 (corners) | Reading 2 | Notes |
|---|---|---|---|---|---|---|---|---|
| R01 | | | | | | | | |
| R02 | | | | | | | | |
| R03 | | | | | | | | |
| R04 | | | | | | | | |

### Sheet C: walls

Wall length is the distance along the wall between its two inside corners, finished surface to finished surface (paint or plaster, not skirting boards), measured **1.0 m above the floor**.
If furniture blocks that height, measure at another height and write the height in the notes.

| Wall ID | From corner to corner | Reading 1 | Reading 2 | Measured at height | Notes (mirror, glass, obstructions) |
|---|---|---|---|---|---|
| R01-W1 | | | | 1.0 | |
| R01-W2 | | | | 1.0 | |
| ... | | | | | |

### Sheet D: openings, connections and damage

Door and passage width is the **clear opening between the jambs** (the frame faces, not the door stops or trim) at **1.0 m above the floor**.
Door height is from the floor to the underside of the head jamb.
Window width is the clear opening inside the frame at mid-height; also record the sill height from the floor.
Offset is measured along the host wall from the wall's start corner (the corner where you begin walking clockwise) to the nearest jamb.
Wall thickness is the jamb depth, wall face to wall face, at the opening.

| Opening ID | Type (door, passage, window) | Host wall | Leads to room | Width reading 1 | Reading 2 | Height | Sill (windows) | Offset from wall start | Wall thickness |
|---|---|---|---|---|---|---|---|---|---|
| R01-D1 | | | | | | | | | |
| ... | | | | | | | | | |

For staged damage, position is the centre of the item: distance along the wall from its start corner, and height above the floor.

| Damage ID | Class (stain, crack) | Surface (wall ID, floor, ceiling) | Along wall | Height | Length | Width (stains only) | Close-up photo file |
|---|---|---|---|---|---|---|---|
| R0?-X1 | stain | | | | | | |
| R0?-X2 | crack | | | | | | |
| Undamaged walls in that room | | | | | | | |

### Sheet E: hard conditions

| Location (room or wall ID) | Condition (mirror, glass, glossy or wet-looking floor, low light, other) | Notes |
|---|---|---|
| | | |

## Incumbent comparison (magicplan)

1. Install magicplan from the App Store and record its version (App Store page > Version History, or the app's About screen).
2. Create a project and scan the two chosen rooms with its room-scanning mode, following the app's own guidance.
3. Do not edit or correct any dimension inside the app.
4. Export whatever the free tier allows (PDF, image or data export), and also take screenshots of each room's dimensions screen.
5. Write down the date, the phone model and the two room IDs from your sketch; both must be rooms on the sketch.
A screenshot transcription is recorded as a transcription, never as an export: it is kept as supporting evidence, but the incumbent item stays unmet until the app's own export file is handed over.

## Hand-over

Put the files on the Mac under `data/raw/<property_id>/`, which Git ignores:

```text
data/raw/<property_id>/
  sheet/            the filled sheet (photo or scan) and the sketch
  damage_closeups/  close-up photos of the staged damage
  photos_01/        room_001/, room_002/ ... from the photo capture
  video_01/         the walkthrough clip
  lidar_01/         the whole-property Stray Scanner recording folder
  lidar_02_R0?/     the second recording of one room
  magicplan/        the export files, screenshots and a note with the app version
```

Then tell the developer which folders exist; they are registered in `benchmark/manifests/development.json` with their content hashes, and the measurement sheet is transcribed into `benchmark/ground_truth/records/<property_id>/measurements.json` as described in `schema.md`.

Registration records, for each capture:

- `raw_manifest_hash`: provenance of the exact folder tree, file names included (from `uv run python -m benchmark.cases.manifest hash <folder>`).
- `media_sha256`: the content digests of its photo and video files, names ignored (same command); two captures that share any of these are one recording.
- `room_ids`: the sketch rooms, connector included, that the capture covers; for photos, `photo_groups` maps each album folder to its sketch room and photo count.
- Device, app with version and recording date, as the operator wrote them; distinct file content does not by itself prove a fresh session, so this record stays visible in the status report.
- A repeat names the primary recording it repeats (`repeat_of`), at the same tier and covering the same room.
