# FloScan capture card (stock apps, one page)

Follow these steps exactly, in order. They work on any iPhone 15 or newer; the LiDAR part needs a Pro model.
Capture one tier per run: photos, video or LiDAR.

## 0. Before any capture (2 minutes)

1. Settings > Camera > Formats: choose **Most Compatible** (JPEG photos, H.264 video).
2. Settings > Camera > Record Video: choose **1080p at 30 fps**, and turn **Lock Camera** on.
3. Turn on every light in the space, open interior doors fully and leave them open.
4. Move nothing during the capture, and keep people out of view.

## 1. Photos tier (native Camera app, 2 to 8 photos per room)

1. Open Camera in **Photo** mode, tap the **1x** lens, and turn **Live Photos off** (top icon).
2. Treat each room and each hallway or connector as its own "room".
3. In each room, take **6 to 8 photos** (never fewer than 2, never more than 8):
   - Stand in a different spot for every photo, at least one step (about 1 m) from the previous spot. Do not stand still and turn.
   - Hold the phone level, in landscape, at chest height. Do not zoom.
   - Every photo shows a corner where floor meets wall, and the line where wall meets ceiling.
   - For every doorway, take one photo looking through it into the next room, so both door frames are visible.
4. Before moving to the next room, in the Photos app select that room's photos and add them to a new album named `room_001`, then `room_002`, and so on.

## 2. Video tier (native Camera app, one continuous clip)

1. Open Camera in **Video** mode, **1x** lens, landscape. Start recording in the first room.
2. Walk slowly (half your normal pace) along the walls of each room, and through every doorway into every room and hallway.
3. At each wall, tilt the phone down to show the floor edge, then up to show the ceiling edge.
4. Stop for 2 seconds in every doorway. If you can, finish where you started.
5. Do not stop recording, zoom, switch lenses or turn the phone to portrait. Aim for about 1 minute per room, 6 minutes at most.

## 3. LiDAR tier (Pro iPhone only, Stray Scanner app)

1. Install **Stray Scanner** (Stray Robots) from the App Store and allow camera access.
2. Open it and tap **record**. Walk the same route as for video: slowly, along every wall, through every doorway, finishing where you started.
3. Keep surfaces between 0.5 m and 4 m away. Sweep the floor edge and the ceiling edge of every wall. Hold still for 2 seconds facing each doorway so both side frames are in view.
4. Tap **record** again to stop. One recording covers the whole space.

## 4. What to avoid (all tiers)

- Fast turns, walking backwards into doorways, or pointing at the floor for long stretches.
- Standing close to mirrors, glass doors and shiny or wet floors. Capture them normally, but do not linger on them.
- Dark rooms: turn the lights on rather than relying on Night mode.
- Fingers over the lens, and pausing or restarting a video or LiDAR recording.

## 5. Hand the files over

1. Connect the iPhone to the Mac running FloScan.
2. Photos: in the Photos app, AirDrop each album to the Mac, and put each album's files in its own folder: `capture/photos/room_001/`, `capture/photos/room_002/` ...
3. Video: AirDrop the clip and put it in `capture/video/`.
4. LiDAR: in Finder, select the iPhone > **Files** > **Stray Scanner**, and drag the recording folder into `capture/lidar/`.
5. Run one command for the tier you captured, for example:
   `./run.sh --input capture/photos --tier photo --output runs/walkin-photo --mode live`
   (use `--tier video` with `capture/video`, or `--tier lidar` with the recording folder).

Room folder names are labels only; FloScan finds how rooms connect from the images, never from the names.
