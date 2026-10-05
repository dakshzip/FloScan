# Up-sign visual check (P07)

The world up sign comes from the source's documented ARKit gravity alignment and is not verified automatically (a P06 convention gap).
Under that sign, the cameras of both sample sessions look down in 95 to 97 percent of frames and never look up, and no down-facing (ceiling-like) plane is found.

`down_1420.png` and `down_1236.png` are the c00a170fe1 frames whose optical axis points most steeply down under the documented sign (forward z = -0.75 and -0.52 in W).
Each was rotated so that the documented world-down direction points down in the image.
Both appear upright: floor tiles at a door threshold, and a toilet standing on a tiled floor.
With the opposite sign they would be ceiling views, or would appear upside down after this rotation.

This is a visual check of two frames by the developer, not an automated verification; the convention gap stays open.
It also means the observed parts of c00a170fe1 are floor-facing views, not ceiling views as earlier described.
