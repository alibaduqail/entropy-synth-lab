# Getting a metric 3D room from a few phone photos or a quick scan (short note)

Research track, night of 2026-11-23. This topic got the least time: two abstracts were read and
the rest is search-summary level or general knowledge, marked as such. No method was run.

## Options, best first for this project

1. **Use the LiDAR the team already has.** A few seconds of static Mid-360 data, accumulated
   (the non-repetitive pattern fills in over time), gives a metric point cloud at 2-3 cm
   precision (Livox spec, https://www.livoxtech.com/mid-360/specs). Fit planes with RANSAC for
   walls, floor and ceiling. It is already in the sensor frame the simulators need, and it has
   true scale. This beats every photo-based method on accuracy and effort.
2. **Phone LiDAR scan.** iPhone/iPad Pro apps built on Apple RoomPlan output a parametric room
   (walls, doors, windows, furniture boxes) with metric scale in about a minute (general
   knowledge, not checked tonight; accuracy is typically a few centimetres).
3. **Feed-forward multi-view models from plain photos.**
   - MapAnything: Keetha et al., "MapAnything: Universal Feed-Forward Metric 3D
     Reconstruction", 3DV 2026, https://arxiv.org/abs/2509.13414 (abstract read). Takes one or
     more images plus optional intrinsics, poses or depth, and regresses **metric** geometry and
     cameras. The authors note weaker metric scale on ScanNet (snippet), which is the indoor
     case.
   - Depth Anything 3: Lin et al., arXiv 2511.10647 (Nov 2025),
     https://arxiv.org/abs/2511.10647 (abstract read). Any number of views, with or without
     poses; reports +44.3 % pose accuracy and +25.1 % geometric accuracy over VGGT. The abstract
     does not say the multi-view output is metric.
   - VGGT, DUSt3R and MASt3R are the earlier baselines; their output is up to scale unless a
     metric head or a known length is used (general knowledge).
   No head-to-head test on few-photo indoor rooms with ground-truth dimensions was found.
4. **Classical SfM (COLMAP) or Gaussian splatting** from 30+ photos: accurate shape, no scale;
   plain painted walls are a known failure case for feature matching (general knowledge).

## The rule that matters

Whatever the method, fix scale and check it with two tape-measured lengths (for example room
width and door height). A learned metric prior can be off by several percent, and 5 % of a 4 m
room is 20 cm, which changes LiDAR range, fog path length and thermal pixel footprint.

## What to do with this tonight

- If the room-3d track is working from photos, run one feed-forward model (MapAnything if it
  installs, since it is the one that claims metric output), then rescale to any known dimension
  and record the residual on a second known dimension in the scene file.
- Keep `scene/room_v*.json` as planes and boxes; the simulators need nothing denser.

## What to do with this next month

- Record 10 s of static Mid-360 data in the empty room and fit planes; make that the reference
  geometry and express the thermal camera pose in it.
- Tape-measure four lengths and store them with the scene file as ground truth.
