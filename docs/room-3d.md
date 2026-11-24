## Metric estimation (Astra pass 1)

**Room interior estimate: 2.56 ± 0.26 m east–west, 2.66 ± 0.32 m south–north, 2.56 ± 0.23 m floor to ceiling.** The uncertainties are judgmental one-sigma estimates. Depth is the weakest dimension because the south boundary is only partly visible and the west wall curves into the glass jamb in photo 4. “North” denotes the TV wall by convention. Absolute compass bearing is unknown.

### Landmarks and scale

I picked corresponding image landmarks by hand in the supplied upright frames and fitted a rectangular room, desk corners, open leaf, wall fixtures, rig and five camera poses by robust least squares. The model uses a centered pinhole with one effective focal length for photos 1–4 and another for photo 5, with camera roll fitted. The two rig crops add identification detail only. Photos 1 and 2 have almost no stereo baseline. The export rounds the fit and separates solids for ray tracing.

The table reports sensitivity to the choice of physical ruler. In a comparison fit, the desk top set the scale at 0.74 m. I removed the door, keyboard, switch, TV, strobe, thermostat and desk-plan size priors, kept the focal-length and hinge-setback priors, and rescaled the resulting geometry by each nominal reference divided by its reconstructed length. Each row thus uses a different physical standard but shares the same image picks and shape errors. The uncertainty column gives an approximate **scale-induced** one-sigma error per room dimension, before wall and camera model errors. Triples are width × depth × height in metres.

| Physical reference | What was picked or checked | Implied room or local dimension | Approx. uncertainty |
|---|---|---|---|
| Desk top, 0.74 m high | Top quadrilateral in 1 and 3, reverse edge in 4, floor corners in 3/4. In 3 the near edge runs from (256,842) to (528,862) px. | 2.74 × 2.88 × 2.70. Desk plan 0.59 × 1.48. | ±0.15 m per room dimension for ±0.04 m desk height |
| Door width, 0.914 m | Leaf top in 1 from approximately (720,85) to (1032,82), lower hinge-side corner near (834,775). Comparison fit recovered 0.985 m. | 2.54 × 2.67 × 2.50 | ±0.15 m for ±0.05 m effective width or edge choice |
| Door height, 2.134 m | Same leaf top and lower corners in 1/2. Desk-scaled reconstruction gave 2.545 m. | 2.30 × 2.41 × 2.26. A 2.03 m leaf gives 2.19 × 2.29 × 2.15. | ±0.11 m for ±0.10 m height, plus a larger pose systematic |
| Full keyboard, 0.44 m | Primary keyboard long edges in 1/2, roughly 185 px in 1. Reconstructed end-to-end segment 0.423 m. | 2.86 × 2.99 × 2.81 | ±0.17 m for ±0.025 m in case width/endpoints |
| 16:9 TV, nominal 43-inch diagonal | In 3 the top bezel spans (430,360)–(687,358), bottom right (669,518). Estimated width 0.969 m under desk scaling. | A 0.952 m active width gives 2.69 × 2.83 × 2.65. Choosing 50 or 55 inches gives room widths near 3.13 or 3.45 m. | ±0.15 m conditional on 43 inches. Model identity dominates. |
| Switch center, nominal 1.20 m | East-wall plate center (578,519) in 1, corroborated in 2. Desk-scaled height 1.215 m. | 2.71 × 2.84 × 2.66 | ±0.23 m for ±0.10 m mounting height |
| Thermostat and strobe | On west wall in 4, centers near (179,515) and (534,226). Desk-scaled heights 1.49 and 2.38 m. | Joint fit gives 1.40 and 2.22 m, broadly compatible with nominal 1.5 and 2.0–2.3 m mounting heights. | ±0.17 m center, plus ambiguity in fixture height convention |
| Livox body, 65 × 65 × 60 mm | Crop 6 shows finned base about 60 px across and black enclosure about 185 px. Crop 7 confirms the stack order. | Roughly 0.20 m black enclosure width, modeled 0.205 m. The two faces lie at different ranges. | ±0.04 m enclosure width |
| Jetson carrier, nominal 100 × 79 mm | Fan board by east edge in 3/7, close-up in 5. | 0.100 × 0.079 m footprint prior. Board identity and enclosure thickness uncertain. | ±0.02 m footprint, ±0.10 m absolute center |

The independent keyboard and door-width standards differ by 0.32 m in width, 0.32 m in depth and 0.31 m in height. The combined fit used a 0.74 m desk plane, soft priors on the scale references and downweighted ambiguous desk and wall corners. It recovered a 2.37 m open leaf despite a 2.134 m height prior. I retained the taller leaf to match the photographed outline. A door measurement would resolve much of this disagreement. The lever appears around (686,583) in 1, but the lower free corner is partly hidden. I did not treat the handle and switch as interchangeable heights.

Carpet texture and a few seams are visible, but there is no countable run with known pitch between opposite walls. The ceiling shows grid lines without a confirmed tile size or a complete wall-to-wall count. The rubber baseboard helps identify the floor junction in 3, but its nominal 0.10 m is too small and its lower edge too ambiguous for an independent room scale. Single-view dimensions can easily miss by 5–10%, and the depth and door estimates can miss by more.

### Geometry and viewpoints

Compared with v0, width changed from 2.70 to 2.56 m, depth from 3.50 to 2.66 m and height from 2.70 to 2.56 m. The desk long axis now runs north–south. Its top is 1.40 × 0.56 m with a 0.09 m north-wall gap, an east modesty panel and chairs along the west side. Photo 3 places the floor monitor east of the desk, about 0.80 m south of the teal wall. The rig sits on the southern half. The visible camera-like device faces roughly west. The v0 south-facing sensor assumption has been replaced with a southwest sightline for the two unidentified sensors.

The open leaf lies near the east wall. Photos 1 and 4 support black-framed glazing along the south boundary. I infer that the closed door spans the **south** opening at the east end, hinged on its east jamb. The east wall carries the switch. Door and glazing cutouts are separate from the room's opaque surface fallback. The estimated pane width, west return, opaque end strips, header, hinge setback and 90° opening angle have low confidence. The physical door swing and amount of south drywall cannot be uniquely recovered. A renderer that treats the entire south wall as opaque will suppress transmitted LiDAR rays. A renderer that treats every object as solid will fill the door opening.

| Photo | Camera xyz (m) | Yaw / pitch (degrees) | Effective horizontal × vertical FOV | Position σ |
|---|---|---|---|---|
| 1 | 0.13, 1.62, 1.60 | −1.1 / −27.3 | 82.5° × 98.9° | 0.18 m |
| 2 | 0.12, 1.63, 1.61 | −0.2 / −26.6 | 82.5° × 98.9° | 0.18 m |
| 3 | 1.76, 0.22, 1.55 | 94.9 / −25.3 | 82.5° × 98.9° | 0.18 m |
| 4 | 2.37, 1.68, 1.45 | 178.1 / −20.3 | 82.5° × 98.9° | 0.18 m |
| 5 | 1.99, −0.26, 1.38 | 100.7 / −43.7 | 38.7° × 64.0° | 0.32 m |

Yaw increases from east toward north. Pitch is positive upward. The JSON up vectors include roll. Photo 5 fits from the corridor looking through the opening. It has fewer shared landmarks, including partly extrapolated desk corners, and its pose is the least certain. The fitted ultrawide FOV is narrower than the approximate 88° × 108° input prior. Allow several degrees of camera-angle error and roughly 5–10° of FOV error. These are registration estimates for preview rendering.

### Sensor placement and validation

The Mid-360 body is visible. Its scan origin is a free-space proxy 6 mm above the body box. Its yaw is assumed. Neither the Lepton nor the radar can be identified in these photos. Their synthetic origins sit on the west/front side of the stack and point southwest toward torso height at the person mark (0.27, 0.23, 0). The horizontal Lepton-to-mark distance is 1.60 m. Along its azimuth the ray exits the desk footprint after about 0.32 m, then crosses about 1.58 m of unobstructed floor before reaching the south boundary. The heat lamp and mist source are hypothetical placements. The scene represents the desk arrangement in photos 1–4. Photo 5 shows a keyboard moved to the floor.

Temperatures, emissivities, reflectivities and transmission fractions are simulation inputs. The glass is LWIR-opaque with a nominal reflected fraction of 0.10. The 905 nm transmission estimate is particularly sensitive to incidence angle and coatings. Chair envelopes, desk supports, webcam proxy, board identities and unmeasured sensor offsets are approximations. Each object carries an absolute center uncertainty. Shared support surfaces constrain relative vertical ordering much better than absolute room coordinates.

The inline geometry check verified 33 solids inside the room, at least 5 mm separation along a separating axis for every pair, free sensor origins and clear segments from the Lepton and radar to both the feet and torso at the mark. The 34th object is the non-solid door opening. The scene supplies geometry for matched A/B/C and null simulations. These photos provide no entropy, fog, latency or power measurements.
