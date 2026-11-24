## Metric estimation (Astra pass 1)

**The adopted interior is 2.56 m east–west × 2.66 m south–north × 2.56 m high.** North is the TV wall. The pass‑1 positional uncertainties of ±0.26, ±0.32 and ±0.23 m are judgmental estimates. Depth is particularly uncertain because the south boundary is only partly visible. The shared model uses metres, with its origin at the southwest floor corner. [Fit report](../results/room-3d/fit_report.json) · [Geometry record](../results/room-3d/verification_round_1.json)

### Landmarks and scale

The recovered pass‑1 picks locate wall corners, the tabletop, leaf, keyboard, fixtures and sensor stack. Each photo has a centered pinhole camera with fitted roll. The first four share a focal length. The close-up has its own. Photos 1 and 2 have little stereo baseline. The rig crops identify components but do not constrain room scale. The far desk corners in the original close-up were partly extrapolated. The verification fit retains these original picks for reproducing pass 1 and fits the final close-up pose separately. [Pixel picks and world parametrization](../results/room-3d/landmarks.json)

The joint fit anchors the desk height and gives the other rulers soft weights. It returns **2.555 × 2.659 × 2.556 m** and a **2.374 m** leaf. To check the scale independently of the soft metric priors, a second fit retains the desk height, focal-length and hinge-setback priors, then rescales the reconstructed room using each ruler in turn. Rows share the same image observations and shape errors. Fixture mounting heights and TV width are assumptions. [Fit settings and unrounded results](../results/room-3d/fit_report.json)

| Ruler | Assumed length (m) | Desk-scaled reconstruction (m) | Implied width × depth × height (m) |
|---|---:|---:|---|
| Desk height | 0.740 | 0.740 | 2.74 × 2.88 × 2.70 |
| Door width | 0.914 | 0.985 | 2.54 × 2.67 × 2.50 |
| Short leaf height | 2.134 | 2.545 | 2.30 × 2.41 × 2.26 |
| Tall leaf height | 2.438 | 2.545 | 2.63 × 2.75 × 2.58 |
| Full keyboard width | 0.440 | 0.423 | 2.86 × 2.99 × 2.81 |
| TV active width | 0.952 | 0.969 | 2.69 × 2.83 × 2.65 |
| Switch center height | 1.200 | 1.215 | 2.71 × 2.84 × 2.66 |

The thermostat and strobe have only a single shared view and uncertain installed heights. The report includes their conditional scales. The Livox enclosure and Jetson carrier constrain local rig dimensions. Perspective, uncertain board identity and the absence of a surveyed baseline limit their use as room rulers. Carpet seams, ceiling tiles and baseboard edges also lack a confirmed pitch or height. [Landmarks](../results/room-3d/landmarks.json) · [Reference reconstructions](../results/room-3d/fit_report.json)

**The 2.438 m leaf is more consistent with the other references.** At that scale, the reconstructed desk, keyboard, door width and switch differ from nominal by −4.2%, −8.0%, +3.3% and −3.0%. The 2.134 m leaf gives −16.2%, −19.5%, −9.6% and −15.1%. The longer leaf is plausible, but its actual height has never been measured. Its conditional room estimate differs from the adopted dimensions by only about 7, 9 and 2 cm, all within the estimated room uncertainty. The nominal scene therefore retains the joint-fit dimensions and 2.37 m modeled leaf. [Door-height comparison](../results/room-3d/fit_report.json)

### Geometry and viewpoints

The long desk axis runs north–south, with chairs on the west, a modesty panel on the east and the monitor on the floor beyond the panel. The door swings toward the east wall from an inferred south jamb. The glass width, residual wall strips, opening angle and extent of the corridor remain uncertain. The preview continues the carpet color outside the room as a visual backdrop. Exported meshes and sensor ray casts end at the modeled boundary.

| Photo | Fitted camera xyz (m) | Focal length (native px) | Landmarks | Fit RMS (native px) | Exported geometry RMS (native px) |
|---|---|---:|---:|---:|---:|
| 1 | 0.126, 1.624, 1.599 | 598.6 | 13 | 5.6 | 13.5 |
| 2 | 0.125, 1.625, 1.608 | 598.6 | 13 | 5.7 | 13.3 |
| 3 | 1.755, 0.218, 1.548 | 598.6 | 12 | 8.6 | 8.8 |
| 4 | 2.369, 1.675, 1.454 | 598.6 | 7 | 25.8 | 26.1 |
| 5 | 2.242, −0.757, 1.696 | 1343.7 | 10 | 16.9 | 16.9 |

RMS is the root mean squared Euclidean pixel distance per fitted point at full upright resolution. The exported-geometry column projects the actual box corners, faces or centers used for those landmarks. The close-up fits directly against those proxies. All values describe registration on the picked points. Camera focal lengths include any crop and remain correlated with estimated camera distance. [Poses, fields of view and per-landmark residuals](../results/room-3d/fit_report.json)

### Sensor placement and validation

The Mid-360 cap is visible and its synthetic origin lies above it. Its yaw is assumed. The Lepton and radar identities and poses remain **ASSUMED**. Their aim is southwest toward torso height at the reviewer’s mark **(0.45, 0.45, 0) m**. Geometry checks cover both feet and torso sightlines, free origins, child containment and a minimum **5 mm** separating-axis gap between top-level solids in both peripheral layouts. [Geometry record](../results/room-3d/verification_round_1.json)

The south wall now has distinct doorway and glazing apertures. A ray through open air escapes. A ray aimed at the pane hits glass for thermal or first-surface queries. The range preview shows the branch conditional on glass transmission, where an unobstructed transmitted ray escapes. Actual transmission, surface temperature, emissivity and return probabilities remain simulation inputs. Heat lamp and fog source positions are hypothetical. This verification supplies matched scene geometry for A/B/C and the null. It contains no observations of entropy, power or latency.

## Verification round 1 (Astra critique of renders)

The comparisons below use **preview pixels** at half the native image scale. Distances are rounded from the exported landmark residuals. Original and final object values are recorded alongside the figures. [Residuals](../results/room-3d/fit_report.json) · [Object changes](../results/room-3d/verification_round_1.json)

| Photo | What changed after inspection | Remaining discrepancy |
|---|---|---|
| 1 | The chair blocks hid the desk. Mesh-chair depth along x went from 0.47 to 0.264 m and its center x from 0.854 to 0.971 m. The seat, framed back and legs now leave floor visible. The green chair has a shell, post and spread base. | The door’s top hinge corner is about 14 px off. Chair backs have square edges and the black mesh uses a surface proxy. |
| 2 | The green chair center x moved from 0.855 to 0.971 m. Its seat reaches the desk edge with a 7 mm horizontal gap. The primary keyboard center xy moved from (1.186, 1.750) to (1.200, 1.712) m. | The rendered hinge corner differs by about 15 px and the keyboard’s north endpoint by 8 px. A literal tuck under the top would intersect the required one-box chair envelope. |
| 3 | The desk had no apron. Two end rails now join the legs visually. The angled keyboard went from 0° to −34° yaw and the floor monitor center xy from (1.90, 1.86) to (1.89, 1.94) m. | TV bezel picks differ by about 2–3 px and the near desk edge by 1–4 px. The northwest floor junction is about 11 px off. Cable loops and chair curvature remain unresolved. |
| 4 | The strobe and thermostat now read separately from the wall. The sage wall replaces saturated teal, the white wall is lighter, and the framed chair back replaces the full 0.91 m block. | Visible fixtures and desk corners agree within about 3 px. The partly obscured northwest floor junction misses by 34 px. The rectangular shell also simplifies the curved west return. |
| 5 | Four tabletop picks became ten desk, leg, panel, bag, monitor and hinge-foot picks for the pose refit. Camera xyz moved from (1.99, −0.26, 1.38) to (2.24, −0.76, 1.70) m. The floor keyboard and mouse use a photo-specific arrangement. On the same final proxies and ten picks, native RMS fell from 100.7 to 16.9 px. | The desk foot is about 2 px off, the bag base 13 px, the screen’s top right 14 px and the estimated hinge foot 10 px. The soft bag and screen stand remain simplified. |

The fit can be regenerated from the repo root with the following command. Both the SciPy solver and the NumPy fallback were exercised. Neither draws random samples. The exported cameras, material colors and idealized sensor views serve visual and geometric checks only. [Fit report](../results/room-3d/fit_report.json) · [Solver agreement](../results/room-3d/verification_round_1.json)

```sh
python3 -m synthlab.room3d.fit_room --landmarks results/room-3d/landmarks.json --out results/room-3d/fit_report.json
```
