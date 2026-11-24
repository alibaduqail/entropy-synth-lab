# The two entropy metrics: what the sources define, and alternatives

Research track, night of 2026-11-23. Numbers in the BIE noise table come from
`results/research/derived_numbers.json` (script `synthlab/research/derived_numbers.py`, seed
20261123). The earlier project note `docs/research_2026-11-10/entropy_performance_review.md`
already reviews Ayunts et al. and the BIE code; this file adds the dos Santos reading, the noise
explanation with numbers, and alternatives.

## HL: dos Santos et al. 2019 does not define "eigenentropy"

R. C. dos Santos, G. G. Pessoa, A. C. Carrilho, M. Galo, "Building Detection from LiDAR Data
Using Entropy and the K-means Concept", ISPRS Archives XLII-2/W13, pp. 969-974, 2019.
https://isprs-archives.copernicus.org/articles/XLII-2-W13/969/2019/isprs-archives-XLII-2-W13-969-2019.pdf
(PDF text extracted and read tonight.)

What it defines (their Eqs. 5-11):

```
E_i = -L ln(L) - P ln(P) - S ln(S)
L = (l1 - l2) / l1,   P = (l2 - l3) / l1,   S = l3 / l1,   l1 >= l2 >= l3
E_c = mean of E_i over the points of a cluster
```

`l1..l3` are eigenvalues of the 3x3 covariance of a point and its N nearest neighbours; N is
derived from average point density and the smallest area to represent. `L + P + S = 1`, so
`E_i` ranges from 0 (purely linear, planar or scattered) to `ln 3 = 1.0986`.

Three things the team should know:

1. This is the **entropy of the dimensionality features** (linearity, planarity, scattering). It
   is not the quantity usually called eigenentropy, `-sum e_i ln e_i` with `e_i = l_i / sum(l)`
   (Weinmann et al.). The two differ: a perfect plane has dimensionality entropy 0 but
   eigenentropy `ln 2 = 0.69`. Check which one `compute_hl.py` implements and name it correctly.
2. The paper is an airborne building-detection method. Entropy separates roofs (low) from
   vegetation (high), on data at 4 and 12.5 points/m^2, judged visually. It makes no claim about
   scene disorder, fog, or compute cost.
3. It credits the formula to Shannon (1948) and Weinmann et al. (2015). The better primary
   citations are Demantke et al., "Dimensionality based scale selection in 3D lidar point
   clouds", ISPRS Archives XXXVIII-5/W12, 2011
   (https://isprs-archives.copernicus.org/articles/XXXVIII-5-W12/97/2011/isprsarchives-XXXVIII-5-W12-97-2011.pdf,
   not read tonight) and Weinmann, Jutzi, Mallet, "Semantic 3D scene interpretation: a framework
   combining optimal neighborhood size selection with relevant features", ISPRS Annals II-3,
   2014 (https://isprs-annals.copernicus.org/articles/II-3/181/2014/, search-snippet level).

Known weaknesses of eigen-feature entropy as a "disorder" measure:

- It depends on neighbourhood size. Weinmann's own use of it is to *choose* the neighbourhood
  by minimising entropy, which shows how strongly it varies with k or radius.
- It depends on local point density and sensor noise. A flat wall measured with 2 cm range noise
  at a 3 cm neighbourhood radius looks volumetric. With the Mid-360 (2-3 cm 1-sigma range
  precision) and a non-repetitive pattern whose density varies across the frame, HL will change
  with range and integration time even in a static room.
- A mean over points weights dense regions (near the sensor) more than far ones.
- It is bounded and saturates; it cannot tell "some clutter" from "a lot".
- Its cost is a k-nearest-neighbour search plus an eigen-decomposition per point, which is the
  ~350 ms per window Sam measured. The metric is itself an input-dependent workload.

## HT: block image entropy (Ayunts, Grigoryan, Agaian 2024)

"Novel Entropy for Enhanced Thermal Imaging and Uncertainty Quantification", Entropy 26(5):374,
2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11120493/ (page fetched tonight; the equation
came through garbled, and the MDPI PDF returned HTTP 403, so the exact bracketing of Eq. 5 is
**not** confirmed from the publisher). The paper states that the numerators capture global
contrast (ADP and global SD) and the denominators capture local variation (block entropy and
block SD). It was validated on four thermal datasets against Shannon entropy, AME, EME and SD
across 11 enhancement methods, with no human-rated scores. It is a no-reference *quality* score
for enhancement, where higher means better contrast.

The project's implementation (`scripts/experimentA.py`, read-only) computes

```
BIE = ADP * [mean_k(-M_k ln M_k) / (1 + mean_k E(I_k))] * [SD(I) / (1 + mean_k SD(I_k))]
ADP = 1 - |mean(I) - 127.5| / 127.5,   M_k = (max_k - min_k) / 255,   block size 15
```

on an 8-bit image mapped from a fixed Celsius range.

### Is "HT falls when noise is added while Shannon entropy rises" expected? Yes.

Per-pixel independent noise raises the Shannon entropy of every block and the standard deviation
of every block, and both sit in the denominator. The numerator terms barely move: global SD
grows slowly because the noise variance adds to an already large scene variance, and ADP is
unchanged. The modulation term `-M ln M` rises at first (it peaks at `M = 1/e`) but not enough.

Check on a synthetic 120x160 frame (22 C background, one 32 C blob, mapped 15-40 C to 8 bits):

| Noise sigma (C) | BIE | Global Shannon (bits) | Mean block entropy | Mean block SD | Global SD |
|---|---|---|---|---|---|
| 0 | 0.0819 | 2.45 | 1.22 | 2.29 | 15.26 |
| 0.035 (Lepton typical NETD) | 0.0801 | 2.82 | 1.86 | 2.60 | 15.18 |
| 0.1 | 0.0777 | 3.51 | 2.68 | 3.03 | 15.23 |
| 0.3 | 0.0678 | 4.59 | 3.91 | 4.75 | 15.48 |
| 1.0 | 0.0486 | 5.84 | 5.26 | 11.15 | 18.38 |
| 3.0 | 0.0253 | 7.08 | 6.41 | 30.90 | 34.15 |

BIE falls by 69 % while global Shannon entropy almost triples. This is one synthetic scene, not
a general proof, but it follows directly from the formula. The audit track reports r = -0.98
between BIE and noise level in the 279-trial sweep, which agrees.

Consequences for the hypotheses:

- BIE is a contrast-over-local-variation score. "HT rises with scene complexity" (Hyp A) holds
  only when complexity adds large warm regions (global contrast), not fine clutter or noise.
- Fog lowers global contrast and adds little local variation, so BIE should fall slightly in
  fog. "HT stays flat" in Hyp B is roughly right, for a reason unrelated to disorder.
- BIE depends on the Celsius-to-8-bit mapping through ADP: the same scene scores differently if
  the mapping range shifts the mean away from mid-grey.

## Better-established alternatives

Image side (all cheap on 160x120):

| Metric | Definition | Standing | Behaviour under noise |
|---|---|---|---|
| Spatial information (SI) | SD over the frame of the Sobel-filtered luminance | ITU-T P.910; the standard scene-complexity descriptor in video quality work (from memory, not re-fetched) | Rises |
| Compressed size | Bytes of the PNG or JPEG of the frame | Common complexity proxy; approximates Kolmogorov complexity | Rises |
| Shannon entropy of the 16-bit histogram | `-sum p log p` | Textbook; ignores spatial arrangement | Rises |
| Gradient / edge density | Mean gradient magnitude or Canny edge fraction | Common clutter measure | Rises |
| Warm-blob count and area | Connected components above ambient + delta | Not a standard metric, but it is the variable that drives detections | Flat until noise is extreme |
| Temporal noise | Per-pixel SD over a static window | Standard sensor characterisation (NETD) | Rises; separates noise from clutter |

Point-cloud side:

| Metric | Definition | Note |
|---|---|---|
| Eigenentropy (Weinmann) | `-sum e_i ln e_i`, `e_i = l_i / sum l` | The quantity the name refers to; pair with a fixed radius |
| Dimensionality entropy (Demantke / dos Santos) | As above with L, P, S | What the cited paper defines |
| Occupancy entropy | Shannon entropy of the voxel-occupancy histogram at a fixed voxel size | Density-normalised, no kNN, cheap; from memory, no single canonical citation checked |
| Planar-residual fraction | Share of points farther than 3 sigma from the fitted room planes | Direct "how much is not the empty room"; needs the room model |
| Points per frame, range histogram | Counts | The variables that actually drive compute |
| Tag-flag fraction | Share of points with Livox particle bits set | Sensor's own fog indicator |

Recommendation: keep HL and HT for continuity with the team's plan, but report next to them one
count-type variable per modality (detections or warm blobs; points per frame) and one
noise-sensitive variable (SI or compressed size; occupancy entropy). The count variables are the
ones the literature ties to cost (see `sponge_and_energy_attacks.md`).

## What to do with this tonight

- Every simulator track: log SI, PNG size, 16-bit Shannon entropy and warm-blob count next to
  BIE for each thermal bank, and points per frame plus near-field fraction next to HL.
- In the slides, call HT "block image entropy (a contrast quality score)", and state plainly
  that it falls under noise by construction.
- Check which formula Sam's `compute_hl.py` uses and write the name that matches.

## What to do with this next month

- Obtain the publisher PDF of Ayunts et al. and settle Eq. 5 (bracketing and the sign of the
  `ln M` term, flagged in the earlier project note).
- Fix the HL neighbourhood by radius, not k, and report HL of the empty room at each range so
  the sensor-noise floor is known.
- Test whether any candidate metric predicts cost after controlling for detection count and
  point count. If none does, entropy is a descriptor of the scene, not a driver of cost.
