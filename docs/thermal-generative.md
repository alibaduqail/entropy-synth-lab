# thermal-generative: a learned generative model of Lepton 3.5 frames, trained on the Jetson

Status: IN PROGRESS (training running on the Orin Nano; numbers are filled in as result files land
in `results/thermal-generative/`). Everything below marked "measured" comes from a file there.

## What this track builds
- `synthlab/thermal_generative/mapping.py` - fixed, invertible TLinear-counts <-> model-space map
  (affine over the pipeline's 10-50 C range, log-compressed above 50 C for the heat lamp), and exact
  ports of the real pipeline's 8-bit image and block image entropy (HT). The HT port reproduces the
  HT values logged in the real `trial_data.csv` to 4e-9 (300 random frames, measured on the Jetson).
- `data.py` - index of the real radiometric recordings, trial-level labels and whole-trial hold-out.
- `models.py`, `diffusion.py` - patch-2 conditional U-Net, cosine-schedule v-prediction DDPM, DDIM
  sampler with classifier-free guidance on person / lamp / session; conditional VAE baseline.
- `train.py`, `run_chunks.sh` - time-boxed resumable training under the shared Jetson lock, with
  tegrastats power logging.
- `sample.py`, `banks.py` - banks in the hand-off format; `measure.py` - runs them through the shared harness.
- `evaluate.py`, `metrics.py`, `plots.py` - held-out evaluation.

## Data (measured on the Jetson pack, every 3rd frame of each recording)
| session | person | lamp | train trials / frames | held-out trials / frames | what is in view |
|---|---|---|---|---|---|
| heat_lamp (8 Oct) | 0 | 0 | 20 / 3447 | 5 / 845 | empty room |
| heat_lamp (8 Oct) | 1 | 1 | 20 / 3451 | 5 / 867 | one standing person holding the heat lamp |
| warmup120 | 0 | 0 | 4 / 1376 | 1 / 344 | empty, darker (cooler) scene |
| warmup120 | 1 | 0 | 4 / 1377 | 1 / 344 | TWO seated people, close to the camera |
| radiometric | 0 | 0 | 2 / 346 | 1 / 169 | empty room, different view |
| radiometric | 1 | 0 | 2 / 337 | 1 / 173 | one standing person |

13,076 frames, 71 trials, 3 camera set-ups. Hold-out is by whole trial (every 5th run of the 25-run
groups, the last run of the small groups). Labels are trial-level (A0/A1, lamp folder), not per-frame.

Design facts that limit any model trained on this:
- Lamp and person are perfectly confounded: the lamp is hand-held by the person and "lamp on, no
  person" was never recorded. "Person, lamp off" exists only in the two other camera set-ups.
- Only three scenes and two or three people. There is no fog, no crowd, no clutter sweep.
- "person=1" means one person in two sessions and two people in the third.
