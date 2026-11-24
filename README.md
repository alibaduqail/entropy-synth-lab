# entropy-synth-lab

Synthetic sensor data (thermal, LiDAR, radar) for the entropy-vs-compute-cost study, generated and
measured on a Jetson Orin Nano Super. Each approach lives on its own branch/worktree; `main` holds
the shared scene description, Jetson helper, and the merged results.

- `AGENTS.md` - project framing and working rules
- `scene/` - 3D description of the test room
- `synthlab/<track>/` - generators and harnesses
- `results/<track>/`, `docs/<track>.md` - measurements and findings
- `tools/jetson.sh` - run commands on the Jetson with an exclusive measurement lock
