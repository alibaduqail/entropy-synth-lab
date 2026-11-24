"""Tests for the Jetson harness that run anywhere (no CUDA, no tegrastats)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).resolve().parents[1] / "synthlab" / "harness"
sys.path.insert(0, str(HARNESS))
import lidar_pipeline as lp  # noqa: E402
import make_test_banks  # noqa: E402
import run_lidar_bank  # noqa: E402
import run_thermal_bank  # noqa: E402
import telemetry as tm  # noqa: E402
import thermal_pipeline as tp  # noqa: E402

LINE = ("11-23-2026 21:42:17 RAM 687/7607MB (lfb 11x4MB) SWAP 0/3804MB (cached 0MB) "
        "CPU [9%@729,30%@1113,0%@729,0%@729,0%@729,3%@729] GR3D_FREQ 41% cpu@46.5C "
        "soc2@46.093C gpu@48.718C tj@48.718C VDD_IN 5325mW/5300mW "
        "VDD_CPU_GPU_CV 1031mW/1031mW VDD_SOC 1430mW/1430mW")


def test_parse_tegrastats_line():
    s = tm.parse_line(LINE)
    assert s["vdd_in_w"] == pytest.approx(5.325)
    assert s["vdd_cpu_gpu_cv_w"] == pytest.approx(1.031)
    assert s["gpu_pct"] == 41 and s["ram_mb"] == 687
    assert s["cpu_pct"] == pytest.approx(7.0) and s["temp_gpu_c"] == pytest.approx(48.718)
    assert tm.parse_line("garbage") is None


def test_summarize_energy_and_coverage(tmp_path):
    path = tmp_path / "t.csv"
    rows = ["unix_time,tegrastats_line"]
    for i in range(60):  # 0.2 s interval, constant 5.325 W
        rows.append(f'{1000 + 0.2 * i:.4f},"{LINE}"')
    path.write_text("\n".join(rows) + "\n")
    out = tm.summarize(path, 1002.0, 1010.0, 0.2, n_units=80)
    assert out["ok"], out["problems"]
    assert out["energy_j"] == pytest.approx(5.325 * 8.0, rel=1e-6)
    assert out["energy_per_unit_j"] == pytest.approx(5.325 * 8.0 / 80)
    bad = tm.summarize(path, 1002.0, 1030.0, 0.2)
    assert not bad["ok"]


def test_thermal_conversion_and_bie():
    raw = np.full((120, 160), 29515, np.uint16)  # 22.0 C
    assert tp.raw_to_celsius(raw).mean() == pytest.approx(22.0, abs=1e-3)
    gray = tp.radiometric_bie_image(raw)
    assert gray.dtype == np.uint8 and int(gray[0, 0]) == round((22 - 10) * 255 / 40)
    assert tp.block_image_entropy(gray)["value"] == 0.0  # flat image has no modulation
    rng = np.random.default_rng(0)
    textured = tp.radiometric_bie_image(make_test_banks.thermal_bank(1, blobs=3, seed=3)[0])
    bie = tp.block_image_entropy(textured)
    assert bie["value"] > 0 and bie["block_count"] == 8 * 11
    assert tp.yolo_frame_from_thermal(textured).shape == (120, 160, 3)
    with pytest.raises(ValueError):
        tp.validate_bank(rng.random((3, 120, 160)))


def test_entropy_limits():
    # line, plane, isotropic eigenvalue triplets
    lam = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2 / 3, 1 / 3]])
    h_ds, h_eigen, L, P, S = lp.entropies(lam)
    assert h_ds[:3] == pytest.approx([0, 0, 0], abs=1e-9)
    assert h_ds[3] == pytest.approx(math.log(3))
    assert h_eigen[:3] == pytest.approx([0, math.log(2), math.log(3)], abs=1e-9)
    assert (L + P + S) == pytest.approx(np.ones(4))


def test_plane_vs_volume_entropy():
    rng = np.random.default_rng(1)
    plane = np.c_[rng.uniform(-2, 2, (40000, 2)), rng.normal(0, 0.002, 40000)]
    volume = rng.uniform(-0.6, 0.6, (40000, 3))
    p = lp.Params(z_min=-5)
    rows = [lp.process_window([(c.astype(np.float32), np.zeros(len(c), np.uint8))], p, "both")
            for c in (plane, volume)]
    assert rows[0]["H_eigen"] < rows[1]["H_eigen"]
    assert rows[0]["mean_S"] < 0.1 < rows[1]["mean_S"]
    assert rows[0]["n_voxel"] <= rows[0]["n_roi"] == 40000
    assert rows[0]["n_clusters"] >= 1 and rows[0]["n_pairs"] > 0
    assert rows[0]["total_ms"] >= rows[0]["entropy_total_ms"]


def test_voxel_downsample_centroids():
    pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.26, 0.0, 0.0]])
    out = lp.voxel_downsample(pts, 0.05)
    assert out.shape == (2, 3)
    assert sorted(out[:, 0]) == pytest.approx([0.02, 0.26])


def test_cluster_counts_two_people():
    rng = np.random.default_rng(2)
    person = lambda cx: np.c_[rng.normal(cx, 0.08, 6000), rng.normal(0, 0.08, 6000),
                              rng.uniform(-0.8, 0.9, 6000)]
    cloud = np.r_[person(-1.5), person(1.5)]
    row = lp.process_window([(cloud.astype(np.float32), np.zeros(len(cloud), np.uint8))],
                            lp.Params(), "cluster")
    assert row["n_person_candidates"] == 2
    assert math.isnan(row["H_DS"]) and row["knn_ms"] == 0.0


def test_runners_end_to_end(tmp_path):
    make_test_banks.write(tmp_path / "banks")
    out = tmp_path / "thermal_out"
    rc = run_thermal_bank.main(["--bank", str(tmp_path / "banks" / "thermal_blobs"), "--out", str(out),
                                "--seconds", "1", "--warmup", "0.2", "--idle-seconds", "0",
                                "--backend", "none", "--no-tegrastats", "--fps", "20"])
    summary = json.loads((out / "summary.json").read_text())
    assert rc == 1 and summary["problems"] == ["tegrastats not available: no power data"]
    assert summary["frames"] == 20 and summary["HT"]["mean"] > 0
    assert (out / "frames.csv").read_text().count("\n") == 21
    out = tmp_path / "thermal_nobie"
    run_thermal_bank.main(["--bank", str(tmp_path / "banks" / "thermal_blobs"), "--out", str(out),
                           "--seconds", "0.5", "--warmup", "0", "--idle-seconds", "0", "--no-bie",
                           "--backend", "none", "--no-tegrastats", "--unpaced"])
    nobie = json.loads((out / "summary.json").read_text())
    assert nobie["HT"]["offline"] and nobie["HT"]["mean"] == pytest.approx(summary["HT"]["mean"], rel=0.05)
    assert nobie["stages_ms"]["bie_ms"]["mean"] < 0.05 and not nobie["bie_in_loop"]
    out = tmp_path / "lidar_out"
    run_lidar_bank.main(["--bank", str(tmp_path / "banks" / "lidar_room_fog"), "--out", str(out),
                         "--seconds", "2", "--warmup", "0", "--idle-seconds", "0", "--unpaced",
                         "--no-tegrastats"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["bank_windows"] == 2 and summary["windows"] >= 1
    assert 0 < summary["values"]["H_eigen"]["mean"] < math.log(3)
    assert summary["values"]["frac_tag_particle"]["mean"] > 0.1


def test_cmp_helpers():
    import pandas as pd
    import analyze_cmp as ac
    assert ac.fog_of({"fog": {"alpha": 0.3}}) == 0.3
    assert ac.fog_of({"frames": [{"fog_level": 1}, {"fog_level": 3}]}) == 2.0
    assert ac.fog_of({"generator": "x"}) == 0.0
    rng = np.random.default_rng(0)
    rows = []
    for b in range(6):
        for k in range(3):
            rows.append({"bank_id": f"b{b}", "block": k, "temp_tj_c": 50 + k + rng.normal(0, .1),
                         "y": 10 + 2 * b + 0.5 * k + rng.normal(0, 0.05)})
    means, sd, _ = ac.adjust(pd.DataFrame(rows), "y")
    assert abs((means["b5"] - means["b0"]) - 10) < 0.3 and sd < 0.2
    x = np.linspace(0, 10, 40)
    knee = ac.hinge_test(x, np.where(x < 5, 1.0, 1 + 2 * (x - 5)) + rng.normal(0, .05, 40))
    assert knee["dBIC_linear_minus_hinge"] > 6 and abs(knee["knee"] - 5) < 1
    flat = ac.hinge_test(x, 2 * x + rng.normal(0, .05, 40))
    assert flat["dBIC_linear_minus_hinge"] < 6
