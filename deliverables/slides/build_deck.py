"""Build the plain overnight-run deck. Usage: python build_deck.py [out.pptx]
Every number is copied from a file under results/ (see notes on each slide)."""
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("overnight_run_2026-11-24.pptx")
EXTRA = Path(__file__).with_name("extra_slides.json")  # optional late results
BLACK = RGBColor(0, 0, 0)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, lines, size=18, bold=False):
    if size <= 20 and not bold:  # shrink bullet text until the estimated height fits the box
        while size > 12:
            cpl = max(10, int(w * 72 / (size * 0.5)))
            n_lines = sum(-(-len(l.strip()) // cpl) for l in lines)
            if (n_lines * size * 1.22 + len(lines) * 6) / 72 <= h:
                break
            size -= 1
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        sub = line.startswith("  ")
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        r = p.add_run()
        r.text = ("– " if sub else "• ") + line.strip() if size <= 20 and not bold else line
        r.font.size = Pt(size - 2 if sub else size)
        r.font.bold = bold
        r.font.name = FONT
        r.font.color.rgb = BLACK
        p.level = 1 if sub else 0
        p.space_after = Pt(6)
    return tb


def image(slide, path, x, y, w, h):
    path = Path(path)
    if not path.exists():
        text(slide, x, y, w, 0.5, [f"(figure missing: {path.name})"], 12)
        return
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, x, y, w, h, size=13, widths=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(255, 255, 255) if i else RGBColor(230, 230, 230)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.name = FONT
                    r.font.bold = i == 0
                    r.font.color.rgb = BLACK
    if widths:
        for j, wd in enumerate(widths):
            t.columns[j].width = Inches(wd)


def slide(title, bullets=None, img=None, layout="right", notes="", tbl=None, widths=None, img2=None):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 12.3, 0.9, [title], 32, bold=True)
    if layout == "right":      # bullets left, figure right
        if bullets:
            text(s, 0.5, 1.35, 5.6, 5.8, bullets, 17)
        if img:
            image(s, img, 6.3, 1.35, 6.6, 5.8 if not img2 else 2.8)
        if img2:
            image(s, img2, 6.3, 4.3, 6.6, 2.9)
    elif layout == "bottom":   # bullets top, wide figure below
        if bullets:
            text(s, 0.5, 1.25, 12.3, 2.4, bullets, 17)
        if img:
            image(s, img, 0.5, 3.75, 12.3, 3.55)
    elif layout == "text":
        if bullets:
            text(s, 0.5, 1.35, 12.3, 2.6 if tbl else 5.8, bullets, 18)
        if tbl:
            table(s, tbl, 0.5, 1.35 if not bullets else 3.9, 12.3, 0.4 * len(tbl), widths=widths)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# 1 ---------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
text(s, 0.8, 2.4, 11.7, 1.2, ["Overnight synthetic-data run"], 40, bold=True)
text(s, 0.8, 3.6, 11.7, 1.6, ["Thermal, LiDAR and radar simulated and measured on the Jetson Orin Nano",
                              "Night of 23–24 November 2026 · repo: alibaduqail/entropy-synth-lab (private)"], 20, bold=True)
s.notes_slide.notes_text_frame.text = ("Ten parallel tracks, each on its own git branch. Every number in this deck comes from a "
                                       "file under results/ in the repo. Fog results are simulation-only.")

# 2 ---------------------------------------------------------------------------
slide("What ran", layout="text", tbl=[
    ["Track", "What it produced", "Measured on Jetson?"],
    ["Jetson harness", "Shared runner, TensorRT engine, noise floor, 74 thermal + 29 LiDAR banks compared", "Yes"],
    ["Thermal: physics", "Ray-cast room + calibrated Lepton 3.5 sensor model, fog term, 17 banks", "Yes"],
    ["Thermal: data-driven", "Real recordings decomposed and recombined, 30 banks, held-out validation", "Via harness"],
    ["Thermal: generative", "Diffusion model and VAE trained on the Jetson itself", "See slide"],
    ["LiDAR fog", "Mid-360 simulator with first-return fog model, entropy + clustering cost", "Yes (1 pass)"],
    ["Radar", "IWR6843ISK FMCW simulator, fog loss, stillness metric", "Yes"],
    ["Sponge search", "Search for the costliest plausible thermal scene, 1,478 timed evaluations", "Yes"],
    ["3D room", "Room model from 5 photos (gpt-6.2-astra via Codex, medium effort)", "—"],
    ["Experiment audit", "Re-analysis of the 279-trial sweep, heat-lamp trials and LiDAR pilot", "No (old data)"],
    ["Research", "Literature notes on synthesis, fog, sponge attacks, metrics", "Sensor probe only"],
], widths=[2.4, 7.6, 2.3], notes="First wave ran on Fable and hit the usage limit after ~30 minutes; the tracks were resumed on Opus and Sonnet from the files on disk.")

# 3 ---------------------------------------------------------------------------
slide("Main result", [
    "Cost follows detections and point count, not entropy",
    "YOLO inference is flat: TensorRT 9.67 ms (SD 0.04 ms across frames); PyTorch ~30.6 ms",
    "Only input-dependent thermal cost is NMS post-processing: +1.82 ms for any detection, +0.16 ms per candidate",
    "Thermal entropy HT varied 11× (0.017–0.194) while paced power stayed within 0.013 W (5.78–5.80 W)",
    "Once detections are controlled, the HT slope is zero: 2.2 ± 1.9 ms per unit HT (65 banks)",
    "Simulated LiDAR fog raises entropy-stage time up to 2× (374 → 741 ms per window), driven by voxel count (r = 0.99), not entropy (r = 0.31)",
    "Module power under load is flat in every experiment; fog changes busy time, not watts",
], layout="text", notes="Sources: results/jetson-harness/step2_*, cmp_verdicts.csv; results/thermal-physics/jetson_bank_runs_paced.json; results/lidar-sim/cost_combined.md.")

# 4 ---------------------------------------------------------------------------
slide("Null test", [
    "88 control trials, all passed telemetry checks",
    "Inference does not follow the image; PyTorch sits at discrete GPU-clock levels (~18/24/27/31 ms)",
    "NMS time tracks pre-NMS candidates (r = 0.93): ~1.7 ms at 0, 3.2 ms at 1, 4.7 ms at 2–10, ~11 ms at 40",
    "Cause of the 0 → 1 step: Ultralytics NMS exits early when there are no candidates",
    "TensorRT FP16 vs PyTorch FP32 (unpaced): 42.5 vs 63.9 ms per frame, 277 vs 465 mJ per frame",
], img=R / "jetson-harness/fig_inference_vs_nms.png", layout="bottom",
    notes="results/jetson-harness/step2_summary.json, step2_stage_correlations.csv. The Guide specifies TensorRT; the thermal pipeline has been PyTorch FP32.")

# 5 ---------------------------------------------------------------------------
slide("Noise floor", [
    "Same real bank, 8 repeats per configuration",
    "PyTorch unpaced: 63.88 ms per frame, SD 0.12 ms (0.2%)",
    "Power SD 0.13 W unpaced; one VDD_IN step is 39.5 mW",
    "Across different banks: residual SD 0.39 ms and 0.045 W; smallest detectable two-bank difference 1.1 ms",
    "Removing BIE from the loop: 64.4 → 25.2 ms per frame, but power rises 7.16 → 9.46 W (GPU stays clocked up)",
    "Paced 8.7 fps is slower per frame than unpaced (73.8 vs 63.9 ms): clocks drop in the idle gaps",
], img=R / "jetson-harness/fig_noise_floor.png", layout="bottom",
    notes="Latency depends on duty cycle through the clock governor. That is a confound for any latency-vs-entropy claim. results/jetson-harness/step2_noise_floor.csv, cmp_summary.json.")

# 6 ---------------------------------------------------------------------------
slide("Thermal comparison", [
    "65 non-adversarial banks from 4 generators plus real frames; 99 trials, 2 blocks (3 were planned)",
    "Raw slope: +1.6 ms per frame over the full HT range (8.3 ms per unit HT, CI 3.6–14.8)",
    "Controlled for any-detection and candidates: 2.2 ± 1.9 ms per unit HT, i.e. zero",
    "People count 0 → 8: +0.010 W per person (CI −0.009 to 0.023)",
    "No knee: hinge model never beats a straight line",
], img=R / "jetson-harness/fig_cmp_thermal_cost_vs_HT.png", layout="bottom",
    notes="results/jetson-harness/cmp_verdicts.csv, cmp_thermal_banks.csv. Hyp A holds only in its trivial form; Hyp C has no thermal support.")

# 7 ---------------------------------------------------------------------------
slide("Sponge search", [
    "Hardware-in-the-loop CMA-ES over plausible scenes (people, warm objects, 21 °C room)",
    "Best plausible scene: +4.4 ms per frame (CI 3.7–5.0, +6.8%), +30 mJ per frame; 16 rounds × 40 frames",
    "At 8.7 fps: +5.8 ms per frame, +7.5 mJ per frame period (+1.1%); mW unchanged",
    "All of it is post-processing (1.7 → 8.6 ms); inference flat",
    "Non-physical upper bound (crafted pixels): +909 ms per frame, 3,026 candidates; power drops 6–13%",
    "Confidence threshold 0.5 cuts the bound ~45%; max_det barely helps",
], img=R / "sponge-search/stage_breakdown.png", img2=R / "sponge-search/best_scenes.png",
    notes="results/sponge-search/paired_finalists.json, paired_paced.json, paired_upper.json. Scenes are rendered, not captured with the real camera; each search ran 5 minutes. The interim +9.1 ms did not survive re-measurement.")

# 8 ---------------------------------------------------------------------------
slide("Entropy decoupled", [
    "1,478 timed evaluations from the searches",
    "Cost vs HT: Pearson r = 0.015 (Spearman 0.27)",
    "Cost vs detections: Spearman 0.45; ~1.35 ms and ~11.5 mJ per detection",
    "Partial r(cost, HT | detections, candidates) = −0.01",
    "One scene has HT 0.201 (15× the empty room) at +0.14 ms (CI −0.4 to 0.7)",
    "No scene raised cost while HT stayed in the empty-room band",
], img=R / "sponge-search/cost_vs_entropy_cloud.png",
    notes="results/sponge-search/cloud_stats.json. Single evaluations are noisy, which is why Pearson correlations are small even for detections.")

# 9 ---------------------------------------------------------------------------
slide("LiDAR fog: entropy", [
    "Simulation only: Mid-360 model, first-return rule, fog as extinction (B1/B2/B3 = 0.1/0.3/1.0 per m)",
    "H_eigen rises clearly: +0.18 / +0.21 / +0.22 from 0.837 (clear-air SD 0.0012 per window)",
    "Scene complexity A0 → A3 moves H_eigen only +0.018",
    "H_eigen saturates near 1.05 by light fog; onset is model-dependent",
    "H_DS (the metric in compute_hl.py) moves +0.03 to +0.07 and flips sign with voxel size: not a fog metric",
    "Clear-air check vs Sam's lab data: H_eigen 0.834 vs 0.823, voxels 11,743 vs 12,146",
], img=R / "lidar-sim/fig_hl_vs_fog.png",
    notes="results/lidar-sim/fog_sweep_first.csv, real_reference.json. No real fog data exists; scan pattern is an analytic approximation.")

# 10 --------------------------------------------------------------------------
slide("LiDAR fog: cost", layout="text", bullets=[
    "Entropy-stage time follows occupied voxels (r = 0.99, ~33 ms per 1,000 voxels); vs H_eigen r = 0.31",
    "Cost rises then falls with fog; loaded power stays 6.3–6.6 W",
    "In clear air the entropy node costs ~6× the clustering workload; at B2, 88% of the extra energy is entropy measurement",
    "Fog removes the person from the clustering output at B1–B2",
], tbl=[
    ["Condition", "Voxels", "H_eigen", "Entropy stage", "Clustering", "Person found"],
    ["B0 clear", "11,820", "0.837", "374 ms, 2.39 J", "66 ms, 0.40 J", "1.00"],
    ["B1", "15,338", "0.889", "480 ms, 3.06 J", "82 ms, 0.50 J", "0.00"],
    ["B2", "19,348", "0.931", "592 ms, 3.77 J", "96 ms, 0.59 J", "0.00"],
    ["B3", "17,123", "1.044", "572 ms, 3.65 J", "105 ms, 0.64 J", "0.40"],
    ["Peak (α 0.45)", "22,797", "0.977", "741 ms, 4.71 J", "112 ms, 0.68 J", "0.00"],
    ["Very heavy (α 3.0)", "7,477", "1.022", "264 ms, 1.70 J", "80 ms, 0.50 J", "0.19"],
], notes="results/lidar-sim/cost_combined.md, measured on the Jetson on simulated clouds, one pass (the second was cancelled: queue). These banks use the strongest-echo rule, so H_eigen differs slightly from the first-return sweep on the previous slide.")

# 11 --------------------------------------------------------------------------
slide("Thermal in fog", [
    "Simulated: Beer-Lambert plus path emission, mist 1 K cooler than air",
    "HT is not flat with a person in view: −4.5% (B1), −14.4% (B2), −34.0% (B3)",
    "Empty room: −2.8 / −7.0 / −9.7%",
    "Plain Shannon entropy on a fixed 10–50 °C scale: −1.0 / −1.6 / −2.6%",
    "Thermal cost in fog is flat: 63.98–64.18 ms, 7.12–7.13 W",
    "Experiment B needs an equivalence margin for 'HT flat', or a steadier metric",
], img=R / "thermal-physics/ht_vs_fog.png", layout="bottom",
    notes="results/thermal-physics/fog_levels_summary.json, jetson_bank_summary.json. Fog constants are from literature; no real fog recording exists.")

# 12 --------------------------------------------------------------------------
slide("Thermal realism", [
    "Physics renderer: temporal noise 25.0 vs 23.6–24.0 mK real; HT within 0.78–1.49× of real",
    "  YOLO detects its people in 0–29% of frames (real: 57–100%)",
    "Data-driven: HT within 0.94–1.09× of held-out real sessions (real-vs-real floor 0.93–1.03)",
    "  Classifier tells synthetic from real at 91–99% (real-vs-real floor 72–95%)",
    "  Synthetic A1 costs the same as real A1: −0.40 ms (CI −1.08 to 0.29)",
    "  But YOLO finds the person less often: 0.65 vs 0.97 detections per frame",
    "Renderer speed on Jetson CUDA: 9.5–87 fps",
], img=R / "thermal-physics/contact_sheet_real_vs_sim.png",
    img2=R / "thermal-datadriven/contact_sheet_real_vs_synth.png",
    notes="results/thermal-physics/validation.json; results/thermal-datadriven/validation/*.json; results/jetson-harness/cmp_named_comparisons.csv. Detection rate is the realism measure that matters for cost, and both generators under-drive it.")

# 13 (generative) and any late slides -----------------------------------------
if EXTRA.exists():
    for e in json.loads(EXTRA.read_text()):
        slide(e["title"], e.get("bullets"), ROOT / e["img"] if e.get("img") else None,
              layout=e.get("layout", "right"), notes=e.get("notes", ""))

# 14 --------------------------------------------------------------------------
slide("Radar witness", [
    "IWR6843ISK FMCW simulation; no real radar data yet",
    "Fog two-way loss over 4 m at 63 GHz: 0.0015–0.029 dB (B1–B3)",
    "Detections flat at 3.14 per frame at every fog level",
    "A 50 µm water film on the radome costs 9.0 dB: keep the sensor dry",
    "2 cm radial shift and 20 cm step detected every time; sideways 2 cm only 67% (8% with sway)",
    "Cost on Jetson: ~6.8 ms fixed + 0.52 ms + 0.019 ms per detection",
], img=R / "radar-sim/fog_attenuation.png", layout="bottom",
    notes="results/radar-sim/, docs/radar-sim.md. Radar is a fog-blind control channel; its cost scales only with detections.")

# 15 --------------------------------------------------------------------------
slide("Room model", [
    "Built from 5 phone photos with gpt-6.2-astra (medium effort) via Codex CLI",
    "2.56 × 2.66 × 2.56 m, 40 objects, door and glass openings",
    "Scale rests on nominal desk height and door width: could be 5–10% off",
    "Sensor poses are assumed from the photos",
    "Weakest fit: photo 4, 26 px RMS",
    "One tape-measured wall or a phone LiDAR scan would fix scale",
], img=R / "room-3d/compare_photo_4.png", img2=R / "room-3d/floorplan.png",
    notes="scene/room_v1.json, docs/room-3d.md (includes a 10-minute capture protocol).")

# 16 --------------------------------------------------------------------------
slide("Heat-lamp artefact", [
    "Live A1 − A0 gap was 263 mW",
    "Between A1 runs 10 and 11 (same condition) live inference jumped 30.8 → 37.0 ms and power +365 mW",
    "Replaying the same frames shows no jump",
    "83% of the live gap is that system-state change",
    "Replay gives A1 − A0 = 73 mW (1.3%) and 3.6 ms, all post-processing",
    "All A0 trials ran before all A1 trials",
], img=R / "experiment-audit/fig_heatlamp_live_vs_replay.png", layout="bottom",
    notes="results/experiment-audit/heatlamp_live_vs_replay.json. The cause of the jump is not in the logs.")

# 17 --------------------------------------------------------------------------
slide("Design problems", [
    "HT (BIE) is not a disorder measure: it falls 0.077 → 0.024 as added noise rises",
    "H_DS is not an isotropy measure; the fog argument needs H_eigen",
    "Measuring entropy costs more than anything it changes: BIE is 41–46% of the thermal loop; HL node ~6× the LiDAR consumer",
    "Nothing in the real stack consumes LiDAR except the entropy node",
    "Effects are 0.6–1.3% of module power; 3 repeats resolve only 48–68 mW",
    "Trials within a session are autocorrelated (lag-1 0.86); drift ~12 mW per °C",
    "Detector post-processing vs object count on a Jetson is already published (Liu et al., arXiv 2209.05487)",
], img=R / "experiment-audit/fig_power_analysis.png",
    notes="docs/experiment-audit.md, docs/research/. Literature items were read by the research track; several only at snippet level.")

# 18 --------------------------------------------------------------------------
slide("Hypotheses", layout="text", tbl=[
    ["", "Tonight's evidence", "Status"],
    ["Null: fixed graph, constant cost", "Inference and watts flat in every experiment; latency varies only through NMS and point count", "Holds for network and power; fails for post-processing time"],
    ["A: complexity raises entropy and power", "Detections add 1.8–6 ms and tens of mJ per frame; entropy adds nothing beyond that", "Trivial form only"],
    ["B: fog raises HL, HT flat, power rises", "Simulated: H_eigen up, thermal HT down 4–34%, energy per window up to +66%, watts flat; most of the cost is the entropy node", "Partly, in time and energy; not testable as designed"],
    ["C: knee in power vs entropy", "No thermal knee; LiDAR cost rises then falls with fog (tracks voxels)", "No support for a power knee"],
], widths=[3.0, 5.9, 3.4], notes="results/jetson-harness/cmp_verdicts.csv, cmp_summary.json; results/lidar-sim/. All fog evidence is simulated.")

# 19 --------------------------------------------------------------------------
slide("Recommendations", [
    "Stop testing 'entropy → YOLO cost'; report cost against detections, candidates and voxel count",
    "Compute entropy offline, outside the measured process",
    "Put a real LiDAR consumer in the loop (clustering or late fusion) and report it separately from the entropy node",
    "Pre-register Experiment B with H_eigen, mean S and voxel count, with expected signs from the simulation",
    "Give 'HT flat' an equivalence margin, or use plain Shannon entropy on a fixed scale",
    "Randomised blocks, 10+ trials per condition, chip temperature as covariate, log clock frequencies",
    "Report energy per frame and busy time, not only watts; run paced and unpaced",
    "Decide TensorRT vs PyTorch once: the Guide says TensorRT and it is 1.5× faster here",
    "Reframe novelty: board power under a real obscurant on a thermal + LiDAR pipeline is still open",
], layout="text", notes="docs/experiment-audit.md section 7 ranks 14 recommendations.")

# 20 --------------------------------------------------------------------------
slide("Limits", [
    "All fog results are simulated; no real fog, radar or Mid-360 scan-pattern data was used",
    "Cross-approach comparison ran 1–2 blocks, not 3; LiDAR cost is a single pass",
    "Synthetic people are detected less often than real ones, so synthetic banks under-drive NMS cost",
    "Power is tegrastats VDD_IN (module input, 39.5 mW steps), not wall power; CT-3 not attached",
    "Room scale and sensor poses are estimated, not measured",
    "Sponge scenes were rendered, not built and filmed",
    "First wave hit the usage limit after ~30 minutes; tracks were resumed on Opus and Sonnet",
    "One Jetson, one lock: up to 14 queued jobs, which cost about 2.5 hours of waiting",
], layout="text", notes="Each track's docs/<track>.md has its own 'unverified' list.")

# 21 --------------------------------------------------------------------------
slide("Jetson overnight", [
    "tegrastats every 5 s from 21:50",
    "Peaks near 20 W and 69 °C during on-device training; no throttling range reached",
    "Paced thermal runs sit near 5.8 W; unpaced near 7.2 W",
], img=R / "overnight/overnight_power_temp.png", layout="bottom",
    notes="results/overnight/tegrastats_2026-11-23_overnight.log.")

# 22 --------------------------------------------------------------------------
slide("Where things are", [
    "Repo: github.com/alibaduqail/entropy-synth-lab (private), all tracks merged into main",
    "Write-ups: docs/<track>.md; figures and CSVs: results/<track>/",
    "Message board: GitHub issues labelled 'board' (#1–#11)",
    "Jetson: ~/synthlab/harness (runners), ~/synthlab/*/banks (103 banks), ~/synthlab/overnight_log",
    "Room: scene/room_v1.json, .obj, .glb",
    "Rebuild this deck: deliverables/slides/build_deck.py",
], layout="text")

prs.save(OUT)
print("saved", OUT, len(prs.slides._sldIdLst), "slides")
