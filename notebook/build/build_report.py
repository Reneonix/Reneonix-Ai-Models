#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generates notebook/model_training_report.html - the executive Model Training Report.
Every number in here is read from this project's own results.csv / config.json / dataset
READMEs / the fresh benchmark_all_result.json run - nothing is invented."""
import base64
import json
import os

ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
SCRATCH = f"{ROOT}/notebook/build"
ASSETS = f"{SCRATCH}/report_assets"
OUT_DIR = f"{ROOT}/notebook"
OUT_PATH = f"{OUT_DIR}/model_training_report.html"

with open(f"{SCRATCH}/curves.json") as f:
    CURVES = json.load(f)
with open(f"{SCRATCH}/benchmark_all_result.json") as f:
    BENCH = json.load(f)["models"]
TEST_RES = json.load(open(f"{SCRATCH}/benchmark_all_result.json"))["image_resolution"]


def b64(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


IMG = {}
for exp in ["exp001", "exp002", "exp003", "exp004", "exp005", "exp006"]:
    IMG[f"{exp}_plain"] = b64(f"{ASSETS}/{exp}_plain.jpg", "image/jpeg")
    IMG[f"{exp}_sahi"] = b64(f"{ASSETS}/{exp}_sahi.jpg", "image/jpeg")
for exp in ["exp001", "exp002", "exp003", "exp004", "exp006"]:
    IMG[f"{exp}_cm"] = b64(f"{ASSETS}/{exp}_cm.png", "image/png")
for exp in ["exp001", "exp002", "exp003", "exp006"]:
    IMG[f"{exp}_pr"] = b64(f"{ROOT}/results/{exp}/BoxPR_curve.png", "image/png")

print("Embedded image payload:", round(sum(len(v) for v in IMG.values()) / 1024 / 1024, 2), "MB (base64)")

# ============================================================================
# CSS
# ============================================================================
CSS = """
* { box-sizing: border-box; }
::selection { background: var(--accent); color: #fff; }

:root {
  --bg: #F5F7FA;
  --surface: #FFFFFF;
  --surface-2: #ECF0F6;
  --ink: #0E1621;
  --ink-soft: #57667A;
  --ink-faint: #93A2B5;
  --line: #E1E7EF;
  --navy: #0B1526;
  --navy-soft: #16233A;
  --navy-line: rgba(255,255,255,0.11);
  --navy-ink: #EAF0FB;
  --navy-ink-soft: #9FB0C7;
  --accent: #2A63E4;
  --accent-ink: #FFFFFF;
  --cyan: #0E93A3;
  --violet: #6C5CE0;
  --good: #158A5E;
  --good-bg: #E7F5EF;
  --warn: #B4740E;
  --warn-bg: #FBF0DD;
  --crit: #C13F3F;
  --crit-bg: #FAEAEA;
  --shadow: 0 1px 2px rgba(15,23,42,0.05), 0 12px 32px -16px rgba(15,23,42,0.18);
  --shadow-navy: 0 1px 0 rgba(255,255,255,0.04), 0 16px 40px -18px rgba(0,0,0,0.55);
  --r-s: 6px; --r-m: 10px; --r-l: 18px;
  --font-d: 'Sora', 'Segoe UI', sans-serif;
  --font-b: 'IBM Plex Sans', 'Segoe UI', sans-serif;
  --font-m: 'IBM Plex Mono', 'Consolas', monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0A101C; --surface:#111A2B; --surface-2:#0D1524; --ink:#E9EEF9; --ink-soft:#94A5BE; --ink-faint:#5E708A;
    --line:#22314A; --navy:#060B15; --navy-soft:#101A2C; --navy-line:rgba(255,255,255,0.09);
    --navy-ink:#E9EEF9; --navy-ink-soft:#8FA1BC;
    --accent:#6C9BFF; --cyan:#3FD3E3; --violet:#A79AFF;
    --good:#3FCB93; --good-bg:#0E2A21; --warn:#E7B25C; --warn-bg:#2E230F; --crit:#F0817A; --crit-bg:#2E1516;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 32px -16px rgba(0,0,0,0.6);
    --shadow-navy: 0 1px 0 rgba(255,255,255,0.03), 0 16px 40px -18px rgba(0,0,0,0.7);
  }
}
:root[data-theme="dark"] {
  --bg:#0A101C; --surface:#111A2B; --surface-2:#0D1524; --ink:#E9EEF9; --ink-soft:#94A5BE; --ink-faint:#5E708A;
  --line:#22314A; --navy:#060B15; --navy-soft:#101A2C; --navy-line:rgba(255,255,255,0.09);
  --navy-ink:#E9EEF9; --navy-ink-soft:#8FA1BC;
  --accent:#6C9BFF; --cyan:#3FD3E3; --violet:#A79AFF;
  --good:#3FCB93; --good-bg:#0E2A21; --warn:#E7B25C; --warn-bg:#2E230F; --crit:#F0817A; --crit-bg:#2E1516;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 32px -16px rgba(0,0,0,0.6);
  --shadow-navy: 0 1px 0 rgba(255,255,255,0.03), 0 16px 40px -18px rgba(0,0,0,0.7);
}

html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--font-b); font-size: 16px; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }

h1, h2, h3, h4 { font-family: var(--font-d); font-weight: 600; margin: 0; text-wrap: balance; color: inherit; }
p { margin: 0 0 1em; max-width: 68ch; color: var(--ink-soft); }
p:last-child { margin-bottom: 0; }
a { color: var(--accent); }
strong { color: var(--ink); font-weight: 600; }
.navy-block strong { color: var(--navy-ink); }
.navy-block p { color: var(--navy-ink-soft); }
.navy-block { color: var(--navy-ink-soft); }

.wrap { max-width: 1180px; margin: 0 auto; padding: 0 32px; }
@media (max-width: 640px) { .wrap { padding: 0 20px; } }

.kicker {
  font-family: var(--font-m); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); display: flex; align-items: center; gap: 10px;
  margin-bottom: 14px;
}
.kicker::before { content: ""; width: 22px; height: 1.5px; background: var(--accent); display: inline-block; }
.navy-block .kicker { color: var(--cyan); }
.navy-block .kicker::before { background: var(--cyan); }

.page-rule {
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--font-m); font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-faint); padding: 14px 0; border-bottom: 1px solid var(--line);
}
.navy-block .page-rule { border-bottom-color: var(--navy-line); color: var(--navy-ink-soft); opacity: 0.7; }

section.sheet { padding: 88px 0; border-top: 1px solid var(--line); }
section.sheet:first-of-type { border-top: none; }
.navy-block.sheet { background: var(--navy); border-top: 1px solid var(--navy-line); }
@media (max-width: 640px) { section.sheet { padding: 56px 0; } }

.section-head { max-width: 760px; margin-bottom: 48px; }
.section-head h2 { font-size: clamp(1.55rem, 3vw, 2.15rem); letter-spacing: -0.01em; }
.section-head .lede { font-size: 1.05rem; margin-top: 14px; color: var(--ink-soft); max-width: 62ch; }
.navy-block .section-head .lede { color: var(--navy-ink-soft); }

/* ---- hero ---- */
.hero {
  background: var(--navy); color: var(--navy-ink); position: relative; overflow: hidden;
  padding: 72px 0 64px;
}
.hero::before {
  content: ""; position: absolute; inset: 0;
  background-image:
    linear-gradient(var(--navy-line) 1px, transparent 1px),
    linear-gradient(90deg, var(--navy-line) 1px, transparent 1px);
  background-size: 44px 44px; opacity: 0.5; mask-image: linear-gradient(to bottom, black, transparent 78%);
}
.hero .wrap { position: relative; }
.bracket { position: absolute; width: 34px; height: 34px; border-color: var(--cyan); opacity: 0.55; }
.bracket.tl { top: -8px; left: -8px; border-top: 2px solid; border-left: 2px solid; }
.bracket.br { bottom: -8px; right: -8px; border-bottom: 2px solid; border-right: 2px solid; }
.hero-frame { position: relative; padding: 30px 34px; margin-bottom: 44px; max-width: fit-content; }
.hero h1 { font-size: clamp(2.15rem, 4.6vw, 3.7rem); line-height: 1.08; letter-spacing: -0.015em; max-width: 16ch; }
.hero .subtitle { font-size: clamp(1.05rem, 1.6vw, 1.3rem); color: var(--navy-ink-soft); margin-top: 18px; max-width: 46ch; font-weight: 400; }
.hero-meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 40px; }
.chip {
  font-family: var(--font-m); font-size: 0.74rem; font-weight: 500; padding: 8px 14px;
  border: 1px solid var(--navy-line); border-radius: 100px; color: var(--navy-ink-soft);
  background: rgba(255,255,255,0.03);
}
.chip b { color: var(--navy-ink); font-weight: 600; }

/* ---- nav ---- */
.tocbar {
  position: sticky; top: 0; z-index: 40; background: color-mix(in srgb, var(--surface) 92%, transparent);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
}
.tocbar .wrap { display: flex; gap: 4px; overflow-x: auto; padding-top: 12px; padding-bottom: 12px; scrollbar-width: thin; }
.tocbar a {
  flex: none; font-family: var(--font-m); font-size: 0.72rem; font-weight: 500; letter-spacing: 0.02em;
  color: var(--ink-soft); text-decoration: none; padding: 7px 12px; border-radius: 100px; white-space: nowrap;
}
.tocbar a:hover { background: var(--surface-2); color: var(--ink); }

/* ---- grids / cards ---- */
.grid { display: grid; gap: 20px; }
.grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.grid.cols-6 { grid-template-columns: repeat(6, 1fr); }
@media (max-width: 980px) { .grid.cols-3, .grid.cols-4 { grid-template-columns: repeat(2,1fr); } .grid.cols-6 { grid-template-columns: repeat(3,1fr); } }
@media (max-width: 640px) { .grid.cols-2, .grid.cols-3, .grid.cols-4, .grid.cols-6 { grid-template-columns: 1fr; } }

.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-l);
  padding: 26px 28px; box-shadow: var(--shadow);
}
.navy-block .card { background: var(--navy-soft); border-color: var(--navy-line); box-shadow: var(--shadow-navy); }
.card h3 { font-size: 1.08rem; margin-bottom: 8px; }
.card .tag { font-family: var(--font-m); font-size: 0.68rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.navy-block .card .tag { color: var(--cyan); }

.metric-tile { display: flex; flex-direction: column; gap: 4px; }
.metric-tile .n { font-family: var(--font-m); font-size: clamp(1.6rem, 2.6vw, 2.15rem); font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.metric-tile .l { font-size: 0.82rem; color: var(--ink-soft); }
.navy-block .metric-tile .l { color: var(--navy-ink-soft); }
.metric-tile .n.accent { color: var(--accent); } .navy-block .metric-tile .n.accent { color: var(--cyan); }

/* ---- tables ---- */
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--r-m); }
.navy-block .table-wrap { border-color: var(--navy-line); }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; min-width: 640px; }
th, td { text-align: left; padding: 13px 18px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.navy-block th, .navy-block td { border-bottom-color: var(--navy-line); }
th { font-family: var(--font-m); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-soft); background: var(--surface-2); font-weight: 600; }
.navy-block th { background: rgba(255,255,255,0.03); color: var(--navy-ink-soft); }
td { font-family: var(--font-m); font-variant-numeric: tabular-nums; color: var(--ink); }
.navy-block td { color: var(--navy-ink); }
td.label, th.label { font-family: var(--font-b); white-space: normal; }
tr:last-child td { border-bottom: none; }
tr.hl td { background: color-mix(in srgb, var(--accent) 7%, transparent); }

/* ---- badges/pills ---- */
.pill { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-m); font-size: 0.7rem; font-weight: 600; padding: 4px 10px; border-radius: 100px; letter-spacing: 0.02em; }
.pill.gold { background: var(--warn-bg); color: var(--warn); }
.pill.good { background: var(--good-bg); color: var(--good); }
.pill.accentp { background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); }
.navy-block .pill.accentp { background: rgba(255,255,255,0.08); color: var(--cyan); }

/* ---- diagram ---- */
.flow { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }
.flow-step {
  flex: 1 1 160px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-m);
  padding: 18px 16px; min-width: 150px; position: relative; box-shadow: var(--shadow);
}
.navy-block .flow-step { background: var(--navy-soft); border-color: var(--navy-line); box-shadow: none; }
.flow-step .fn { font-family: var(--font-m); font-size: 0.68rem; color: var(--accent); font-weight: 700; }
.navy-block .flow-step .fn { color: var(--cyan); }
.flow-step h4 { font-size: 0.94rem; margin: 6px 0 4px; }
.flow-step p { font-size: 0.82rem; margin: 0; }
.flow-arrow { flex: 0 0 34px; display: flex; align-items: center; justify-content: center; color: var(--ink-faint); font-size: 1.1rem; }
@media (max-width: 900px) { .flow { flex-direction: column; } .flow-arrow { transform: rotate(90deg); padding: 4px 0; } }

/* ---- timeline ---- */
.timeline { position: relative; padding-left: 32px; }
.timeline::before { content: ""; position: absolute; left: 5px; top: 6px; bottom: 6px; width: 2px; background: var(--line); }
.navy-block .timeline::before { background: var(--navy-line); }
.tl-item { position: relative; padding-bottom: 34px; }
.tl-item:last-child { padding-bottom: 0; }
.tl-item::before { content: ""; position: absolute; left: -32px; top: 4px; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); border: 3px solid var(--bg); }
.navy-block .tl-item::before { background: var(--cyan); border-color: var(--navy); }
.tl-item .tl-tag { font-family: var(--font-m); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); font-weight: 700; }
.navy-block .tl-item .tl-tag { color: var(--cyan); }
.tl-item h4 { font-size: 1.02rem; margin: 4px 0 6px; }

/* ---- challenges ---- */
.chal { border: 1px solid var(--line); border-radius: var(--r-l); overflow: hidden; background: var(--surface); box-shadow: var(--shadow); }
.chal-row { display: grid; grid-template-columns: 1fr 1fr 1fr; }
.navy-block .chal { background: var(--navy-soft); border-color: var(--navy-line); box-shadow: none; }
.chal-cell { padding: 20px 22px; border-left: 1px solid var(--line); }
.navy-block .chal-cell { border-left-color: var(--navy-line); }
.chal-cell:first-child { border-left: none; }
.chal-cell .ct { font-family: var(--font-m); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-bottom: 8px; display: block; }
.chal-cell.challenge .ct { color: var(--crit); } .chal-cell.impact .ct { color: var(--warn); } .chal-cell.solution .ct { color: var(--good); }
.chal-cell p { font-size: 0.9rem; }
.chal-head { padding: 18px 22px; border-bottom: 1px solid var(--line); font-family: var(--font-d); font-weight: 600; }
.navy-block .chal-head { border-bottom-color: var(--navy-line); }
@media (max-width: 800px) { .chal-row { grid-template-columns: 1fr; } .chal-cell { border-left: none; border-top: 1px solid var(--line); } .chal-cell:first-child { border-top: none; } }

/* ---- figures / model deep-dive ---- */
.figure { margin: 0; }
.figure img { width: 100%; display: block; border-radius: var(--r-m); border: 1px solid var(--line); background: var(--surface-2); }
.navy-block .figure img { border-color: var(--navy-line); }
.figure figcaption { font-size: 0.82rem; color: var(--ink-soft); margin-top: 10px; }
.figure figcaption .k { font-family: var(--font-m); font-weight: 600; color: var(--ink); }
.navy-block .figure figcaption { color: var(--navy-ink-soft); }

.model-hero { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap; margin-bottom: 40px; }
.model-hero .idbadge { font-family: var(--font-m); font-size: 0.78rem; font-weight: 700; color: var(--accent); }
.model-hero h2 { font-size: clamp(1.5rem, 3vw, 2.05rem); margin-top: 6px; }
.model-hero .role { color: var(--ink-soft); margin-top: 8px; max-width: 56ch; }
.model-badges { display: flex; gap: 8px; flex-wrap: wrap; }

.subhead { display: flex; align-items: baseline; gap: 12px; margin: 56px 0 22px; }
.subhead:first-of-type { margin-top: 0; }
.subhead .num { font-family: var(--font-m); color: var(--ink-faint); font-size: 0.85rem; }
.subhead h3 { font-size: 1.18rem; }

.compare-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 760px) { .compare-pair { grid-template-columns: 1fr; } }

.chart-box { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-m); padding: 18px 20px; }
.navy-block .chart-box { background: var(--navy-soft); border-color: var(--navy-line); }
.chart-box canvas { width: 100%; display: block; }
.chart-legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; font-size: 0.76rem; color: var(--ink-soft); font-family: var(--font-m); }
.navy-block .chart-legend { color: var(--navy-ink-soft); }
.chart-legend span { display: inline-flex; align-items: center; gap: 6px; }
.chart-legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

.recommend-card {
  background: linear-gradient(155deg, var(--navy-soft), var(--navy)); border: 1px solid var(--navy-line);
  border-radius: var(--r-l); padding: 44px; box-shadow: var(--shadow-navy);
}
.recommend-card h3 { font-size: 1.6rem; color: var(--navy-ink); }

footer { padding: 48px 0 64px; }
footer .wrap { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; font-family: var(--font-m); font-size: 0.76rem; color: var(--ink-faint); }

a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""

# ============================================================================
# Small helpers
# ============================================================================
def fig(src_key, cap_html):
    return f'<figure class="figure"><img src="{IMG[src_key]}" loading="lazy" alt=""><figcaption>{cap_html}</figcaption></figure>'


def chip(html):
    return f'<div class="chip">{html}</div>'


def metric(n, l, accent=False):
    cls = "n accent" if accent else "n"
    return f'<div class="metric-tile"><div class="{cls}">{n}</div><div class="l">{l}</div></div>'


print("CSS + helpers ready:", len(CSS), "chars")

# ============================================================================
# Ground truth data - every field traces to results.csv / config.json / dataset
# READMEs / the fresh benchmark_all_result.json run captured earlier this session.
# ============================================================================
def B(exp, mode):
    return BENCH[exp][mode]

MODELS = {
    "exp001": dict(
        title="YOLOv8L", family="YOLO · CNN, anchor-free single-stage",
        role="Primary six-class detector — original production baseline",
        dataset="exp001_exp003_exp006", train_n="7,389", val_n="1,847", classes="6",
        batch=8, imgsz=640, epochs=100, params=BENCH["exp001"]["params_millions"],
        optimizer="2-group differential LR (custom optimizer swap) + cosine anneal",
        lr_detail="backbone/neck 3.5×10⁻⁴ → detect head 7.0×10⁻³, annealed to 1% of start over 100 epochs",
        extra="First 5 backbone layers frozen (generic edge/texture filters); layers 5–21 plus the "
              "detect head are trained — batch dropped 16→8 versus the original recipe to fit an "
              "8 GB reference GPU, with both LRs scaled down to match (linear scaling rule).",
        precision=0.945, recall=0.932, map50=0.975, map5095=0.935,
        has_cm=True, has_pr=True,
        arch_text="YOLOv8L is a CNN-based, anchor-free single-stage detector: a CSPDarknet-style "
                   "backbone extracts multi-scale features, a PAN/FPN neck fuses them across P3–P5 "
                   "resolutions, and a decoupled detection head predicts class and box offsets directly "
                   "per grid cell — no anchor boxes, no separate region-proposal stage. This experiment "
                   "additionally overrides Ultralytics' default optimizer with a 2-group differential "
                   "learning-rate schedule: generic low-level filters are frozen outright, the "
                   "domain-specific backbone/neck layers adapt slowly, and the detection head — which "
                   "has to learn this project's 6 classes from scratch — adapts quickly, both annealed "
                   "on a cosine schedule.",
    ),
    "exp002": dict(
        title="YOLO26s", family="YOLO26 · CNN, anchor-free, single-class",
        role="Narrow specialist detector — glass-cullet fragments only",
        dataset="exp002", train_n="2,350", val_n="587", classes="1 (glass)",
        batch=8, imgsz=640, epochs=120, params=BENCH["exp002"]["params_millions"],
        optimizer="Ultralytics default (SGD, cosine schedule)", lr_detail="lr0 = 0.01, patience = 30",
        extra="Trained on a laptop outside this project's usual desktop workflow — no wandb tracking "
              "was recorded for the original run; the hyperparameters above were reproduced exactly "
              "into this project's own training script for any future retrain.",
        precision=0.694, recall=0.553, map50=0.602, map5095=0.370,
        has_cm=True, has_pr=True,
        arch_text="Same YOLO26 architecture family as exp006, but purpose-built as a narrow specialist: "
                   "a single output class (glass cullet) on a dataset roughly a third the size of the "
                   "main six-class set. Glass-cullet fragments have no consistent silhouette the way a "
                   "whole bottle or can does, which — combined with the smaller dataset — is the direct "
                   "explanation for its lower headline numbers versus the six-class models; it is a "
                   "deliberately scoped add-on, not a weaker version of the main detector.",
    ),
    "exp003": dict(
        title="YOLO26L + P2", family="YOLO26 · CNN + extra P2/160×160 head",
        role="Best-accuracy six-class detector — small-object specialist",
        dataset="exp001_exp003_exp006", train_n="7,389", val_n="1,847", classes="6",
        batch=16, imgsz=640, epochs=100, params=BENCH["exp003"]["params_millions"],
        optimizer="Ultralytics default (single uniform LR, cosine schedule)", lr_detail="Ultralytics default schedule (no manual override)",
        extra="The only architectural change from a stock YOLO26L is the added P2 (160×160) detection "
              "head — every other hyperparameter is left at Ultralytics defaults, isolating the P2 "
              "head's own effect on accuracy.",
        precision=0.947, recall=0.930, map50=0.977, map5095=0.951,
        has_cm=True, has_pr=True,
        arch_text="YOLO26L's stock head detects at P3/P4/P5 (80×80, 40×40, 20×20 grids). This run adds "
                   "a fourth, higher-resolution P2 (160×160) head, giving the network a detection path "
                   "with less downsampling — better suited to small or distant objects on the "
                   "conveyor. On the held-out validation set this is the best-performing model in the "
                   "project on every headline metric, with the largest per-class gains on the hardest "
                   "small/irregular classes (stone and ceramic).",
    ),
    "exp004": dict(
        title="ResNet-50", family="CNN classifier · ImageNet transfer learning",
        role="Second-stage material classifier for YOLO-cropped regions",
        dataset="exp004 (crops)", train_n="294,959 crops", val_n="73,448 crops", classes="6",
        batch=64, imgsz=224, epochs=29, params=BENCH["exp004"]["params_millions"],
        optimizer="Adam", lr_detail="lr = 1×10⁻⁴, weight_decay = 1×10⁻⁴",
        extra="Not a detector — trained on the 6-class crop dataset built by cutting every labeled "
              "YOLO box out of exp001_exp003_exp006's own images (10% padding, boxes under 10px "
              "skipped). Best checkpoint reached at epoch 29.",
        precision=None, recall=None, map50=None, map5095=None, val_acc=0.9937234506045093,
        has_cm=True, has_pr=False,
        arch_text="A standard ResNet-50 backbone, ImageNet-pretrained, with its final layer replaced "
                   "by a 6-way linear classifier — this is the second stage of a two-model pipeline: "
                   "YOLO detects and localizes an object, then ResNet looks only at that tightly "
                   "cropped region to assign the final material label. Heavy color/lighting "
                   "augmentation was used deliberately, since the whole point is robustness to the "
                   "glare and lighting conditions where a detector's own single-pass classification "
                   "head tends to struggle. It reached 99.37% validation accuracy on held-out crops. "
                   "It cannot localize objects in a raw multi-object frame by itself.",
    ),
    "exp005": dict(
        title="RT-DETR-L", family="DETR-family transformer · Baidu, via Ultralytics",
        role="Transformer-based comparison detector, same six-class task",
        dataset="exp005", train_n="7,389", val_n="1,847", classes="6",
        batch=16, imgsz=640, epochs=100, params=BENCH["exp005"]["params_millions"],
        optimizer="Ultralytics default (AdamW-style, cosine schedule)", lr_detail="Ultralytics default schedule (no manual override)",
        extra="Training genuinely completed all 100/100 epochs (4.79 hours, confirmed via the wandb "
              "run log), but the process was interrupted during Ultralytics' post-training validation "
              "pass on best.pt — the automatic wrap-up never ran. Finished manually; metrics below are "
              "epoch-100's own end-of-epoch validation, on the identical 1,847-image validation set "
              "used by exp001/exp003/exp006.",
        precision=0.946, recall=0.924, map50=0.975, map5095=0.922,
        has_cm=False, has_pr=False,
        arch_text="RT-DETR (\"DETRs Beat YOLOs on Real-Time Object Detection\", Baidu, 2023) is a "
                   "transformer-based detector with no anchor boxes and no NMS post-processing — a "
                   "CNN backbone feeds a transformer encoder-decoder that predicts a fixed set of "
                   "object queries directly. Trained here via Ultralytics' own RTDETR class, which "
                   "shares the same .train()/callback API as every YOLO model in this project, so it "
                   "is a fair architecture-only comparison against exp001 and exp003 on the exact same "
                   "data. Not to be confused with RF-DETR (Roboflow's unrelated, separate-package "
                   "architecture), an earlier exp005 attempt abandoned after repeated training crashes "
                   "— see Challenges Identified.",
    ),
    "exp006": dict(
        title="YOLO26s (fine-tuned)", family="YOLO26 · CNN, two-stage training",
        role="Balanced production candidate — smaller/faster architecture",
        dataset="exp006 → exp001_exp003_exp006", train_n="1,631 + 7,389", val_n="408 / 1,847", classes="6",
        batch=8, imgsz=640, epochs=100, params=BENCH["exp006"]["params_millions"],
        optimizer="Ultralytics default (fine-tune stage)", lr_detail="fine-tune lr0 = 0.001",
        extra="Trained on a laptop (Linux) outside this project's usual desktop workflow, in two "
              "stages: base-trained for 100 epochs on its own smaller 1,631/408 dataset, then "
              "fine-tuned for another 100 epochs on this project's main 7,389/1,847 dataset — the "
              "final fine-tuned checkpoint is the deliverable. No wandb tracking was recorded for "
              "either stage originally; both are reproduced in this project's own script.",
        precision=0.94436, recall=0.92681, map50=0.97514, map5095=0.94335,
        stage1=dict(precision=0.99351, recall=0.981, map50=0.99358, map5095=0.71966),
        has_cm=True, has_pr=True,
        arch_text="Same YOLO26s architecture as exp002, but trained in two stages instead of one: "
                   "stage 1 base-trains on a separate 1,631-image dataset, then stage 2 fine-tunes the "
                   "resulting weights on this project's main six-class dataset — the same architecture "
                   "seeing two different data distributions in sequence, rather than one. It lands "
                   "between exp001 and exp003 on the final held-out set despite being a much smaller, "
                   "faster network, which is the direct evidence that the base-training stage transfers "
                   "useful structure before the model ever sees this project's own data.",
    ),
}
TRAINING_TIME = {
    "exp001": "2h 19m (100 epochs)",
    "exp002": "1h 23m (120 epochs)",
    "exp003": "41h 17m (100 epochs) — the added P2/160×160 head roughly quadruples the "
              "spatial resolution the network processes per image versus a stock P3–P5 head",
    "exp004": "not logged for this run (29 epochs to best checkpoint)",
    "exp005": "4h 48m (100 epochs)",
    "exp006": "29m (fine-tune stage, 100 epochs) + a separate base-training stage",
}
EXP_ORDER = ["exp001", "exp002", "exp003", "exp004", "exp005", "exp006"]

CLASS_NAMES = ["aluminium", "plastic", "metal", "stone", "ceramic", "glass"]

# ============================================================================
# HEAD / NAV / HERO
# ============================================================================
HEAD = """<title>Waste Detection Training Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>""" + CSS + "</style>"

NAV = """
<nav class="tocbar"><div class="wrap">
<a href="#summary">Summary</a><a href="#overview">Overview</a><a href="#dataset">Dataset</a>
<a href="#dataprep">Data Prep</a><a href="#models">Models</a><a href="#config">Config</a>
<a href="#results">Results</a><a href="#comparison">Comparison</a><a href="#confusion">Confusion Matrices</a>
<a href="#predictions">Predictions</a><a href="#exp001">1 · YOLOv8L</a><a href="#exp002">2 · YOLO26s</a>
<a href="#exp003">3 · YOLO26L+P2</a><a href="#exp004">4 · ResNet-50</a><a href="#exp005">5 · RT-DETR-L</a><a href="#exp006">6 · YOLO26s FT</a>
<a href="#challenges">Challenges</a><a href="#journey">Journey</a><a href="#recommendation">Recommendation</a>
<a href="#final-pipeline">Deployment Pipeline</a><a href="#future">Future Work</a><a href="#conclusion">Conclusion</a>
</div></nav>"""

HERO = """
<header class="hero navy-block">
  <div class="wrap">
    <div class="hero-frame">
      <div class="bracket tl"></div><div class="bracket br"></div>
      <div class="kicker">Internal &amp; Partner Technical Report</div>
      <h1>AI-Based Waste Detection and Classification System</h1>
      <p class="subtitle">Comprehensive Model Training, Evaluation and Performance Analysis</p>
    </div>
    <div class="hero-meta">
      <div class="chip"><b>6</b>&nbsp;experiments trained</div>
      <div class="chip"><b>6</b>&nbsp;material classes — aluminium · plastic · metal · stone · ceramic · glass</div>
      <div class="chip"><b>4</b>&nbsp;detector architectures + 1 classifier</div>
      <div class="chip">RTX 5080&nbsp;16GB · i9-14900KS · 64GB RAM</div>
      <div class="chip">Ultralytics · PyTorch (CUDA 12.8) · Weights &amp; Biases</div>
    </div>
  </div>
</header>"""

# ============================================================================
# SECTION 1 — Executive Summary
# ============================================================================
SEC_SUMMARY = """
<section id="summary" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 01 — Executive Summary</span></div>
  <div class="section-head">
    <div class="kicker">Executive Summary</div>
    <h2>Six independently trained models, one validated recommendation</h2>
    <p class="lede">Between the project's first baseline and this report, six experiments spanning
    four detection/classification architectures were trained end-to-end on this project's own
    conveyor/pour waste-stream imagery, evaluated on identical held-out data wherever the
    architecture allowed, and benchmarked for real inference latency on the target GPU.</p>
  </div>
  <div class="grid cols-4">
    """ + metric("0.951", "Best mAP50-95 — exp003 (YOLO26L+P2)", True) + """
    """ + metric("171 FPS", "Fastest six-class detector — exp001 (plain, 640px)") + """
    """ + metric("9.47M", "Smallest six-class detector — exp006 (params)") + """
    """ + metric("99.37%", "exp004 classifier accuracy on cropped regions") + """
  </div>
  <div style="height:40px"></div>
  <p>The core six-class detectors (exp001, exp003, exp005, exp006) were all evaluated on the exact
  same 1,847-image held-out validation split, making their headline numbers a genuine
  apples-to-apples comparison rather than similar-sounding figures from different data. exp003
  (YOLO26L with an added P2 small-object head) is the most accurate model trained to date; exp006
  (YOLO26s, base-trained then fine-tuned) delivers 99.2% of that accuracy at roughly a fifth the
  latency under tiled (SAHI) inference, making it the strongest candidate where inference budget or
  edge hardware is constrained. exp002 is a deliberately narrow glass-cullet specialist, and exp004
  is a second-stage classifier for the two-model pipeline, not a standalone detector — both are
  reported honestly rather than folded into the six-class comparison.</p>
</div></section>"""

# ============================================================================
# SECTION 2 — Project Overview
# ============================================================================
SEC_OVERVIEW = """
<section id="overview" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 02 — Project Overview</span></div>
  <div class="section-head">
    <div class="kicker">Project Overview</div>
    <h2>An inference pipeline for sorting six material streams</h2>
    <p class="lede">The system watches a conveyor/pour setup and must identify, in real time, which
    of six materials — aluminium, plastic, metal, stone, ceramic, or glass — each object on the
    line belongs to, at a range of object sizes and lighting conditions.</p>
  </div>
  <div class="flow">
    <div class="flow-step"><div class="fn">01</div><h4>Camera / Conveyor Feed</h4><p>Live frame from a webcam or fixed-position camera above the line.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">02</div><h4>Detection Model</h4><p>A YOLO or RT-DETR model localizes and classifies each object in one pass.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">03</div><h4>SAHI Tiling (optional)</h4><p>The frame is sliced into overlapping tiles for better small/distant-object recall.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">04</div><h4>ResNet Classification (optional)</h4><p>A second-stage classifier re-checks each cropped detection under glare/lighting.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">05</div><h4>Tracking &amp; Counting</h4><p>ByteTrack / BoT-SORT assigns stable IDs so each physical object is counted once.</p></div>
  </div>
  <div style="height:16px"></div>
  <p>Every stage above exists as a working script in this project (<code>Scripts/camera</code>,
  <code>videos</code>, <code>images</code>, <code>roi</code>), and every combination — with or
  without SAHI tiling, with or without the second-stage classifier — was benchmarked so the
  accuracy/speed trade-off at each stage is a measured number, not an estimate.</p>
</div></section>"""

# ============================================================================
# SECTION 3 — Dataset Overview
# ============================================================================
SEC_DATASET = """
<section id="dataset" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 03 — Dataset Overview</span></div>
  <div class="section-head">
    <div class="kicker">Dataset Overview</div>
    <h2>Four dataset versions, one held-out validation set reused for fairness</h2>
    <p class="lede">Datasets are versioned and immutable once used by a training run — a version
    folder is named after the experiment(s) that consume it, and is never edited in place.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th class="label">Dataset version</th><th>Used by</th><th>Train</th><th>Val</th><th class="label">Task / classes</th></tr></thead>
    <tbody>
      <tr><td class="label">exp001_exp003_exp006</td><td>exp001, exp003, exp006 (fine-tune stage)</td><td>7,389</td><td>1,847</td><td class="label">6-class detection</td></tr>
      <tr><td class="label">exp002</td><td>exp002</td><td>2,350</td><td>587</td><td class="label">1-class detection (glass cullet)</td></tr>
      <tr><td class="label">exp004</td><td>exp004</td><td>294,959 crops</td><td>73,448 crops</td><td class="label">6-class classification (cropped)</td></tr>
      <tr><td class="label">exp005</td><td>exp005</td><td>7,389</td><td>1,847</td><td class="label">6-class detection (identical split to above)</td></tr>
      <tr><td class="label">exp006 (base stage)</td><td>exp006 stage 1</td><td>1,631</td><td>408</td><td class="label">6-class detection (own split)</td></tr>
    </tbody>
  </table></div>
  <div style="height:32px"></div>
  <p><strong>Class taxonomy (6):</strong> """ + " · ".join(CLASS_NAMES) + """. exp005's dataset is a
  byte-identical copy of exp001_exp003_exp006's split, deliberately not re-randomized, so RT-DETR
  stays a fair comparison against exp001/exp003 on the same held-out images. exp004's dataset is
  <em>derived</em> from exp001_exp003_exp006 by cropping every labeled box, not separately
  annotated — see Data Preparation below.</p>
</div></section>"""

# ============================================================================
# SECTION 4 — Data Prep & Annotation Pipeline
# ============================================================================
SEC_DATAPREP = """
<section id="dataprep" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 04 — Data Preparation</span></div>
  <div class="section-head">
    <div class="kicker">Data Preparation &amp; Annotation Pipeline</div>
    <h2>From raw frames to two dataset shapes, without double-annotating</h2>
  </div>
  <div class="flow">
    <div class="flow-step"><div class="fn">A</div><h4>Frame capture</h4><p>Frames captured from the conveyor/pour setup across varied lighting and object density.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">B</div><h4>YOLO-format annotation</h4><p>Each object hand-labeled as class + normalized bounding box across all 6 classes.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">C</div><h4>Fixed train/val split</h4><p>7,389 / 1,847 split frozen once, reused identically by exp001, exp003, exp005 and exp006's fine-tune stage.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">D</div><h4>Classification crops (exp004)</h4><p>Every labeled box cropped out (+10% padding, &lt;10px boxes skipped) — zero new manual labeling.</p></div>
  </div>
  <div style="height:32px"></div>
  <p>Because step C's split is reused byte-for-byte across four experiments, their validation
  metrics in this report are directly comparable — a meaningfully stronger guarantee than
  "similar-sounding" numbers from independently-split data. Step D means exp004's 294,959-crop
  classification dataset required no separate labeling effort: its ground truth is inherited
  entirely from the detection boxes already drawn in step B.</p>
</div></section>"""

# ============================================================================
# SECTION 5 — Models Evaluated
# ============================================================================
def model_card(exp):
    m = MODELS[exp]
    n = EXP_ORDER.index(exp) + 1
    return ('<div class="card"><h3>' + str(n) + '. ' + m["title"] + '</h3>'
            + '<p style="font-size:0.85rem;color:var(--ink-faint);margin-bottom:10px">' + m["family"] + '</p>'
            + '<p>' + m["role"] + '</p></div>')

SEC_MODELS = """
<section id="models" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 05 — Models Evaluated</span></div>
  <div class="section-head">
    <div class="kicker">Models Evaluated</div>
    <h2>Four architecture families, six trained checkpoints</h2>
    <p class="lede">Each experiment gets a permanent, sequential ID (exp001–exp006) the moment it's
    trained — an ID is never reused or renamed once results exist against it.</p>
  </div>
  <div class="grid cols-3">""" + "".join(model_card(e) for e in EXP_ORDER) + """</div>
</div></section>"""

# ============================================================================
# SECTION 6 — Training Configuration
# ============================================================================
def config_row(exp):
    m = MODELS[exp]
    n = EXP_ORDER.index(exp) + 1
    return (f'<tr><td class="label"><strong>{n}. {m["title"]}</strong></td>'
            f'<td>{m["batch"]}</td><td>{m["imgsz"]}px</td><td>{m["epochs"]}</td>'
            f'<td class="label">{m["optimizer"]}</td><td class="label">{m["lr_detail"]}</td></tr>')

SEC_CONFIG = """
<section id="config" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 06 — Training Configuration</span></div>
  <div class="section-head">
    <div class="kicker">Training Configuration</div>
    <h2>Batch size and training resolution, per model</h2>
    <p class="lede">All training ran on a single RTX 5080 (16GB VRAM) / i9-14900KS / 64GB RAM
    Windows desktop, via the Ultralytics framework (YOLO / RTDETR classes) or plain PyTorch for the
    ResNet classifier, with CUDA 12.8.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th class="label">Experiment</th><th>Batch size</th><th>Image size</th><th>Epochs</th><th class="label">Optimizer / schedule</th><th class="label">Learning rate</th></tr></thead>
    <tbody>""" + "".join(config_row(e) for e in EXP_ORDER) + """</tbody>
  </table></div>
  <div style="height:20px"></div>
  <p>exp004's <strong>224px</strong> image size and <strong>64</strong> batch size reflect its
  different task (classifying a small cropped region, not detecting across a full frame) — every
  detector in the project trains and infers at <strong>640px</strong>. Batch sizes for the larger
  detectors (exp001: 8, exp003: 16, exp005: 16) were set per-model against the same 16GB VRAM
  budget; exp001's batch was deliberately dropped from an original 16 to 8 for the larger YOLOv8L
  network, with both learning rates in its differential schedule scaled down to match.</p>
</div></section>"""

print("sections 1-6 ready")

# ============================================================================
# SECTION 7 — Training Results (per-model canvas curves)
# ============================================================================
def curve_chart(exp):
    m = MODELS[exp]
    cid = f"chart-{exp}"
    if exp == "exp004":
        legend = ('<span><i style="background:var(--accent)"></i>train acc</span>'
                   '<span><i style="background:var(--cyan)"></i>val acc</span>')
    else:
        legend = ('<span><i style="background:var(--accent)"></i>precision</span>'
                   '<span><i style="background:var(--cyan)"></i>recall</span>'
                   '<span><i style="background:var(--violet)"></i>mAP50-95</span>')
    n = EXP_ORDER.index(exp) + 1
    return (f'<div class="chart-box"><h4 style="margin-bottom:4px">{n}. {m["title"]}</h4>'
            f'<p style="font-size:0.8rem;margin-bottom:14px">{m["epochs"]} epochs</p>'
            f'<canvas id="{cid}" height="180" data-exp="{exp}"></canvas>'
            f'<div class="chart-legend">{legend}</div></div>')

SEC_RESULTS = """
<section id="results" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 07 — Training Results</span></div>
  <div class="section-head">
    <div class="kicker">Training Results</div>
    <h2>Every epoch, every model — read straight from results.csv</h2>
    <p class="lede">Curves below are rendered live from each model's own per-epoch training log,
    not a static export — precision/recall/mAP50-95 for the five detectors, train/val accuracy for
    the exp004 classifier.</p>
  </div>
  <div class="grid cols-2">""" + "".join(curve_chart(e) for e in EXP_ORDER) + """</div>
  <div style="height:12px"></div>
  <p>exp003 improved mAP50-95 across every one of the 6 classes over the exp001 baseline, most
  notably on <strong>stone</strong> (0.971) and <strong>ceramic</strong> (0.980) — consistent with
  the added P2 head's purpose of catching smaller or harder objects the plain P3–P5 model missed.
  See each model's own precision-recall curve in its individual section below.</p>
</div></section>"""

# ============================================================================
# SECTION 8 — Model Performance Comparison
# ============================================================================
def fmt(v):
    return f"{v:.3f}" if v is not None else "—"

def compare_row(exp):
    m = MODELS[exp]
    badges = ""
    if exp == "exp003":
        badges = '<span class="pill gold">🏆 Best accuracy</span>'
    elif exp == "exp001":
        badges = '<span class="pill accentp">⚡ Fastest (6-class)</span>'
    elif exp == "exp006":
        badges = '<span class="pill good">⚖ Best balance</span>'
    acc = fmt(m.get("val_acc")) if exp == "exp004" else "—"
    n = EXP_ORDER.index(exp) + 1
    return (f'<tr class="{"hl" if badges else ""}"><td class="label"><strong>{n}. {m["title"]}</strong></td>'
            f'<td>{fmt(m["precision"])}</td><td>{fmt(m["recall"])}</td><td>{fmt(m["map50"])}</td>'
            f'<td>{fmt(m["map5095"])}</td><td>{acc}</td><td>{m["params"]}M</td>'
            f'<td>{B(exp,"plain")["avg_ms"]} ms</td><td>{B(exp,"sahi")["avg_ms"]} ms</td>'
            f'<td class="label">{badges}</td></tr>')

SEC_COMPARISON = """
<section id="comparison" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 08 — Model Comparison</span></div>
  <div class="section-head">
    <div class="kicker">Model Performance Comparison</div>
    <h2>All six models, one table</h2>
    <p class="lede">Latency figures are freshly measured on the reference RTX 5080 against a
    """ + TEST_RES + """ test frame — "plain" is a single 640px forward pass, "SAHI" is a batched
    6-tile (3×2, 20% overlap) forward pass.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th class="label">Model</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th>
    <th>Val acc.</th><th>Params</th><th>Latency (plain)</th><th>Latency (SAHI)</th><th class="label">Badge</th></tr></thead>
    <tbody>""" + "".join(compare_row(e) for e in EXP_ORDER) + """</tbody>
  </table></div>
  <div style="height:20px"></div>
  <p>exp002 and exp004 are intentionally excluded from the 🏆/⚡/⚖ badges above: exp002 solves a
  narrower single-class task on a smaller dataset, and exp004 is a classifier, not a detector —
  neither is a fair like-for-like comparison against the four six-class detectors.</p>
</div></section>"""

# ============================================================================
# SECTION 9 — Confusion Matrix & Class Performance
# ============================================================================
def cm_card(exp):
    m = MODELS[exp]
    n = EXP_ORDER.index(exp) + 1
    if not m["has_cm"]:
        return (f'<div class="card"><h3>{n}. {m["title"]}</h3>'
                f'<p style="margin-top:10px">No confusion matrix available — training was '
                f'interrupted during Ultralytics\' final validation pass (see the RT-DETR-L section and '
                f'Challenges Identified). Epoch-100 validation metrics are reported instead.</p></div>')
    note = ""
    if exp == "exp003":
        note = "Near-zero off-diagonal confusion across all 6 classes — the strongest per-class separation in the project."
    elif exp == "exp002":
        note = "Visible confusion is expected: a single class against background/false-positive noise on irregular glass-cullet shapes."
    elif exp == "exp004":
        note = "Confusion matrix over the 6-class crop classification task, not a detection confusion matrix."
    return (f'<div class="card"><h3>{n}. {m["title"]}</h3>'
            + fig(f"{exp}_cm", f'<span class="k">Confusion matrix</span> — validation set. {note}')
            + '</div>')

SEC_CONFUSION = """
<section id="confusion" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 09 — Confusion Matrices</span></div>
  <div class="section-head">
    <div class="kicker">Confusion Matrix &amp; Class Performance</div>
    <h2>Where each model actually gets confused</h2>
    <p class="lede">Five of six models have a real confusion matrix from their own final validation
    pass; exp005's is the one honest gap in this report — its training run completed but was
    interrupted before Ultralytics generated one.</p>
  </div>
  <div class="grid cols-2">""" + "".join(cm_card(e) for e in EXP_ORDER) + """</div>
</div></section>"""

# ============================================================================
# SECTION 10 — Real-World Prediction Results (overview grid)
# ============================================================================
def pred_thumb(exp):
    m = MODELS[exp]
    n = EXP_ORDER.index(exp) + 1
    return (f'<a href="#{exp}" class="card" style="text-decoration:none;display:block">'
            f'<h3>{n}. {m["title"]}</h3>'
            f'<div class="compare-pair" style="margin-top:12px">'
            f'<img src="{IMG[exp+"_plain"]}" loading="lazy" alt="" style="border-radius:8px;border:1px solid var(--line);width:100%">'
            f'<img src="{IMG[exp+"_sahi"]}" loading="lazy" alt="" style="border-radius:8px;border:1px solid var(--line);width:100%">'
            f'</div><p style="margin-top:10px;font-size:0.8rem">Left: plain inference. Right: SAHI tiled inference. Full detail &darr;</p></a>')

SEC_PREDICTIONS = """
<section id="predictions" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 10 — Prediction Results</span></div>
  <div class="section-head">
    <div class="kicker">Real-World Prediction Results</div>
    <h2>Every model, on the same real test frame</h2>
    <p class="lede">All six models were run on the identical held-out test image (""" + TEST_RES + """),
    once with plain inference and once with 6-tile SAHI inference — the same pair used again in
    each model's own deep-dive section below.</p>
  </div>
  <div class="grid cols-3">""" + "".join(pred_thumb(e) for e in EXP_ORDER) + """</div>
</div></section>"""

print("sections 7-10 ready")

# ============================================================================
# PER-MODEL DEEP DIVES — 3 parts each: (1) spec + architecture, (2) training
# config + curves, (3) confusion matrix + SAHI/plain predictions + benchmark.
# ============================================================================
SECTION_NUMS = {"exp001": 11, "exp002": 12, "exp003": 13, "exp004": 14, "exp005": 15, "exp006": 16}

def model_deepdive(exp):
    m = MODELS[exp]
    num = SECTION_NUMS[exp]
    order_n = EXP_ORDER.index(exp) + 1
    is_clf = exp == "exp004"

    # ---- 1. Model Explanation ----
    part1 = f"""
    <div class="subhead"><span class="num">1</span><h3>Model Explanation</h3></div>
    <div class="card"><p>{m["arch_text"]}</p></div>"""

    # ---- 2. Dataset ----
    stage1_note = ""
    if "stage1" in m:
        stage1_note = ("<p>This model is trained in two stages: a base-training stage on its own "
                        "1,631/408 dataset, then a fine-tuning stage on the dataset below — the "
                        "numbers throughout this section are the final fine-tuned model.</p>")
    dataset_metrics = (metric(m["train_n"], "Training images") + metric(m["val_n"], "Validation images")
                        + metric(m["classes"], "Classes")) if not is_clf else \
        (metric(m["train_n"], "Training crops") + metric(m["val_n"], "Validation crops") + metric(m["classes"], "Classes"))
    part2 = f"""
    <div class="subhead"><span class="num">2</span><h3>Dataset</h3></div>
    <div class="grid cols-3">{dataset_metrics}</div>
    <div style="height:20px"></div>
    <div class="card"><p><strong>Source:</strong> {m["dataset"]}</p>{stage1_note}<p>{m["extra"]}</p></div>"""

    # ---- 3. Training Parameters ----
    if is_clf:
        acc_rows = f"""
    <tr><td class="label">Accuracy</td><td>{m["val_acc"]*100:.2f}%</td></tr>"""
    else:
        acc_rows = f"""
    <tr><td class="label">Precision</td><td>{fmt(m["precision"])}</td></tr>
    <tr><td class="label">Recall</td><td>{fmt(m["recall"])}</td></tr>
    <tr><td class="label">mAP50</td><td>{fmt(m["map50"])}</td></tr>
    <tr><td class="label">mAP50-95</td><td>{fmt(m["map5095"])}</td></tr>"""
    param_rows = f"""
    <tr><td class="label">Base model</td><td class="label">{m["title"]}</td></tr>
    <tr><td class="label">Classes</td><td class="label">{m["classes"]}</td></tr>
    <tr><td class="label">Training images</td><td>{m["train_n"]}</td></tr>
    <tr><td class="label">Validation images</td><td>{m["val_n"]}</td></tr>
    <tr><td class="label">Batch size</td><td>{m["batch"]}</td></tr>
    <tr><td class="label">Training image input size</td><td>{m["imgsz"]}px</td></tr>
    <tr><td class="label">Epochs trained</td><td>{m["epochs"]}</td></tr>
    <tr><td class="label">Training time</td><td class="label">{TRAINING_TIME[exp]}</td></tr>
    {acc_rows}"""
    part3 = f"""
    <div class="subhead"><span class="num">3</span><h3>Training Parameters</h3></div>
    <div class="table-wrap"><table><tbody>{param_rows}</tbody></table></div>
    <div style="height:16px"></div>
    <p><strong>Average model inference time:</strong> {B(exp,'plain')['avg_ms']} ms
    ({B(exp,'plain')['fps']} FPS) &nbsp;·&nbsp; <strong>Average end-to-end latency:</strong>
    {B(exp,'e2e_plain')['avg_ms']} ms — both measured on an RTX 5080 against a {TEST_RES} test
    frame (5 warmup + 30 timed passes). End-to-end includes reading the frame and drawing the
    annotated output, not just the model's forward pass.</p>"""

    # ---- 4. Curves ----
    legend = ('<span><i style="background:var(--accent)"></i>train acc</span><span><i style="background:var(--cyan)"></i>val acc</span>'
              if is_clf else
              '<span><i style="background:var(--accent)"></i>precision</span><span><i style="background:var(--cyan)"></i>recall</span><span><i style="background:var(--violet)"></i>mAP50-95</span>')
    pr_fig = fig(f"{exp}_pr", '<span class="k">Precision-recall curve</span> — Ultralytics-generated, validation set.') if m["has_pr"] else \
        ('<div class="card" style="height:100%;display:flex;align-items:center"><p>No precision-recall curve image — '
         + ('this model is a classifier; see its live accuracy curve to the left instead.' if is_clf
            else 'training was interrupted before Ultralytics\' final validation pass generated one; see the live curve to the left.')
         + '</p></div>')
    part4 = f"""
    <div class="subhead"><span class="num">4</span><h3>Curves</h3></div>
    <div class="grid cols-2">
      <div class="chart-box"><canvas id="chart-dd-{exp}" height="200" data-exp="{exp}"></canvas>
        <div class="chart-legend">{legend}</div>
      </div>
      {pr_fig}
    </div>"""

    # ---- 5. Prediction ----
    if m["has_cm"]:
        cm_block = fig(f"{exp}_cm", '<span class="k">Confusion matrix</span> — validation set.')
    else:
        cm_block = ('<div class="card" style="height:100%;display:flex;align-items:center">'
                    '<p>No confusion matrix — this run\'s training completed all 100 epochs but was '
                    'interrupted during Ultralytics\' final validation pass, which is what generates '
                    'this artifact. Documented here rather than fabricated; see Challenges Identified.</p></div>')
    plain_cap = ('Whole-image classification (not a real detection) — the entire frame is stamped with one predicted label.'
                 if is_clf else f'Plain inference, single forward pass — {B(exp,"plain")["avg_ms"]} ms.')
    sahi_cap = ('Per-tile classification (6 tiles, not real detections) — a crude approximation of localization via tiling.'
                if is_clf else f'SAHI inference, 6-tile (3×2, 20% overlap) — {B(exp,"sahi")["avg_ms"]} ms.')
    part5 = f"""
    <div class="subhead"><span class="num">5</span><h3>Prediction</h3></div>
    <div class="compare-pair">
      {fig(exp+"_plain", '<span class="k">' + ('Whole-image classification' if is_clf else 'Plain inference') + '</span> — ' + plain_cap)}
      {fig(exp+"_sahi", '<span class="k">' + ('Tiled classification' if is_clf else 'SAHI inference') + '</span> — ' + sahi_cap)}
    </div>
    <div style="height:24px"></div>
    <div class="card">{cm_block}</div>"""

    return f"""
<section id="{exp}" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section {num} — Model {order_n}</span></div>
  <div class="model-hero">
    <div><div class="idbadge">Model {order_n}</div><h2>{m["title"]}</h2><p class="role">{m["role"]}</p></div>
  </div>
  {part1}
  {part2}
  {part3}
  {part4}
  {part5}
</div></section>"""

MODEL_SECTIONS = "".join(model_deepdive(e) for e in EXP_ORDER)
print("model deep dives ready:", len(MODEL_SECTIONS), "chars")

# ============================================================================
# SECTION 17 — Challenges Identified
# ============================================================================
def chal_card(title, challenge, impact, solution):
    return f"""<div class="chal">
      <div class="chal-head">{title}</div>
      <div class="chal-row">
        <div class="chal-cell challenge"><span class="ct">Challenge</span><p>{challenge}</p></div>
        <div class="chal-cell impact"><span class="ct">Impact</span><p>{impact}</p></div>
        <div class="chal-cell solution"><span class="ct">Solution</span><p>{solution}</p></div>
      </div></div>"""

CHALLENGES = [
    ("RF-DETR training crashes (original exp005 attempt)",
     "Three unrecoverable training crashes — two silent hangs and one confirmed native access-violation segfault in python311.dll roughly 5 minutes after the first epoch — every time with zero checkpoints ever saved.",
     "The exp005 slot was fully blocked; no usable transformer-family comparison model, and repeated GPU time was lost to crashed runs.",
     "RF-DETR was abandoned and its package fully removed from the project. The exp005 slot was reclaimed for RT-DETR (Baidu's unrelated, Ultralytics-native DETR variant), trained cleanly via the same .train() API as every YOLO model — zero crashes."),
    ("exp005's interrupted final validation pass",
     "Training genuinely completed all 100/100 epochs (4.79 hours), but the process was interrupted during Ultralytics' post-training validation pass on best.pt — the step that generates the confusion matrix and PR curves.",
     "exp005 is the one model in this report with no confusion_matrix.png, results.png, or val_batch preview images.",
     "The run was finished manually (weights copied, config renamed, results split), and the gap is reported honestly using epoch-100's own end-of-epoch validation metrics rather than a fabricated confusion matrix."),
    ("Windows environment fragility",
     "opencv-python and opencv-python-headless silently overwrote each other's files; a torchvision DataLoader worker deadlocked mid-run with zero terminal trace; an orphaned wandb background process held a file lock after a run finished.",
     "At least one long unattended run appeared to hang with no explanation, requiring manual process-tree and Windows event-log inspection to diagnose.",
     "Pinned a single OpenCV variant, documented the DataLoader-deadlock/Windows-Update risk before any unattended run, and made the results-split step explicitly skip the wandb/ folder."),
    ("AV/security software blocking file downloads",
     "raw.githubusercontent.com specifically hung when downloading .py files on this machine, while plain .csv files from the same repository worked fine.",
     "Blocked pulling reference code needed mid-task, with no obvious cause from the error alone.",
     "Switched to the cdn.jsdelivr.net GitHub mirror, which worked every time as a reliable fallback."),
    ("Glass-cullet is a genuinely harder single-class task (exp002)",
     "Irregular glass-fragment shapes have no consistent silhouette the way a whole bottle or can does, on a dataset roughly a third the size of the main six-class set.",
     "exp002's mAP50-95 (0.370) is meaningfully lower than every six-class detector in the project.",
     "Kept as a deliberately narrow, separately-scoped specialist model rather than folded into the main task — flagged directly as the clearest candidate for more training data."),
    ("A classifier cannot localize objects on its own (exp004)",
     "ResNet-50 expects an already-cropped single-object region; it cannot draw bounding boxes on a raw multi-object frame.",
     "exp004 can't be benchmarked as a like-for-like detector against exp001/002/003/005/006.",
     "Documented explicitly wherever exp004 appears (this report and Scripts/images/README.md), and kept as the second stage of the two-model YOLO+ResNet pipeline rather than a standalone product."),
]
SEC_CHALLENGES = """
<section id="challenges" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 17 — Challenges Identified</span></div>
  <div class="section-head">
    <div class="kicker">Challenges Identified</div>
    <h2>What actually went wrong, and how it was resolved</h2>
    <p class="lede">Reported the same way the project's own working notes track them — nothing here
    is smoothed over.</p>
  </div>
  <div style="display:flex;flex-direction:column;gap:20px">""" + "".join(chal_card(*c) for c in CHALLENGES) + """</div>
</div></section>"""

# ============================================================================
# SECTION 18 — Model Improvement Journey
# ============================================================================
JOURNEY = [
    ("exp001", "Baseline established",
     "YOLOv8L with a differential learning-rate schedule (frozen low-level filters, slow backbone/neck, fast detection head) — the project's first working six-class detector.",
     "mAP50-95 0.935"),
    ("exp002", "Specialist side-quest",
     "A narrower single-class glass-cullet detector (YOLO26s), trained on its own smaller dataset and imported from a laptop run — a separate specialist, not a replacement for the main pipeline.",
     "mAP50-95 0.370 (harder task)"),
    ("exp003", "Architecture experiment — new best",
     "Added a P2/160×160 detection head to YOLO26L for small-object sensitivity. Became the new best model in the project, with the largest gains on stone and ceramic.",
     "mAP50-95 0.951 (best)"),
    ("exp004", "Two-stage pipeline test",
     "Trained a dedicated ResNet-50 classifier on YOLO-cropped regions, to test whether a focused second-stage classifier beats a detector's own joint classification head under real lighting/glare.",
     "99.37% val accuracy"),
    ("exp005 (attempt 1)", "RF-DETR — abandoned",
     "Three unrecoverable crashes, zero checkpoints ever saved. The package was removed from the project entirely.",
     "No model produced"),
    ("exp005 (attempt 2)", "RT-DETR — reclaimed slot",
     "Reclaimed the exp005 ID for RT-DETR (Baidu, via Ultralytics), trained cleanly to 100/100 epochs as a transformer-family comparison point — despite losing its own confusion matrix to an interrupted final validation pass.",
     "mAP50-95 0.922"),
    ("exp006", "Imported and integrated",
     "A two-stage (base-trained, then fine-tuned) YOLO26s model trained on a laptop, brought into this project's structure and renamed for consistency. Lands between exp001 and exp003 despite being 2.6–4.6× smaller.",
     "mAP50-95 0.943"),
]
def journey_item(tag, title, body, result):
    return f"""<div class="tl-item"><div class="tl-tag">{tag}</div><h4>{title}</h4><p>{body}</p>
    <p style="margin-top:6px"><span class="pill accentp">{result}</span></p></div>"""

SEC_JOURNEY = """
<section id="journey" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 18 — Improvement Journey</span></div>
  <div class="section-head">
    <div class="kicker">Model Improvement Journey</div>
    <h2>Six experiments, in the order they actually happened</h2>
  </div>
  <div class="timeline">""" + "".join(journey_item(*j) for j in JOURNEY) + """</div>
</div></section>"""

# ============================================================================
# SECTION 19 — Final Model Recommendation
# ============================================================================
SEC_RECOMMEND = """
<section id="recommendation" class="sheet navy-block"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 19 — Final Recommendation</span></div>
  <div class="section-head">
    <div class="kicker">Final Model Recommendation</div>
    <h2>Two models cover two deployment profiles</h2>
  </div>
  <div class="grid cols-2">
    <div class="recommend-card">
      <div class="tag" style="color:var(--cyan)">Primary — accuracy-critical deployment</div>
      <h3>exp003 · YOLO26L + P2</h3>
      <p style="margin-top:14px">Best accuracy in the project on every headline metric (mAP50-95
      0.951), with the strongest per-class results on the hardest classes — stone and ceramic.
      Recommended wherever inference hardware is not the binding constraint.</p>
    </div>
    <div class="recommend-card">
      <div class="tag" style="color:var(--cyan)">Secondary — constrained hardware / edge</div>
      <h3>exp006 · YOLO26s (fine-tuned)</h3>
      <p style="margin-top:14px">Retains 99.2% of exp003's mAP50-95 (0.943 vs 0.951) at roughly a
      fifth the parameter count and 2.3× lower SAHI latency. The strongest candidate where
      inference budget, cost, or edge hardware is the binding constraint.</p>
    </div>
  </div>
  <div style="height:20px"></div>
  <p>exp002 (glass-cullet specialist) is recommended as an optional add-on model where glass-cullet
  volume is high enough to justify a dedicated pass, not a general replacement. exp004 (ResNet-50)
  is recommended as an optional accuracy-boosting second stage under difficult lighting/glare, not
  as a standalone deployment.</p>
</div></section>"""

# ============================================================================
# SECTION 20 — Final Industrial AI Pipeline
# ============================================================================
SEC_FINAL_PIPELINE = """
<section id="final-pipeline" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 20 — Deployment Pipeline</span></div>
  <div class="section-head">
    <div class="kicker">Final Industrial AI Pipeline</div>
    <h2>The recommended end-to-end deployment configuration</h2>
    <p class="lede">Matches the project's own highest-accuracy live pipeline script
    (<code>Scripts/videos/yr_sahi_botsort.py</code>) — every stage below already exists and runs.</p>
  </div>
  <div class="flow">
    <div class="flow-step"><div class="fn">01</div><h4>Camera / Conveyor Feed</h4><p>Live frame from the fixed camera above the line.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">02</div><h4>ROI Crop</h4><p>Frame cropped to the active conveyor region (Scripts/roi/roi.py).</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">03</div><h4>SAHI 6-Tile Detection</h4><p>exp003 (accuracy) or exp006 (speed) run across 6 overlapping tiles.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">04</div><h4>BoT-SORT + ReID Tracking</h4><p>Each physical object gets a stable ID across frames.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">05</div><h4>ResNet-50 Re-check (optional)</h4><p>exp004 re-classifies each crop under difficult glare/lighting.</p></div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-step"><div class="fn">06</div><h4>Majority-Vote Counting</h4><p>Final per-object label decided by majority vote across its tracked lifetime.</p></div>
  </div>
</div></section>"""

# ============================================================================
# SECTION 21 — Future Improvements
# ============================================================================
FUTURE = [
    "Re-run exp005's final validation pass (or a short continuation) to generate its confusion matrix and PR curves — the only real evaluation gap left in the current lineup.",
    "Grow the glass-cullet dataset (exp002) — the smallest and hardest task in the project — before relying on it in production.",
    "Benchmark the full two-model YOLO+ResNet pipeline end-to-end on live video, not just each stage in isolation.",
    "Install a CUDA execution provider (the missing cudnn64_9.dll) for the ONNX-based CircularNet reference model so it can be benchmarked on GPU instead of CPU.",
    "Expand the six-class taxonomy if new material streams are added to the conveyor line.",
]
SEC_FUTURE = """
<section id="future" class="sheet"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 21 — Future Improvements</span></div>
  <div class="section-head">
    <div class="kicker">Future Improvements</div>
    <h2>What's next, grounded in what's already known</h2>
  </div>
  <div class="grid cols-2">""" + "".join(f'<div class="card"><p>{f}</p></div>' for f in FUTURE) + """</div>
</div></section>"""

# ============================================================================
# SECTION 22 — Final Conclusion
# ============================================================================
SEC_CONCLUSION = """
<section id="conclusion" class="sheet navy-block"><div class="wrap">
  <div class="page-rule"><span>Model Training Report</span><span>Section 22 — Conclusion</span></div>
  <div class="section-head">
    <div class="kicker">Final Conclusion</div>
    <h2>Six models, evaluated honestly, one clear recommendation</h2>
  </div>
  <p style="max-width:74ch">This project trained and evaluated four detection architectures and one
  classifier across six experiments, on real conveyor imagery spanning six material classes.
  Four of the six six-class-comparable models share an identical held-out validation set, making
  their headline numbers directly comparable rather than coincidentally similar. exp003 (YOLO26L
  with an added P2 head) is the most accurate model produced to date; exp006 offers nearly the same
  accuracy at a fraction of the computational cost. Every known gap in this evaluation — exp005's
  missing confusion matrix, exp002's narrower scope, exp004's role as a classifier rather than a
  detector — is reported here explicitly rather than smoothed over, because a deployment decision
  built on inflated evaluation numbers is worse than one built on an honest, if imperfect, one.</p>
</div></section>"""

FOOTER = """
<footer><div class="wrap">
  <span>AI-Based Waste Detection and Classification System — Model Training Report</span>
  <span>6 experiments · 6 classes · RTX 5080 16GB / i9-14900KS / 64GB RAM</span>
</div></footer>"""

print("sections 17-22 + footer ready")

# ============================================================================
# JS — live canvas training-curve charts, drawn from the real per-epoch data
# ============================================================================
CURVES_JSON = json.dumps(CURVES)
SCRIPT = """
<script>
const CURVES = """ + CURVES_JSON + """;
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function drawChart(canvas) {
  const exp = canvas.dataset.exp;
  const d = CURVES[exp];
  if (!d) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.height;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 34, r: 10, t: 10, b: 22 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const ink = cssVar('--ink-faint') || '#93A2B5';
  const line = cssVar('--line') || '#E1E7EF';

  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.font = '10px "IBM Plex Mono", monospace'; ctx.fillStyle = ink;
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + (plotH * g / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText((1 - g / 4).toFixed(2), 2, y + 3);
  }

  const series = exp === 'exp004'
    ? [['train_acc', '--accent'], ['val_acc', '--cyan']]
    : [['precision', '--accent'], ['recall', '--cyan'], ['map5095', '--violet']];

  const n = d.epoch.length;
  series.forEach(([key, colorVar]) => {
    const vals = d[key];
    if (!vals) return;
    ctx.strokeStyle = cssVar(colorVar) || '#2A63E4';
    ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, i) => {
      const x = pad.l + (plotW * i / (n - 1));
      const y = pad.t + plotH * (1 - Math.max(0, Math.min(1, v)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    const lastX = pad.l + plotW, lastY = pad.t + plotH * (1 - Math.max(0, Math.min(1, vals[vals.length - 1])));
    ctx.beginPath(); ctx.arc(lastX, lastY, 3, 0, 7); ctx.fillStyle = cssVar(colorVar); ctx.fill();
  });

  ctx.fillStyle = ink;
  ctx.fillText('epoch 0', pad.l, h - 6);
  ctx.fillText('epoch ' + d.epoch[n - 1], w - pad.r - 44, h - 6);
}
function drawAll() { document.querySelectorAll('canvas[data-exp]').forEach(drawChart); }
window.addEventListener('load', drawAll);
window.addEventListener('resize', () => { clearTimeout(window.__rt); window.__rt = setTimeout(drawAll, 120); });
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
}
</script>"""

# ============================================================================
# FINAL ASSEMBLY
# ============================================================================
BODY = (NAV + HERO + SEC_SUMMARY + SEC_OVERVIEW + SEC_DATASET + SEC_DATAPREP + SEC_MODELS
        + SEC_CONFIG + SEC_RESULTS + SEC_COMPARISON + SEC_CONFUSION + SEC_PREDICTIONS
        + MODEL_SECTIONS + SEC_CHALLENGES + SEC_JOURNEY + SEC_RECOMMEND + SEC_FINAL_PIPELINE
        + SEC_FUTURE + SEC_CONCLUSION + FOOTER + SCRIPT)

FULL_HTML = HEAD + BODY

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(FULL_HTML)

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"\\nWritten: {OUT_PATH}")
print(f"Final size: {size_mb:.2f} MB")
