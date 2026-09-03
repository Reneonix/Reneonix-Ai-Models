# notebook/build

Regenerates `../model_training_report.html` from this project's own real data - run in this order
whenever a new experiment is added:

```
python benchmark_all.py     # re-measures inference latency for every model, writes benchmark_all_result.json
python prep_images.py       # downscales/recompresses results/predicted_images/* into report_assets/ (JPEG, ~1360px wide)
python build_report.py      # assembles the full HTML report, writes ../model_training_report.html
python build_docx.py        # assembles the plain Google-Docs-style .docx, writes ../Model_Training_Report.docx
python build_pdf.py         # assembles the same content directly as a PDF, writes ../Model_Training_Report.pdf
```

- **`benchmark_all.py`** loads every model in `experiments/expNNN_.../weights/best.pt` and times it
  against `results/testing_images/testing.png.png` (5 warmup + 30 timed iterations, both plain and
  6-tile SAHI inference) - the same real benchmark numbers quoted throughout the report.
- **`prep_images.py`** shrinks the ~8.5MB annotated prediction images in
  `results/predicted_images/<expNNN>/` down to ~150-220KB JPEGs so the whole report can embed them
  as inline `data:` URIs and stay well under the 16MB artifact size limit; confusion matrices are
  copied as-is (already small).
- **`curves.json`** is a one-off export of every model's full per-epoch `results.csv` (precision /
  recall / mAP50-95, or train/val accuracy for exp004) - re-export it if you want the live
  in-browser training-curve charts to include a newly trained experiment:
  ```python
  import csv, json
  # see build_report.py's own CURVES loading for the exact column names per model type
  ```
- **`build_report.py`** is the HTML page generator - a `MODELS` dict holds every real fact quoted
  in the report (dataset counts, batch/imgsz/epochs, final metrics, architecture description).
  **To add a new experiment**: add its entry to `MODELS`, add its ID to `EXP_ORDER`/`SECTION_NUMS`,
  and re-run `benchmark_all.py` → `prep_images.py` → `build_report.py` in order.
- **`build_docx.py`** generates `../Model_Training_Report.docx` - a plain, Times New Roman,
  bordered-table version with yellow-highlighted key figures, built to import cleanly into Google
  Docs (the HTML's live canvas charts/dark theme don't survive that trip). It generates its own
  matplotlib training-curve PNGs for exp004 and exp005 (the two models with no Ultralytics PR-curve
  image) straight from `curves.json`, and reuses the same `report_assets/` JPEGs/PNGs as the HTML
  report. Has its own `MODELS` list (duplicated, not imported, so this file works standalone) -
  update both `build_report.py`'s `MODELS` dict and this one when adding a new experiment.
- **`build_pdf.py`** renders the identical content straight to PDF with reportlab (Times-Roman
  base font, no external font files, no Word/LibreOffice dependency) - it does not read the
  `.docx`, it has its own copy of the same `MODELS` list, so update all three files' `MODELS`
  together when adding a new experiment.

Nothing in here is checked against a specific Python environment beyond what the rest of this
project already requires (`ultralytics`, `torch`, `opencv-python`, `torchvision`) - it reuses the
same environment as `Scripts/images/*.py`.
