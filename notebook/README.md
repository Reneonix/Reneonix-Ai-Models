# notebook

Executive-facing writeup of everything trained in this project so far.

```
notebook/
├── model_training_report.html   Executive/CEO-facing report - open directly in a browser
├── Model_Training_Report.docx   Plain Google-Docs-style version - see below
├── Model_Training_Report.pdf    Same content as the .docx, rendered directly as a PDF
└── build/                       regenerates all three of the above - see build/README.md
```

**`Model_Training_Report.docx`** and **`Model_Training_Report.pdf`** are both plain,
black-and-white, Times New Roman versions (bordered tables, yellow-highlighted key figures) - the
interactive HTML's live charts and dark theming don't survive a trip into Google Docs, so these
are separate, simpler documents with the same real numbers. The PDF is generated directly with
reportlab (not by "printing" the .docx), so it doesn't depend on Word or LibreOffice being
installed.

**Is the `.docx` supported by Google Docs?** Yes - it's a standard Office Open XML `.docx`
(python-docx output), and Google Docs imports `.docx` natively. Upload it to Drive under
naveenkumar@reneonix.com, then right-click → Open with → Google Docs (or File → Open) - headings,
the bordered tables, yellow highlights, and embedded images all carry over; only the exact font
metrics may render very slightly differently since Google substitutes its own Times New Roman.
There is no Google Drive/Docs connector available in *this* session to upload it directly - see
the connector note further down / ask in a fresh session once one is connected.

**`model_training_report.html`** covers all 6 experiments (exp001–exp006): dataset train/val
counts, batch size + training image size, precision/recall/mAP curves (live-rendered from each
model's own `results.csv`), confusion matrix, a plain-vs-SAHI prediction image pair, and
freshly-measured inference latency (RTX 5080) for every model, plus a project-wide comparison
table, challenges/journey/recommendation narrative, and the recommended deployment pipeline.

Every number and image in it traces to this project's own `results/`, `data/versions/*/README.md`,
and `experiments/*/config.json` / `src/*.py` — nothing is estimated or invented. Two honest gaps
are called out rather than hidden: exp005 has no confusion matrix (training completed but the
process was interrupted during Ultralytics' final validation pass), and exp004 is a classifier,
not a real detector, so its "predictions" are whole-image/per-tile classification, not detections.

Also published as a Claude Artifact for sharing:
https://claude.ai/code/artifact/da06eb43-5688-448b-8b82-0a888c6a65fc

Regenerate it any time new experiments are added — see `build/README.md`.
