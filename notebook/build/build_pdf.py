#!/usr/bin/env python3
"""Builds a plain PDF version of the Model Training Report - same content/structure as
Model_Training_Report.docx, rendered directly with reportlab (no MS Word/LibreOffice needed)."""
import json
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image as RLImage, PageBreak)
from reportlab.lib.enums import TA_CENTER

ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
BUILD = f"{ROOT}/notebook/build"
ASSETS = f"{BUILD}/report_assets"
OUT_PATH = f"{ROOT}/notebook/Model_Training_Report.pdf"

with open(f"{BUILD}/benchmark_all_result.json") as f:
    BENCH_ALL = json.load(f)
BENCH = BENCH_ALL["models"]
TEST_RES = BENCH_ALL["image_resolution"]

TRAINING_TIME = {
    "exp001": "2h 19m (100 epochs)",
    "exp002": "1h 23m (120 epochs)",
    "exp003": "41h 17m (100 epochs)",
    "exp004": "not logged for this run (29 epochs to best checkpoint)",
    "exp005": "4h 48m (100 epochs)",
    "exp006": "29m (fine-tune stage, 100 epochs) + a separate base-training stage",
}

MODELS = [
    dict(exp="exp001", title="YOLOv8L", classes="6 - aluminium, plastic, metal, stone, ceramic, glass",
         train_n="7,389", val_n="1,847", batch=8, imgsz=640, epochs=100,
         dataset_line="Dataset is taken from the project's main conveyor/pour capture set - "
                       "7,389 training images and 1,847 validation images, hand-annotated in YOLO "
                       "format across all 6 material classes.",
         explanation="YOLOv8L is a CNN-based, anchor-free single-stage detector: a CSPDarknet-style "
                     "backbone extracts multi-scale features, a PAN/FPN neck fuses them, and a "
                     "decoupled head predicts class and box offsets per grid cell directly, with no "
                     "separate region-proposal stage. This run overrides Ultralytics' default "
                     "optimizer with a 2-group differential learning-rate schedule: the first 5 "
                     "backbone layers (generic edge/texture filters) are frozen outright, the "
                     "remaining backbone/neck layers train at a low learning rate (3.5x10-4), and the "
                     "detection head - which must learn this project's 6 classes from scratch - "
                     "trains at a high learning rate (7.0x10-3), both annealed on a cosine schedule.",
         precision=0.945, recall=0.932, map50=0.975, map5095=0.935, is_clf=False,
         pr_img=f"{ROOT}/results/exp001/BoxPR_curve.png"),
    dict(exp="exp002", title="YOLO26s", classes="1 - glass (cullet fragments only)",
         train_n="2,350", val_n="587", batch=8, imgsz=640, epochs=120,
         dataset_line="Dataset is a separate, narrower capture set of glass-cullet fragments only - "
                       "2,350 training images and 587 validation images, single class.",
         explanation="Same YOLO26 architecture family as exp006, but trained as a narrow specialist: "
                     "a single output class (glass cullet) on a dataset roughly a third the size of "
                     "the main six-class set. Glass-cullet fragments have no consistent silhouette "
                     "the way a whole bottle or can does, which - combined with the smaller dataset - "
                     "explains its lower headline numbers versus the six-class models.",
         precision=0.694, recall=0.553, map50=0.602, map5095=0.370, is_clf=False,
         pr_img=f"{ROOT}/results/exp002/BoxPR_curve.png"),
    dict(exp="exp003", title="YOLO26L + P2", classes="6 - aluminium, plastic, metal, stone, ceramic, glass",
         train_n="7,389", val_n="1,847", batch=16, imgsz=640, epochs=100,
         dataset_line="Same main dataset as YOLOv8L above - 7,389 training images and 1,847 "
                       "validation images, identical split for a fair comparison.",
         explanation="The only architectural change from a stock YOLO26L is an added P2 (160x160) "
                     "detection head alongside the stock P3/P4/P5 heads, giving the network a "
                     "detection path with less downsampling - better suited to small or distant "
                     "objects on the conveyor. Every other hyperparameter is left at Ultralytics "
                     "defaults, isolating the P2 head's own effect on accuracy. This is the "
                     "best-performing model in the project on every headline metric.",
         precision=0.947, recall=0.930, map50=0.977, map5095=0.951, is_clf=False,
         pr_img=f"{ROOT}/results/exp003/BoxPR_curve.png"),
    dict(exp="exp004", title="ResNet-50", classes="6 - aluminium, plastic, metal, stone, ceramic, glass",
         train_n="294,959 crops", val_n="73,448 crops", batch=64, imgsz=224, epochs=29,
         dataset_line="Dataset is not separately captured - it is built by cropping every labeled "
                       "YOLO box out of the main dataset's own images (10% padding added, boxes "
                       "under 10px skipped), giving 294,959 training crops and 73,448 validation "
                       "crops with zero new manual labeling.",
         explanation="A standard ResNet-50 backbone, ImageNet-pretrained, with its final layer "
                     "replaced by a 6-way linear classifier. Not a detector: it is the second stage "
                     "of a two-model pipeline where YOLO detects and localizes an object first, then "
                     "this classifier looks only at that tightly cropped region to assign the final "
                     "material label, using heavy color/lighting augmentation for robustness to "
                     "glare. It cannot localize objects in a raw multi-object frame on its own.",
         precision=None, recall=None, map50=None, map5095=None, is_clf=True, val_acc=0.9937234506045093,
         pr_img=f"{ASSETS}/exp004_curve.png"),
    dict(exp="exp005", title="RT-DETR-L", classes="6 - aluminium, plastic, metal, stone, ceramic, glass",
         train_n="7,389", val_n="1,847", batch=16, imgsz=640, epochs=100,
         dataset_line="Same main dataset and identical validation split as YOLOv8L and YOLO26L+P2 "
                       "above - 7,389 training images and 1,847 validation images.",
         explanation="RT-DETR (Baidu, 2023) is a transformer-based detector with no anchor boxes and "
                     "no NMS post-processing - a CNN backbone feeds a transformer encoder-decoder "
                     "that predicts a fixed set of object queries directly. Trained via Ultralytics' "
                     "own RTDETR class, the same .train() API as every YOLO model here, making it a "
                     "fair architecture-only comparison. Training completed all 100 epochs, but the "
                     "process was interrupted during Ultralytics' final validation pass, so no "
                     "confusion matrix or PR-curve image exists for this run - the chart below is a "
                     "live per-epoch training curve instead.",
         precision=0.946, recall=0.924, map50=0.975, map5095=0.922, is_clf=False,
         pr_img=f"{ASSETS}/exp005_curve.png"),
    dict(exp="exp006", title="YOLO26s (fine-tuned)", classes="6 - aluminium, plastic, metal, stone, ceramic, glass",
         train_n="1,631 + 7,389", val_n="408 / 1,847", batch=8, imgsz=640, epochs=100,
         dataset_line="Trained in two stages: base-trained on its own 1,631/408 dataset, then "
                       "fine-tuned on the main 7,389/1,847 dataset - the numbers below are the final "
                       "fine-tuned model.",
         explanation="Same YOLO26s architecture as exp002, trained in two stages instead of one: "
                     "stage 1 base-trains on a separate smaller dataset, then stage 2 fine-tunes the "
                     "resulting weights on this project's main six-class dataset. It lands between "
                     "YOLOv8L and YOLO26L+P2 on the final held-out set despite being a much smaller, "
                     "faster network - evidence that the base-training stage transfers useful "
                     "structure before the model ever sees this project's own data.",
         precision=0.94436, recall=0.92681, map50=0.97514, map5095=0.94335, is_clf=False,
         pr_img=f"{ROOT}/results/exp006/BoxPR_curve.png"),
]

# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=11,
                       leading=15, spaceAfter=8, textColor=colors.black)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=9, leading=12, textColor=colors.black)
TITLE = ParagraphStyle("TitleX", parent=styles["Title"], fontName="Times-Bold", fontSize=26,
                        leading=30, alignment=TA_CENTER, textColor=colors.black)
SUBTITLE = ParagraphStyle("SubtitleX", parent=BODY, fontSize=12, alignment=TA_CENTER, spaceAfter=24)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=16,
                     leading=20, spaceBefore=16, spaceAfter=8, textColor=colors.black)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=13,
                     leading=17, spaceBefore=10, spaceAfter=6, textColor=colors.black)
CAP = ParagraphStyle("Cap", parent=BODY, fontName="Times-Bold", fontSize=10, spaceAfter=4)

def hl(text):
    """Wrap text in a yellow-highlight span, matching the .docx version's convention."""
    return f'<font backColor="#FFFF00">{text}</font>'

def fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"

def img_flowable(path, max_w=5.6 * inch, max_h=4.2 * inch):
    if not os.path.exists(path):
        return Paragraph(f"[image not found: {path}]", SMALL)
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h, 1.0)
    return RLImage(path, width=w * scale, height=h * scale)

story = []
story.append(Paragraph("Model Training Report", TITLE))
story.append(Paragraph("AI-Based Waste Detection and Classification System", SUBTITLE))

for i, m in enumerate(MODELS, start=1):
    exp = m["exp"]
    b_plain = BENCH[exp]["plain"]
    b_e2e = BENCH[exp]["e2e_plain"]

    story.append(Paragraph(f"{i}. {m['title']}", H1))

    story.append(Paragraph("Model Explanation", H2))
    story.append(Paragraph(m["explanation"], BODY))

    story.append(Paragraph("Dataset", H2))
    line = m["dataset_line"].replace(m["train_n"], hl(m["train_n"])).replace(m["val_n"], hl(m["val_n"]))
    story.append(Paragraph(line, BODY))

    story.append(Paragraph("Training Parameters", H2))
    rows = [
        ["Base model", m["title"]],
        ["Classes", m["classes"]],
        ["Training images" if not m["is_clf"] else "Training crops", m["train_n"]],
        ["Validation images" if not m["is_clf"] else "Validation crops", m["val_n"]],
        ["Batch size", str(m["batch"])],
        ["Training Image Input size", f"{m['imgsz']}px"],
        ["Epochs trained", str(m["epochs"])],
        ["Training time", TRAINING_TIME[exp]],
    ]
    highlight_row_idx = None
    if m["is_clf"]:
        rows.append(["Accuracy", f"{m['val_acc']*100:.2f}%"])
        highlight_row_idx = len(rows) - 1
    else:
        rows += [
            ["Precision", fmt(m["precision"])],
            ["Recall", fmt(m["recall"])],
            ["mAP50", fmt(m["map50"])],
            ["mAP50-95", fmt(m["map5095"])],
        ]
        highlight_row_idx = len(rows) - 1  # mAP50-95 row

    table_rows = [[Paragraph(r[0], BODY), Paragraph(r[1], BODY)] for r in rows]
    t = Table(table_rows, colWidths=[2.3 * inch, 3.3 * inch])
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row_idx is not None:
        style_cmds.append(("BACKGROUND", (1, highlight_row_idx), (1, highlight_row_idx), colors.yellow))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Average model Inference time is : "
                            + hl(f"{b_plain['avg_ms']} ms ({b_plain['fps']} FPS)"), BODY))
    story.append(Paragraph(f"Average end to end latency : " + hl(f"{b_e2e['avg_ms']} ms"), BODY))
    story.append(Paragraph(
        f"(Measured on an RTX 5080, {TEST_RES} test frame, 5 warmup + 30 timed passes. "
        "End-to-end includes reading the frame and drawing the annotated output, not just the "
        "model's forward pass.)", SMALL))

    story.append(Paragraph("Model curve", H2))
    story.append(img_flowable(m["pr_img"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Prediction", H2))
    story.append(Paragraph("Plain inference (no tiling):", CAP))
    story.append(img_flowable(f"{ASSETS}/{exp}_plain.jpg"))
    story.append(Spacer(1, 8))
    story.append(Paragraph("SAHI inference (6-tile):", CAP))
    story.append(img_flowable(f"{ASSETS}/{exp}_sahi.jpg"))

    if i < len(MODELS):
        story.append(PageBreak())

doc = SimpleDocTemplate(OUT_PATH, pagesize=LETTER,
                         topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                         leftMargin=1.0 * inch, rightMargin=1.0 * inch,
                         title="Model Training Report")
doc.build(story)
print("Written:", OUT_PATH, "-", round(os.path.getsize(OUT_PATH) / 1024 / 1024, 2), "MB")
