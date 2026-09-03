"""Downscale/recompress the huge (~8.5MB) prediction images to something embeddable inline as
base64 in the report HTML. Confusion matrices are already small (<210KB) and copied as-is."""
import cv2
import os

ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
OUT = f"{ROOT}/notebook/build/report_assets"
os.makedirs(OUT, exist_ok=True)

TARGET_W = 1360
JPEG_Q = 80

exps = ["exp001", "exp002", "exp003", "exp004", "exp005", "exp006"]
for exp in exps:
    for kind in ["plain", "sahi"]:
        src = f"{ROOT}/results/predicted_images/{exp}/testing.png_{kind}.png"
        if not os.path.exists(src):
            print("MISSING", src)
            continue
        img = cv2.imread(src)
        h, w = img.shape[:2]
        scale = TARGET_W / w
        resized = cv2.resize(img, (TARGET_W, int(h * scale)), interpolation=cv2.INTER_AREA)
        dst = f"{OUT}/{exp}_{kind}.jpg"
        cv2.imwrite(dst, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        print(exp, kind, os.path.getsize(dst) / 1024, "KB")

# confusion matrices - copy as-is (already small), except normalize name
import shutil
cm_map = {
    "exp001": f"{ROOT}/results/exp001/confusion_matrix.png",
    "exp002": f"{ROOT}/results/exp002/confusion_matrix.png",
    "exp003": f"{ROOT}/results/exp003/confusion_matrix.png",
    "exp004": f"{ROOT}/results/exp004/confusion_matrix.png",
    "exp006": f"{ROOT}/results/exp006/confusion_matrix.png",
}
for exp, src in cm_map.items():
    dst = f"{OUT}/{exp}_cm.png"
    shutil.copy(src, dst)
    print(exp, "cm", os.path.getsize(dst) / 1024, "KB")

print("done ->", OUT)
