from ultralytics import YOLOWorld
import torch

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS isn't available in this Python environment")

device = "mps"

DATA = "yolo_dataset/dataset.yaml"
WEIGHTS = "yolov8s-worldv2.pt"

# Baseline: pretrained weights, before fine-tuning
baseline = YOLOWorld(WEIGHTS)

before = baseline.val(
    data=DATA,
    split="val",
    imgsz=640,
    project="runs/detect",
    name="fpds_before",
    device=device
)
print("Before:", before.results_dict)


# Fine-tune a fresh copy of the same pretrained weights
model = YOLOWorld(WEIGHTS)
model.train(
    data=DATA,
    epochs=50,
    imgsz=640,
    project="runs/detect",
    name="fpds_finetune",
    device=device
)

# Evaluate the best checkpoint saved during training
best = YOLOWorld("runs/detect/runs/detect/fpds_finetune/weights/best.pt")
after = best.val(
    data=DATA,
    split="val",
    imgsz=640,
    project="runs/detect",
    name="fpds_after",
    device=device
)
print("After:", after.results_dict)

# Once you have picked a model using validation, evaluate it on the held-out test split
test = best.val(data=DATA, split="test", imgsz=640, name="fpds_test", device=device)
print("Test:", test.results_dict)