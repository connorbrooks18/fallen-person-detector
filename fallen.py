from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes

model.set_classes(["person that fell onto ground / person down", "person not fallen over"])

# model.train()

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict("/Users/connor/c/projects/intellisee/split10_320.png")

# Show results
results[0].show()