from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("forest.jpg", save=True)
