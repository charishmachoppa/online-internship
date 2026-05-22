from ultralytics import YOLO
import cv2
import os

# Load YOLO detection model
model = YOLO("yolov8n.pt")

# Input folder
input_folder = "train/images"

# Output folder
output_folder = "detected_output"

os.makedirs(output_folder, exist_ok=True)

# Process images
for image_name in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image_name)

    # Run object detection
    results = model(image_path)

    # Draw bounding boxes
    detected_frame = results[0].plot()

    # Save output
    output_path = os.path.join(output_folder, image_name)

    cv2.imwrite(output_path, detected_frame)

    print(f"Saved: {output_path}")

print("Object detection completed.")

