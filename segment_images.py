


from ultralytics import YOLO
import cv2
import os

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

# Input images folder
input_folder = "train/images"

# Output folder
output_folder = "segmented_output"

os.makedirs(output_folder, exist_ok=True)

# Process all images
for image_name in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image_name)

    # Perform segmentation
    results = model(image_path)

    # Draw segmentation masks
    segmented_frame = results[0].plot()

    # Save output image
    output_path = os.path.join(output_folder, image_name)

    cv2.imwrite(output_path, segmented_frame)

    print(f"Saved: {output_path}")

print("Segmentation completed.")

