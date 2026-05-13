import cv2
import os

image_folder = "segmented_output"
video_name = "segmented_video.mp4"

images = sorted(os.listdir(image_folder))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (width, height)
)

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()

print("Video created successfully.")

