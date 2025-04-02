import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at regular intervals
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB for PIL compatibility

        frame_count += 1

    cap.release()
    return frames


# Process video and generate captions
video_path = "dataset/videos/_7FSeDtgOgw.mp4"
frames = extract_frames(video_path, frame_interval=30)  # Extract 1 frame every 30 frames (~1 second for 30 FPS video)

captions = []
for idx, frame in enumerate(frames):
    pil_image = Image.fromarray(frame)  # Convert OpenCV frame (numpy array) to PIL Image
    caption = generate_caption(pil_image)
    captions.append(f"Frame {idx}: {caption}")
    print(f"Generated caption for frame {idx}: {caption}")

# Combine captions for video-level description (example heuristic)
video_caption = " ".join(captions)
print("Video Caption:", video_caption)
