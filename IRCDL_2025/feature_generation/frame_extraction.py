import cv2
import torch
from PIL import Image
import open_clip
import os
import json
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import argparse


def process_batch_videos(videos, output_dir, n_jobs=20):
    with tqdm_joblib(tqdm(desc="Processing Videos", total=len(videos))) as progress_bar:
        sampled_points_batch = Parallel(n_jobs=n_jobs)(
            delayed(extract_frames_v2)(v, output_dir) for v in videos
        )
    return sampled_points_batch


def extract_frames_v2(v, output_dir):
    print('video', v)
    output_dir_cur = output_dir + os.sep + v.split('.')[0]
    if not os.path.exists(output_dir_cur):
        cap = cv2.VideoCapture('../dataset/final_videos' + os.sep + v)
        if not cap.isOpened():
            print(f"Error: Cannot open video file.{v}")
            exit(0)
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"Total number of frames: {total_frames}")
            frame_interval = (total_frames - 100) // 50

            if not os.path.exists(output_dir_cur):
                os.makedirs(output_dir_cur)

            ffmpeg_command = [
                "ffmpeg",
                "-i", '../dataset/final_videos' + os.sep + v,  # Input video file
                "-vf", f"select=not(mod(n\\,{frame_interval}))",  # Select every Nth frame
                "-vsync", "vfr",  # Variable frame rate to sync output
                "-q:v", "2",  # Set quality for output frames
                os.path.join(output_dir_cur, "frame_%04d.jpg")  # Output frame naming pattern
            ]
            try:
                subprocess.run(ffmpeg_command, check=True)
            except:
                pass

            extracted_frames = sorted(
                [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
            )


def extract_frames(video_path, output_dir, frame_interval):
    """
    Extracts every 'frame_interval' frame from the video using FFmpeg.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the extracted frames.
        frame_interval (int): Interval of frames to extract (e.g., every 50 frames).

    Returns:
        List[str]: List of file paths for the extracted frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,  # Input video file
        "-vf", f"select=not(mod(n\\,{frame_interval}))",  # Select every Nth frame
        "-vsync", "vfr",  # Variable frame rate to sync output
        "-q:v", "2",  # Set quality for output frames
        os.path.join(output_dir, "frame_%04d.jpg")  # Output frame naming pattern
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)

        extracted_frames = sorted(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
        )
    except:
        pass


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    model.to(device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # path_image_features = 'open_clip_features_museums_agr/images/'
    # os.makedirs(path_image_features, exist_ok=True)

    path_video = '../dataset/final_videos'
    videos = os.listdir(path_video)
    vid_without_tag = [x.split('.')[0] for x in videos]
    museums = json.load(open('../final_museums.json', 'r'))

    output_dir = 'extracted_frames'

    # process_batch_videos(videos=videos, output_dir=output_dir)

    for m in tqdm(museums):
        for r in m['rooms']:
            for v in m['rooms'][r]:
                output_dir_cur = output_dir + os.sep + v
                if not os.path.exists(output_dir_cur):
                    cap = cv2.VideoCapture(path_video + os.sep + videos[vid_without_tag.index(v)])
                    if not cap.isOpened():
                        print(f"Error: Cannot open video file.{v}")
                        exit(0)
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(f"Total number of frames: {total_frames}")
                        interval = (total_frames - 100) // 50
                        extract_frames(path_video + os.sep + videos[vid_without_tag.index(v)], output_dir_cur, interval)

                # cap = cv2.VideoCapture(path_video + os.sep + videos[vid_without_tag.index(v)])
                # fps = cap.get(cv2.CAP_PROP_FPS)
                # # print(f"Frames per second (FPS): {fps}")
                # if not cap.isOpened():
                #     print(f"Error: Cannot open video file.{v}")
                #     exit(0)
                # else:
                #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #     print(f"Total number of frames: {total_frames}")
                #     interval = (total_frames - 100) // 50
                #     print(f'interval sampling is every {interval} frames')
                #     frames = list()
                #     for f_i in range(100, total_frames - 1, interval):
                #         cap.set(cv2.CAP_PROP_POS_FRAMES, f_i)
                #         success, frame = cap.read()
                #         if not success:
                #             print(f'Problem {v} in frame {f_i}')
                #         else:
                #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #             frames.append(Image.fromarray(frame))
                #             # print(frame.shape)
                #
                #     images = torch.empty(len(frames), 3, 224, 224)
                #     print('len frames', len(frames))
                #     for idx, img in enumerate(frames):
                #         images[idx, :, :, :] = preprocess(img)
                #
                #     # torch_imgs = torch.stack(images)
                #     images = images.to(device)
                #     print('shape images', images.shape)
                #     with torch.no_grad(), torch.amp.autocast('cuda'):
                #         image_features = model.encode_image(images)
                #         print(image_features.shape)
                #         # image_features = image_features.cpu()
