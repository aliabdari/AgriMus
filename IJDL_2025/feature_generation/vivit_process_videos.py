import os
from tqdm import tqdm
import numpy as np
import json
from transformers import VivitImageProcessor, VivitModel, VivitConfig
import torch
from PIL import Image

np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    print(1)
    converted_len = int(clip_len * frame_sample_rate)
    print(2, seg_len)
    end_idx = np.random.randint(converted_len, seg_len)
    print(3)
    start_idx = end_idx - converted_len
    print(4)
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    print(5)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    print(6)
    return indices


# def find_video(v_):
#     for idx, ov in enumerate(list_vids_with_extension):
#         if v_ in ov:
#             return ov


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

num_frames_default = 50
ignore_mismatched_sizes = False

model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
model.eval()
model.to(device)

museums = json.load(open('../final_museums.json', 'r'))

list_videos = list()
for m in museums:
    for r in m['rooms']:
        list_videos.extend(m['rooms'][r])

print('Num total vids', len(list_videos))
list_videos = list(set(list_videos))
print('Num unique vids', len(list_videos))

list_fail = list()

img_path = '/data01/aabdari/projects/Agr_dataset_Collection/feature_generation/extracted_frames_v2/'

path_video_features = 'vivit_features/'
os.makedirs(path_video_features, exist_ok=True)

counter = 0
for v in tqdm(list_videos):
    mus_images = []
    jpg_images = [x for x in os.listdir(img_path + os.sep + v)]
    jpg_images.sort()
    # numpy_imgs = np.empty(len(jpg_images), 3, 224, 224)
    numpy_imgs = list()
    for idx, img in enumerate(jpg_images):
        numpy_imgs.append((Image.open(img_path + os.sep + v + os.sep + img)).resize((224, 224)))
        # numpy_imgs[idx, :, :, :] = preprocess(Image.open(img_path + os.sep + v + os.sep + img))
        counter += 1
    numpy_imgs = np.stack(numpy_imgs)
    print(numpy_imgs.shape)
    try:
        num_frames = numpy_imgs[1:50, :, :].shape[0] - 1
        print(num_frames)
        selected_indices = np.linspace(1, num_frames, 32, dtype=int)
        selected_frames = numpy_imgs[selected_indices, :, :]
        print(len(selected_frames))
        inputs = image_processor(list(selected_frames), return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        print(last_hidden_states.shape)
        print(type(last_hidden_states[:, 0, :]))
        torch.save(last_hidden_states[:, 0, :], f'{path_video_features}{v}.pt')
    except:
        list_fail.append(v)

print('Count Image', counter)
print('failing list', list_fail)
