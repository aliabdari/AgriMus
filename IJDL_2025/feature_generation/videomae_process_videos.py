import os
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEConfig
import torch
from PIL import Image

np.random.seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

ignore_mismatched_sizes = False

configuration = VideoMAEConfig(use_mean_pooling=False)
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", config=configuration)
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

path_video_features = 'videomae_features/'
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
        selected_indices = np.linspace(1, num_frames, 16, dtype=int)
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
        print('num failed', len(list_fail))

print('Count Image', counter)
print('failing list', list_fail)
