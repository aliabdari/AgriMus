import torch
from PIL import Image
import open_clip
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'mobile_clip'
pre_train_dataset = ''
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-S1-OpenCLIP')
model.eval()
model.to(device=device)
# tokenizer = open_clip.get_tokenizer(model_name=model_name)


img_path = 'extracted_frames'
list_videos = os.listdir(img_path)

path_image_features = 'mobile_clip_features/frames/'
os.makedirs(path_image_features, exist_ok=True)

counter = 0
for v in tqdm(list_videos):
    mus_images = []
    jpg_images = [x for x in os.listdir(img_path + os.sep + v)]
    jpg_images.sort()
    torch_imgs = torch.empty(len(jpg_images), 3, 256, 256)
    for idx, img in enumerate(jpg_images):
        torch_imgs[idx, :, :, :] = preprocess_val(Image.open(img_path + os.sep + v + os.sep + img))
        counter += 1
    torch_imgs = torch_imgs.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(torch_imgs)
        image_features = image_features.cpu()
    torch.save(image_features, f'{path_image_features}{v}.pt')

print(counter)

# counter = 0
# for mus in tqdm(list_museums):
#     list_rooms = [ls for ls in os.listdir(img_path + mus) if '.meta' not in ls]
#     list_rooms.sort()
#     mus_images = []
#     for room in list_rooms:
#         # existing_file = os.listdir(img_path + os.sep + mus + os.sep + room)
#         png_files = glob.glob(img_path + mus + os.sep + room + '/*.png')
#         png_files.sort()
#         images = torch.empty(len(png_files), 3, 256, 256)
#         for idx, img in enumerate(png_files):
#             images[idx, :, :, :] = preprocess_val(Image.open(img))
#             counter += 1
#         mus_images.extend(images)
#     torch_imgs = torch.stack(mus_images)
#     torch_imgs = torch_imgs.to(device)
#     with torch.no_grad(), torch.cuda.amp.autocast():
#         image_features = model.encode_image(torch_imgs)
#         # print(image_features.shape)
#         image_features = image_features.cpu()
#     torch.save(image_features, f'{path_image_features}{mus}.pt')
#
# print(counter)
