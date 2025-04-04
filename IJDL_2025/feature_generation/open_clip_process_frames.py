import torch
from PIL import Image
import open_clip
import os
import glob
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
model.to(device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = 'extracted_frames'
list_videos = os.listdir(img_path)

path_image_features = 'open_clip_features/frames/'
os.makedirs(path_image_features, exist_ok=True)


counter = 0
for v in tqdm(list_videos):
    mus_images = []
    jpg_images = [x for x in os.listdir(img_path + os.sep + v)]
    jpg_images.sort()
    torch_imgs = torch.empty(len(jpg_images), 3, 224, 224)
    for idx, img in enumerate(jpg_images):
        torch_imgs[idx, :, :, :] = preprocess(Image.open(img_path + os.sep + v + os.sep +img))
        counter += 1
    torch_imgs = torch_imgs.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(torch_imgs)
        image_features = image_features.cpu()
    torch.save(image_features, f'{path_image_features}{v}.pt')

print(counter)
