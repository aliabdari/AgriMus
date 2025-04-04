import torch
from models.blip import blip_feature_extractor
import os, json
from tqdm import tqdm


if __name__ == '__main__':
    queries = list()
    museums = json.load(open('../../../Agr_dataset_Collection/final_museums.json', 'r'))
    for m in museums:
        for r in m['rooms']:
            queries.append(m['context'] + ' ' + r)

    print(queries)
    print('Initial Queries Number', len(queries))
    queries = list(set(queries))
    print('Unique Queries Numbers', len(queries))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    path_queries_features = 'blip_features/queries/'
    os.makedirs(path_queries_features, exist_ok=True)

    for q in tqdm(queries):
        with torch.no_grad():
            features = model(image=None, caption=q, mode='text', device=device)[0, 0]
            features = features.cpu()
            torch.save(features, f'{path_queries_features}{q}.pt')

# for mus in tqdm(list_txt_files):
#     file = open(path_descriptions + os.sep + mus)
#     text = file.read()
#     split_sentence_level = text.split('.')
#     with torch.no_grad():
#         features_sentences = torch.zeros(len(split_sentence_level[:-1]), 768)
#         features_sentences.to(device=device)
#         for idx in range(len(split_sentence_level[:-1])):
#             features_sentences[idx, :] = model(image=None, caption=split_sentence_level[idx], mode='text', device=device)[0, 0]
#         print(features_sentences.shape)
#
#         file_name = mus.replace('.txt', '')
#         features_sentences = features_sentences.cpu()
#         torch.save(features_sentences, f'{path_sentence_level_features}{file_name}.pt')
#
#         # features_tokens = features_tokens.cpu()
#         # torch.save(features_tokens, f'{path_token_level_features}{file_name}.pt')
#         # with open(f'{path_token_strings}{file_name}.pkl', 'wb') as f:
#         #     pickle.dump(split_token_level, f)



