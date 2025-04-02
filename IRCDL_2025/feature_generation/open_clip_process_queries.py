import torch
import open_clip
import os
from tqdm import tqdm
import json


if __name__ == '__main__':
    queries = list()
    museums = json.load(open('../final_museums.json', 'r'))
    for m in museums:
        for r in m['rooms']:
            queries.append(m['context'] + ' ' + r)

    print(queries)
    print('Initial Queries Number', len(queries))
    queries = list(set(queries))
    print('Unique Queries Numbers', len(queries))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    model.to(device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    path_queries_features = 'open_clip_features/queries/'
    os.makedirs(path_queries_features, exist_ok=True)

    for q in tqdm(queries):
        query = tokenizer(q)
        query = query.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            features_sentences = model.encode_text(query)

            features_sentences = features_sentences.cpu()
            torch.save(features_sentences, f'{path_queries_features}{q}.pt')
