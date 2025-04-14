import os
from tqdm import tqdm
import json
from nltk.tokenize import WordPunctTokenizer
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection


def tokenize_paragraph_with_punctuations(paragraph):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(paragraph)
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model.eval()
model.to(device=device)


queries = list()
museums = json.load(open('final_museums.json', 'r'))
for m in museums:
    for r in m['rooms']:
        queries.append(m['context'] + ' ' + r)

print(queries)
print('Initial Queries Number', len(queries))
queries = list(set(queries))
print('Unique Queries Numbers', len(queries))

path_sentence_level_features = 'clip4clip_features/queries/'
os.makedirs(path_sentence_level_features, exist_ok=True)

for q in tqdm(queries):

    with torch.no_grad():
        inputs = tokenizer(q, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach()
        print(final_output.shape)
        torch.save(final_output, f'{path_sentence_level_features}{q}.pt')
