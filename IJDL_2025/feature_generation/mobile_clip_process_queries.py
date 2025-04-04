'''
This script is developed to extract descriptions features using mbileclip model
'''
import torch
import open_clip
import os
from tqdm import tqdm
import json
from nltk.tokenize import WordPunctTokenizer


def tokenize_paragraph_with_punctuations(paragraph):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(paragraph)
    return tokens


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
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-S1-OpenCLIP')
    tokenizer = open_clip.get_tokenizer('hf-hub:apple/MobileCLIP-S1-OpenCLIP')
    model.eval()
    model.to(device=device)

    path_queries_features = 'mobile_clip_features/queries/'
    os.makedirs(path_queries_features, exist_ok=True)

    for q in tqdm(queries):
        query = tokenizer(q)
        query = query.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            features_sentences = model.encode_text(query)

            features_sentences = features_sentences.cpu()
            torch.save(features_sentences, f'{path_queries_features}{q}.pt')


# path_descriptions = '/data01/aabdari/dataset_museum3k/descriptions_museums_3k'
# list_txt_files = os.listdir(path_descriptions)
#
# path_sentence_level_features = f'../museums3k_features/{model_name}_{pre_train_dataset}/descriptions/sentences/'
# path_token_level_features = f'../museums3k_features/{model_name}_{pre_train_dataset}/descriptions/tokens/'
# path_token_strings = f'../museums3k_features/{model_name}_{pre_train_dataset}/descriptions/tokens_strings/'
#
# os.makedirs(path_sentence_level_features, exist_ok=True)
# os.makedirs(path_token_level_features, exist_ok=True)
# os.makedirs(path_token_strings, exist_ok=True)
#
# for mus in tqdm(list_txt_files):
#     file = open(path_descriptions + os.sep + mus)
#     text = file.read()
#     split_sentence_level = text.split('.')
#     split_sentence_level_tokenized = tokenizer(split_sentence_level[:-1])
#     split_sentence_level_tokenized = split_sentence_level_tokenized.to(device)
#
#     split_token_level = tokenize_paragraph_with_punctuations(text)
#     split_token_level_tokenized = tokenizer(split_token_level)
#     split_token_level_tokenized = split_token_level_tokenized.to(device)
#     if len(split_token_level_tokenized) != len(split_token_level):
#         print(len(split_token_level_tokenized))
#         print(len(split_token_level))
#         exit(0)
#     with torch.no_grad(), torch.cuda.amp.autocast():
#         features_sentences = model.encode_text(split_sentence_level_tokenized)
#         print(features_sentences.shape)
#         features_tokens = model.encode_text(split_token_level_tokenized)
#
#         file_name = mus.replace('.txt', '')
#         features_sentences = features_sentences.cpu()
#         torch.save(features_sentences, f'{path_sentence_level_features}{file_name}.pt')
#
#         features_tokens = features_tokens.cpu()
#         torch.save(features_tokens, f'{path_token_level_features}{file_name}.pt')
#         with open(f'{path_token_strings}{file_name}.pkl', 'wb') as f:
#             pickle.dump(split_token_level, f)

