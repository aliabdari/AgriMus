import pickle
import spacy
from tqdm import tqdm
import json


def process(tks, combs):
    nlp = spacy.load("en_core_web_sm")
    tmp_tks = dict()
    final_dict_verb = dict()
    for x in tks:
        doc = nlp(x)
        tmp_tks[x] = doc[0].pos_

    for x in tqdm(tks):
        if tmp_tks[x] == 'VERB':
            final_dict_verb[x] = dict()
            for y in tks:
                if y == x or tmp_tks[y] != 'NOUN':
                    continue
                if (x, y) in combs and len(combs[(x, y)]) >= 1:
                    final_dict_verb[x][y] = combs[(x, y)]
                elif (y, x) in combs and len(combs[(y, x)]) >= 1:
                    final_dict_verb[x][y] = combs[(y, x)]
    sorted_dict = dict(sorted(final_dict_verb.items(), key=lambda item: len(item[1]), reverse=True))
    with open("textual_data/verbs.json", "w") as json_file:
        json.dump(sorted_dict, json_file, indent=4)
    print(sorted_dict)

    final_dict_noun = dict()
    for x in tqdm(tks):
        if tmp_tks[x] == 'NOUN':
            final_dict_noun[x] = dict()
            for y in tks:
                if y == x or tmp_tks[y] != 'VERB':
                    continue
                if (x, y) in combs and len(combs[(x, y)]) >= 1:
                    final_dict_noun[x][y] = combs[(x, y)]
                elif (y, x) in combs and len(combs[(y, x)]) >= 1:
                    final_dict_noun[x][y] = combs[(y, x)]
    sorted_dict = dict(sorted(final_dict_noun.items(), key=lambda item: len(item[1]), reverse=True))
    with open("textual_data/nouns.json", "w") as json_file:
        json.dump(sorted_dict, json_file, indent=4)
    print(sorted_dict)


if __name__ == '__main__':
    tokens = pickle.load(open('textual_data/entire_categories.pkl', 'rb'))
    combinations = pickle.load(open('textual_data/combinations2.pkl', 'rb'))
    process(tokens, combinations)

