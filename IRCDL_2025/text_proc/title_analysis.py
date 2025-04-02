import os.path

from keybert import KeyBERT
import spacy
import pickle
from tqdm import tqdm
from itertools import combinations


def select_frequent_keywords(tk_kw):
    tks = list()
    tks.extend([x for li in tk_kw for x in li])
    tks = [x for x in tks if tks.count(x) >= 2]
    return list(set(tks))


def check_combinations(combs, entire_categories, categories_per_video):
    print('len(combs)', len(combs))
    combs_analysis = dict()
    print('len(categories_per_video)', len(categories_per_video))
    for x in tqdm(combs):
        lists_ = [entire_categories[y] for y in x]
        combs_analysis[x] = list(set(lists_[0]).intersection(*lists_[1:]))
    # for x in combs:
    #     for y in categories_per_video:
    #         if all(elem in y for elem in x):
    #             if x in list(combs_analysis.keys()):
    #                 combs_analysis[x] += 1
    #             else:
    #                 combs_analysis[x] = 1
    return combs_analysis


def get_categories_per_video():
    titles = pickle.load(open('textual_data/titles_info.pkl', 'rb'))
    # titles = [x['title'] for x in titles]

    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    categories = []

    for vid in tqdm(titles):
        topics = kw_model.extract_keywords(vid['title'], keyphrase_ngram_range=(2, 5), top_n=3)

        keywords = [x[0] for x in topics]

        nlp = spacy.load("en_core_web_sm")
        tokenized_keywords = []
        for keyword in keywords:
            doc = nlp(keyword)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            tokenized_keywords.append(tokens)
        # tokenized_keywords = list(set(tokenized_keywords))
        # intersection_tokens = set.intersection(*tokenized_keywords)
        # union_tokens = set.union(*tokenized_keywords)
        categories.append({'vid_id': vid['vid_id'], 'keyword_tokens': select_frequent_keywords(tokenized_keywords)})
        # print('Title: ', txt)
        # print('Tokenized keywords', tokenized_keywords)
        # print('*Tokenized keywords', *tokenized_keywords)
        # print('Obtained Res', select_frequent_keywords(tokenized_keywords))
        # print("Extracted Topics:", topics)
        # print("Intersection Tokens:", intersection_tokens)
        # print("Union Tokens:", union_tokens)
        # print('_' * 20)
    # print(categories)
    # categories_ = list(set([x for xx in categories for x in xx]))
    categories_ = list(set([x for xx in categories for x in xx['keyword_tokens']]))
    final_dict = dict()

    for x in categories_:
        for y in range(len(categories)):
            if x in categories[y]['keyword_tokens']:
                if x in list(final_dict.keys()):
                    final_dict[x].append(categories[y]['vid_id'])
                else:
                    final_dict[x] = [categories[y]['vid_id']]
    print('*' * 10)
    print(final_dict)
    # final_dict_tmp = [x for x in final_dict if len(final_dict[x]) >= 5]
    # print('*'*10)
    # print(final_dict_tmp)

    # categories_ = list(set(categories_))
    with open('textual_data/categories_per_video.pkl', 'wb') as file:
        pickle.dump(categories, file)
    with open('textual_data/entire_categories.pkl', 'wb') as file:
        pickle.dump(final_dict, file)

    return categories, final_dict


if __name__ == '__main__':
    if os.path.exists('textual_data/categories_per_video.pkl'):
        categories = pickle.load(open('textual_data/categories_per_video.pkl', 'rb'))
        final_dict = pickle.load(open('textual_data/entire_categories.pkl', 'rb'))
    else:
        categories, final_dict = get_categories_per_video()
    final_tags = list(final_dict.keys())
    filtered_final_tags = [x for x in final_tags if len(final_dict[x]) >= 5]
    print('filtered_final_tags size', len(filtered_final_tags))
    combinations_2 = check_combinations(combs=list(combinations(filtered_final_tags, 2)), entire_categories=final_dict,
                                        categories_per_video=categories)
    combinations_3 = check_combinations(combs=list(combinations(filtered_final_tags, 3)), entire_categories=final_dict,
                                        categories_per_video=categories)

    with open('textual_data/combinations2.pkl', 'wb') as file:
        pickle.dump(combinations_2, file)
    with open('textual_data/combinations3.pkl', 'wb') as file:
        pickle.dump(combinations_3, file)
