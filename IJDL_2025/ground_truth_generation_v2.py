import json
import pickle


def get_unique_categories(museums):
    categories = list()
    for m in museums:
        for r in m['rooms']:
            categories.append((m['context'], r))
    print('num categories:', len(categories))
    return list(set(categories))


def get_ground_truth(museums, unique_categories):
    titles = pickle.load(open('metadata/entire_gardening_titles_info.pkl', 'rb'))
    titles = {x['vid_id']: x['title'] for x in titles if x is not None}
    ground_truth = dict()
    for c in unique_categories:
        tmp_rank = list()
        for m in museums:
            tmp_score = 0.0
            for r in m['rooms']:
                for v in m['rooms'][r]:
                    if all(item in titles[v] for item in [m['context'], r]):
                        tmp_score += 0.1
            tmp_rank.append(round(tmp_score, 1))
        sorted_indexes = sorted(range(len(tmp_rank)), key=lambda i: tmp_rank[i], reverse=True)
        ground_truth[str(c)] = {'sort': sorted_indexes, 'scores': sorted(tmp_rank, reverse=True)}
    return ground_truth


if __name__ == '__main__':
    museums = json.load(open('final_museums.json', 'r'))
    unique_categories = get_unique_categories(museums)
    gt = get_ground_truth(museums, unique_categories)
    file = open('ground_truth_v2.json', 'w')
    json.dump(gt, file)
    print('Process Finished')
