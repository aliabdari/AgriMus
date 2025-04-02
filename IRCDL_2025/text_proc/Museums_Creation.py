import json
import random
import pickle
from tqdm import tqdm


def find_suitable_video(context, room, used_videos, entire_titles):
    for x in entire_titles:
        if x is not None and all(item in x['title'] for item in [context, room]):
            if x['vid_id'] not in used_videos:
                return x['vid_id']
    return None


def process(js, museums, counter_not_available, entire_titles, list_new_vids):
    museums_id = len(museums) + 1
    for i in tqdm(js):
        num_subcat = len(js[i])
        num_repeat = 5 if num_subcat > 7 else 2
        for _ in range(num_repeat):
            if 4 <= num_subcat:
                num_rooms = random.randint(4, min(num_subcat, 6))
                selected_subcat = random.sample(list(js[i].keys()), num_rooms)
                new_museum = {'id': museums_id, 'context': i, 'rooms': {x: [] for x in selected_subcat}}
                museums_id += 1
                for j in selected_subcat:
                    new_museum['rooms'][j].extend((js[i][j])[:4])
                    if len(js[i][j]) == 1:
                        new_vid = find_suitable_video(i, j, new_museum['rooms'][j], entire_titles=entire_titles)
                        if new_vid is None:
                            del new_museum['rooms'][j]
                        else:
                            list_new_vids.append(new_vid)
                            new_museum['rooms'][j].append(new_vid)
                            counter_not_available += 1
                if len(new_museum['rooms']) >= 4:
                    museums.append(new_museum)
    return museums, counter_not_available, list_new_vids


if __name__ == '__main__':
    verbs = json.load(open('textual_data/verbs.json', 'rb'))
    nouns = json.load(open('textual_data/nouns.json', 'rb'))
    entire_titles = pickle.load(open('textual_data/entire_gardening_titles_info.pkl', 'rb'))
    museums = list()
    counter_not_available = 0
    museums, counter_not_available, list_new_vids = process(js=verbs, museums=museums,
                                                            counter_not_available=counter_not_available,
                                                            entire_titles=entire_titles, list_new_vids=list())
    print('Verbs related Museums', len(museums))
    museums, counter_not_available, list_new_vids = process(js=nouns, museums=museums,
                                                            counter_not_available=counter_not_available,
                                                            entire_titles=entire_titles, list_new_vids=list_new_vids)
    print('Nouns and Verbs Museums', len(museums))
    file = open('../Museums.json', 'w')
    json.dump(museums, file, indent=4)

    file = open('../added_videos.pkl', 'wb')
    pickle.dump(list_new_vids, file)

    print('Number not available video', counter_not_available)
    print('Num new videos', len(list_new_vids))

    # count_rooms_without_None = 0
    # for m in museums:
    #     for r in m['rooms']:
    #         print(m['rooms'][r])
    #         if None in r:
    #             count_rooms_without_None += 1
    # print(count_rooms_without_None)
