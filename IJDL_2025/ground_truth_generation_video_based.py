import json
import pickle


def get_unique_videos_topics(museums):
    videos = list()
    topics = list()
    topics_videos = list()
    videos_dict = dict()
    for m in museums:
        for r in m['rooms']:
            # topic = (m['context'], r)
            for v in m['rooms'][r]:
                if v in list(videos_dict.keys()):
                    videos_dict[v].append((m['context'], r))
                    videos_dict[v] = list(set(videos_dict[v]))
                else:
                    videos_dict[v] = [(m['context'], r)]
            # list_vid_topic = [{v:(m['context'], r)} for v in m['rooms'][r]]
            # videos.extend(m['rooms'][r])
            # topics_videos.extend(list_vid_topic)
            # topics.append((m['context'], r))
    # print('num videos:', len(videos))
    # return list(set(videos))
    return videos_dict


def get_ground_truth_video_topic(museums, unique_videos_dict):
    unique_videos = list(unique_videos_dict.keys())
    # titles = pickle.load(open('text_proc/textual_data/entire_gardening_titles_info.pkl', 'rb'))
    # titles = {x['vid_id']: x['title'] for x in titles if x is not None}
    ground_truth = dict()
    for uv in unique_videos:
        tmp_rank = list()
        for m in museums:
            tmp_score = 0.0
            for r in m['rooms']:
                if (m['context'], r) in unique_videos_dict[uv]:
                    tmp_score += 1
                else:
                    for v in m['rooms'][r]:
                        if bool(set(unique_videos_dict[v]) & set(unique_videos_dict[uv])):
                            tmp_score += 0.1
            tmp_rank.append(round(tmp_score, 1))
        sorted_indexes = sorted(range(len(tmp_rank)), key=lambda i: tmp_rank[i], reverse=True)
        ground_truth[str(uv)] = {'sort': sorted_indexes, 'scores': sorted(tmp_rank, reverse=True)}
    return ground_truth


if __name__ == '__main__':
    museums = json.load(open('final_museums.json', 'r'))
    unique_videos_topics = get_unique_videos_topics(museums)
    gt = get_ground_truth_video_topic(museums, unique_videos_topics)
    file = open('ground_truth_videos.json', 'w')
    json.dump(gt, file)
    print('Process Finished')
