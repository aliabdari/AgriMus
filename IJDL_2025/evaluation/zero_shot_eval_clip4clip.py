import os
import torch
import torch.nn.functional as F
import json
from evaluation_utils import get_mrr, get_rank, get_ndcg, get_recall, create_rank
import statistics


def provide_features(base_features, fvr):
    if len(base_features) == 1:
        # Load Queries
        path_queries = f'../feature_generation/{base_features[0]}/queries/'
        queries_text = [q.split('.')[0] for q in os.listdir(path_queries)]
        queries = [torch.load(path_queries + q, weights_only=True) for q in os.listdir(path_queries)]

        # Load Museums
        path_museums = f'../feature_generation/{base_features[0]}/museums_{fvr[0]}_{fvr[1]}/'
        museums = [torch.load(path_museums + 'museum_' + str(m) + '.pt', weights_only=True).unsqueeze(0) for m in
                   range(len(os.listdir(path_museums)))]

    return queries_text, queries, museums


if __name__ == '__main__':
    path_ground_truth = '../ground_truth.json'
    ground_truth = json.load(open(path_ground_truth, 'r'))
    ground_truth = create_rank(ground_truth)

    base_features_list = ['clip4clip_features']

    base_features = base_features_list[:]
    print('base features', base_features)

    frame_video_room_representation = [('Mean', 'Mean')]

    # if base_features == [base_features_list[0]]:
    #     frame_video_room_representation = [('Mean', 'Mean', 'Mean'), ('Median', 'Mean', 'Mean'),
    #                                        ('Max', 'Mean', 'Mean'), ('Mean', 'Median', 'Mean'),
    #                                        ('Median', 'Median', 'Mean'), ('Max', 'Median', 'Mean'),
    #                                        ('Mean', 'Max', 'Mean'), ('Median', 'Max', 'Mean'), ('Max', 'Max', 'Mean')]
    # else:
    #     frame_video_room_representation = [('Mean', 'Mean', 'Mean')]

    for fvr in frame_video_room_representation:
        # Load Queries
        # path_queries = f'../feature_generation/{base_features}/queries/'
        # queries_text = [q.split('.')[0] for q in os.listdir(path_queries)]
        # queries = [torch.load(path_queries + q, weights_only=True) for q in os.listdir(path_queries)]
        #
        # # Load Museums
        # path_museums = f'../feature_generation/{base_features}/museums_{fvr[0]}_{fvr[1]}_{fvr[2]}/'
        # museums = [torch.load(path_museums + 'museum_' + str(m) + '.pt', weights_only=True) for m in
        #            range(len(os.listdir(path_museums)))]
        queries_text, queries, museums = provide_features(base_features=base_features, fvr=fvr)

        recall_1 = list()
        recall_5 = list()
        recall_10 = list()

        rank = list()
        mrr = list()
        ndcg_5 = list()
        ndcg_10 = list()

        for idx_q, q in enumerate(queries):
            similarities = [(F.cosine_similarity(q, m)).item() for m in museums]
            indexed_similarities = list(enumerate(similarities))
            sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
            sorted_indexes = [i[0] for i in sorted_similarities]

            key_gt = '"(' + "'" + queries_text[idx_q].split(' ')[0] + "'" + ', ' + "'" + queries_text[idx_q].split(' ')[1] + "'" + ')"'
            gt = ground_truth[key_gt.strip('"')]

            recall_1.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=1))
            recall_5.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=5))
            recall_10.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=10))

            rank.append(get_rank(gt=gt, sorted_idxs=sorted_indexes))
            mrr.append(get_mrr(gt=gt, sorted_idxs=sorted_indexes))
            ndcg_5.append(get_ndcg(gt=gt, sorted_idxs=sorted_indexes, n=5))
            ndcg_10.append(get_ndcg(gt=gt, sorted_idxs=sorted_indexes, n=10))

        print('Results of Video ', fvr[0], ', Room ', fvr[1], ' :')

        print('Recall@1', (sum(recall_1) / len(recall_1)) * 100)
        print('Recall@5', (sum(recall_5) / len(recall_5)) * 100)
        print('Recall@10', (sum(recall_10) / len(recall_10)) * 100)

        print('Median Rank', statistics.median(rank))
        print('MRR', (sum(mrr)/len(mrr)) * 100)
        print('NDCG@5', (sum(ndcg_5) / len(ndcg_5)) * 100)
        print('NDCG@10', (sum(ndcg_10) / len(ndcg_10)) * 100)

        print('*'*10)
