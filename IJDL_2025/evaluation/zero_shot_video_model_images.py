import os
import torch
import torch.nn.functional as F
import json
from evaluation_utils import get_mrr, get_rank, get_ndcg, get_recall, create_rank
import statistics
import numpy as np


def provide_features(base_features, fvr):
    queries = dict()
    if len(base_features) == 1:
        # Load Queries
        path_queries = f'../feature_generation/{base_features[0]}/frames/'
        # queries_text = [q.split('.')[0] for q in os.listdir(path_queries)]
        # queries = {q.split('.')[0] : torch.mean(torch.load(path_queries + q, weights_only=True), 0).cpu() for q in os.listdir(path_queries)}
        for q_ in os.listdir(path_queries):
            numpy_imgs = (torch.load(path_queries + q_, weights_only=True).cpu()).numpy()
            num_frames = numpy_imgs[0:50, :].shape[0] - 1
            # print(num_frames)
            selected_indices = np.linspace(0, num_frames, 32, dtype=int)
            selected_frames = torch.from_numpy(numpy_imgs[selected_indices, :])
            queries[q_.split('.')[0]] = torch.mean(selected_frames, 0)
        # Load Museums
        path_museums = f'../feature_generation/{base_features[0]}/museums_32_{fvr[0]}_{fvr[1]}_{fvr[2]}/'
        museums = [torch.load(path_museums + 'museum_' + str(m) + '.pt', weights_only=True).unsqueeze(0) for m in
                   range(len(os.listdir(path_museums)))]
    # else:
    #     path_queries1 = f'../feature_generation/{base_features_videos[0]}/queries/'
    #     queries_text = [q.split('.')[0] for q in os.listdir(path_queries1)]
    #     queries1 = [torch.load(path_queries1 + q, weights_only=True) for q in os.listdir(path_queries1)]
    #
    #     path_queries2 = f'../feature_generation/{base_features_videos[1]}/queries/'
    #     queries_text2 = [q.split('.')[0] for q in os.listdir(path_queries2)]
    #     queries2 = [torch.load(path_queries2 + q, weights_only=True) for q in os.listdir(path_queries2)]
    #
    #     path_museums1 = f'../feature_generation/{base_features_videos[0]}/museums_Mean_Mean_Mean/'
    #     museums1 = [torch.load(path_museums1 + 'museum_' + str(m) + '.pt', weights_only=True) for m in
    #                range(len(os.listdir(path_museums1)))]
    #
    #     path_museums2 = f'../feature_generation/{base_features_videos[1]}/museums_Mean_Mean_Mean/'
    #     museums2 = [torch.load(path_museums2 + 'museum_' + str(m) + '.pt', weights_only=True) for m in
    #                 range(len(os.listdir(path_museums2)))]
    #
    #     if len(base_features_videos) == 2:
    #         print(queries1[0].shape)
    #         print(queries2[0].shape)
    #         print(museums1[0].shape)
    #         print(museums2[0].shape)
    #         # queries = [torch.cat((queries1[i], queries2[i]), 1) for i in range(len(queries1))]
    #         queries = [torch.cat((q1.unsqueeze(0) if q1.dim() == 1 else q1, q2.unsqueeze(0) if q2.dim() == 1 else q2), 1) for
    #             q1, q2 in zip(queries1, queries2)]
    #         museums = [torch.cat((m1.unsqueeze(0) if m1.dim() == 1 else m1, m2.unsqueeze(0) if m2.dim() == 1 else m2), 1) for
    #             m1, m2 in zip(museums1, museums2)]
    #         # museums = [torch.cat((museums1[i].unsqueeze(0), museums2[i].unsqueeze(0)), 1) for i in range(len(museums1))]
    #         print(museums[0].shape)
    #     elif len(base_features_videos) == 3:
    #         path_queries3 = f'../feature_generation/{base_features_videos[2]}/queries/'
    #         queries_text3 = [q.split('.')[0] for q in os.listdir(path_queries3)]
    #         queries3 = [torch.load(path_queries3 + q, weights_only=True) for q in os.listdir(path_queries3)]
    #         # queries = [torch.cat((queries1[i], queries2[i], queries3[i]), 1) for i in range(len(queries1))]
    #         queries = [torch.cat((
    #             q1.unsqueeze(0) if q1.dim() == 1 else q1,
    #             q2.unsqueeze(0) if q2.dim() == 1 else q2,
    #             q3.unsqueeze(0) if q3.dim() == 1 else q3), 1)
    #             for q1, q2, q3 in zip(queries1, queries2, queries3)]
    #
    #         path_museums3 = f'../feature_generation/{base_features_videos[2]}/museums_Mean_Mean_Mean/'
    #         museums3 = [torch.load(path_museums3 + 'museum_' + str(m) + '.pt', weights_only=True) for m in
    #                     range(len(os.listdir(path_museums3)))]
    #         # museums = [torch.cat((museums1[i].unsqueeze(0), museums2[i].unsqueeze(0), museums3[i].unsqueeze(0)), 1) for i in range(len(museums1))]
    #         museums = [torch.cat((
    #             m1.unsqueeze(0) if m1.dim() == 1 else m1,
    #             m2.unsqueeze(0) if m2.dim() == 1 else m2,
    #             m3.unsqueeze(0) if m3.dim() == 1 else m3), 1)
    #             for m1, m2, m3 in zip(museums1, museums2, museums3)]
    # print(queries_text[1], queries_text2[1], queries_text3[1])
    # print(queries_text[10], queries_text2[10], queries_text3[10])
    # print(queries_text[5], queries_text2[5], queries_text3[5])
    # print(queries_text[-1], queries_text2[-1], queries_text3[-1])
    # print(queries_text[-18], queries_text2[-18], queries_text3[-18])
    return queries, museums


if __name__ == '__main__':
    path_ground_truth = '../ground_truth_videos.json'
    ground_truth = json.load(open(path_ground_truth, 'r'))
    ground_truth = create_rank(ground_truth)

    base_features_list = ['mobile_clip_features']

    base_features = [base_features_list[0]]
    print('base features', base_features)

    # video_room_representation = [('Mean', 'Mean', 'Mean'), ('Median', 'Mean', 'Mean'), ('Mean', 'Median', 'Mean')]
    video_room_representation = [('Mean', 'Mean', 'Mean')]

    for fvr in video_room_representation:
        # Load Queries
        # path_queries = f'../feature_generation/{base_features}/queries/'
        # queries_text = [q.split('.')[0] for q in os.listdir(path_queries)]
        # queries = [torch.load(path_queries + q, weights_only=True) for q in os.listdir(path_queries)]
        #
        # # Load Museums
        # path_museums = f'../feature_generation/{base_features}/museums_{fvr[0]}_{fvr[1]}_{fvr[2]}/'
        # museums = [torch.load(path_museums + 'museum_' + str(m) + '.pt', weights_only=True) for m in
        #            range(len(os.listdir(path_museums)))]
        queries_video, museums = provide_features(base_features=base_features, fvr=fvr)

        recall_1 = list()
        recall_5 = list()
        recall_10 = list()

        rank = list()
        mrr = list()
        ndcg_5 = list()
        ndcg_10 = list()

        for idx_q, q in enumerate(queries_video):
            similarities = [(F.cosine_similarity(queries_video[q.split('.')[0]], m)).item() for m in museums]
            indexed_similarities = list(enumerate(similarities))
            sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
            sorted_indexes = [i[0] for i in sorted_similarities]

            key_gt = q
            gt = ground_truth[key_gt.strip('"')]

            recall_1.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=1))
            recall_5.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=5))
            recall_10.append(get_recall(gt=gt, sorted_idxs=sorted_indexes, recall_num=10))

            rank.append(get_rank(gt=gt, sorted_idxs=sorted_indexes))
            mrr.append(get_mrr(gt=gt, sorted_idxs=sorted_indexes))
            ndcg_5.append(get_ndcg(gt=gt, sorted_idxs=sorted_indexes, n=5))
            ndcg_10.append(get_ndcg(gt=gt, sorted_idxs=sorted_indexes, n=10))

        print('Results of Frame ', fvr[0], ', Video ', fvr[1], ', Room ', fvr[1], ' :')

        print('Recall@1', (sum(recall_1) / len(recall_1)) * 100)
        print('Recall@5', (sum(recall_5) / len(recall_5)) * 100)
        print('Recall@10', (sum(recall_10) / len(recall_10)) * 100)

        print('Median Rank', statistics.median(rank))
        print('MRR', (sum(mrr)/len(mrr)) * 100)
        print('NDCG@5', (sum(ndcg_5) / len(ndcg_5)) * 100)
        print('NDCG@10', (sum(ndcg_10) / len(ndcg_10)) * 100)

        print('*'*10)
