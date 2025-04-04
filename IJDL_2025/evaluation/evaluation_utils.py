import math


def create_rank(gts):
    for gt in gts:
        scores = gts[gt]['scores']
        ranks = list()
        r = 1
        for idx_s, s in enumerate(scores):
            if len(ranks) == 0:
                ranks.append(r)
                prev_score = s
            else:
                if s == prev_score:
                    ranks.append(r)
                else:
                    r = idx_s + 1
                    ranks.append(r)
                    prev_score = s
        gts[gt]['rank'] = ranks
    return gts


def get_rank(gt, sorted_idxs):
    indexes_of_ones = [index for index, element in enumerate(gt['rank']) if element == 1]
    rank_founds = [sorted_idxs.index(gt['sort'][i]) + 1 for i in indexes_of_ones]
    return min(rank_founds)


def get_ndcg(gt, sorted_idxs, n):
    dcg = 0
    idcg = 0
    for i, j in enumerate(sorted_idxs[:n]):
        idcg += gt['scores'][i] / math.log2(i + 2)
        dcg += gt['scores'][gt['sort'].index(j)] / math.log2(i + 2)
    return dcg / idcg


def get_mrr(gt, sorted_idxs):
    indexes_of_ones = [index for index, element in enumerate(gt['rank']) if element == 1]
    rank_founds = [sorted_idxs.index(gt['sort'][i]) + 1 for i in indexes_of_ones]
    return 1/min(rank_founds)


def get_recall(gt, sorted_idxs, recall_num):
    # ranks_ = [gt['rank'][gt['sort'].index(x)] for x in sorted_idxs[:recall_num]]
    first_ranks_gt = [gt['sort'][i] for i in range(len(gt['sort'])) if gt['rank'][i] == 1]
    indexes_of_ones = [index for index, element in enumerate(gt['rank']) if element == 1]
    # ranks = [x == gt['sort'][0] for x in sorted_idxs[:recall_num]]
    return 1 if set(first_ranks_gt) & set(sorted_idxs[:recall_num]) else 0
    # return 1 if min(ranks) <= recall_num else 0
