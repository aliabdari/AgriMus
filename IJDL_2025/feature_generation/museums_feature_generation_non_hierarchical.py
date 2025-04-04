import json
import os
import torch
from tqdm import tqdm


def aggregation_func(mod, x):
    if mod == 'Max':
        max_values_keepdim, _ = torch.max(x, dim=0, keepdim=True)
        return max_values_keepdim
    elif mod == 'Mean':
        return torch.mean(x, dim=0)
    elif mod == 'Median':
        median_values, indices = torch.median(x.to(torch.float32), dim=0)
        return median_values.to(torch.float16)

    # if level == 'frame':
    #     if mod[0] == 'Max':
    #         max_values_keepdim, _ = torch.max(x, dim=0, keepdim=True)
    #         return max_values_keepdim
    #     elif mod[0] == 'Mean':
    #         return torch.mean(x, dim=0)
    #     elif mod[0] == 'Median':
    #         median_values, indices = torch.median(x, dim=0)
    #         return median_values
    # if level == 'video':
    #     if mod[1] == 'Max':
    #         max_values_keepdim, _ = torch.max(x, dim=0, keepdim=True)
    #         return max_values_keepdim
    #     elif mod[1] == 'Mean':
    #         return torch.mean(x, dim=0)
    #     elif mod[1] == 'Median':
    #         median_values, indices = torch.median(x, dim=0)
    #         return median_values
    # if level == 'room':
    #     if mod[2] == 'Max':
    #         max_values_keepdim, _ = torch.max(x, dim=0, keepdim=True)
    #         return max_values_keepdim
    #     elif mod[2] == 'Mean':
    #         return torch.mean(x, dim=0)
    #     elif mod[2] == 'Median':
    #         median_values, indices = torch.median(x, dim=0)
    #         return median_values


if __name__ == '__main__':
    museums = json.load(open('../final_museums.json', 'r'))
    base_features_list = ['mobile_clip_features']

    base_features = base_features_list[0]

    if base_features == base_features_list[0]:
        frame_video_representation = [('Mean', 'Mean'), ('Max', 'Mean'), ('Max', 'Max')]

    for fv in tqdm(frame_video_representation):
        path_tensors = f'{base_features}/frames'
        path_output = f'{base_features}/museums_{fv[0]}_{fv[1]}'
        os.makedirs(path_output, exist_ok=True)

        for idx_m, m in enumerate(museums):
            # features_museum = torch.zeros(len(m['rooms']), 512)
            features_museum = list()
            for idx_r, r in enumerate(m['rooms']):
                features_rooms = torch.zeros(len(m['rooms'][r]), 512)
                for idx_v, v in enumerate(m['rooms'][r]):
                    features = torch.load(path_tensors + os.sep + v + '.pt', weights_only=True)
                    features_rooms[idx_v, :] = aggregation_func(fv[0], features)
                features_museum.extend(features_rooms)
            torch.save(aggregation_func(fv[1], torch.stack(features_museum)), path_output + os.sep + 'museum_' + str(idx_m) + '.pt')
