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
    base_features_list = ['open_clip_features', 'mobile_clip_features', 'blip_features']

    base_features = base_features_list[1]

    frame_video_room_representation = [('Mean', 'Mean'), ('Median', 'Mean'), ('Max', 'Mean')]

    for fvr in tqdm(frame_video_room_representation):
        path_tensors = f'{base_features}/frames'
        path_output = f'{base_features}/museums_{fvr[0]}_{fvr[1]}'
        os.makedirs(path_output, exist_ok=True)

        for idx_m, m in enumerate(museums):
            features_museum = torch.zeros(len(m['rooms']), 768)
            for idx_r, r in enumerate(m['rooms']):
                features_rooms = torch.zeros(len(m['rooms'][r]), 768)
                for idx_v, v in enumerate(m['rooms'][r]):
                    features = torch.load(path_tensors + os.sep + v + '.pt', weights_only=True)
                    features_rooms[idx_v, :] = aggregation_func(fvr[0], features)
                features_museum[idx_r, :] = aggregation_func(fvr[1], features_rooms)
            torch.save(aggregation_func(fvr[2], features_museum), path_output + os.sep + 'museum_' + str(idx_m) + '.pt')
