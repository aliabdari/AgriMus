import os

import cv2
import json

if __name__ == '__main__':
    museums = json.load(open('../final_museums.json', 'r'))
    videos = list()
    for m in museums:
        for r in m['rooms']:
            videos.extend(m['rooms'][r])

    print(len(videos))
    videos = list(set(videos))
    print(len(videos))

    path_videos = '../dataset/final_videos/'
    vids = os.listdir(path_videos)
    vids_without_extention = [x.split('.')[0] for x in vids]
    frames_num = list()
    times = list()
    for v in videos:
        cap = cv2.VideoCapture(path_videos + vids[vids_without_extention.index(v)])
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second (FPS): {fps}")
        if not cap.isOpened():
            print("Error: Cannot open video.")
            exit(0)
        else:
            # Get the total number of frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_num.append(frame_count)
            print("Total number of frames:", frame_count)
            times.append(frame_count/fps)
        cap.release()

    print('num times', len(times))
    print('min time', min(times))
    print('max time', max(times))


