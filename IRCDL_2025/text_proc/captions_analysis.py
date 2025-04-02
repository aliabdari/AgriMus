import json
import os

captions = json.load(open('../HowTo100M/caption.json', 'rb'))
videos = os.listdir('dataset/cut_videos')
videos = [x.split('_')[0] for x in videos]
videos = list(set(videos))

print('number of videos:', len(videos))

captions_list = []
for v in videos:
    captions_list.extend(captions[v.split('_')[0]]['text'])

print('Total Captions', len(captions_list))
captions_list = list(set(captions_list))
print(captions_list)
print('Unique Captions', len(captions_list))

# captions
print()
