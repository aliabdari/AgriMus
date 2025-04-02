import json


if __name__ == '__main__':
    museums = json.load(open('../final_museums.json', 'r'))
    rooms_count = list()
    videos_count = list()
    
    for m in museums:
        rooms_count.append(len(m['rooms']))
        vid_count_tmp = 0
        for r in m['rooms']:
            vid_count_tmp += len(m['rooms'][r])
        videos_count.append(vid_count_tmp)

    print('Average Rooms per Museum', sum(rooms_count)/len(rooms_count))
    print('Average Videos per Museum', sum(videos_count)/len(videos_count))
