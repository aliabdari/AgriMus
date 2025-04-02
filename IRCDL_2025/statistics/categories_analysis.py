import json
from collections import Counter
import matplotlib.pylab as plt
from itertools import islice


if __name__ == '__main__':
    museums = json.load(open('../final_museums.json', 'r'))
    tags = list()
    for m in museums:
        for r in m['rooms']:
            tags.append((m['context'], r))
    print(tags)
    print(set(tags))
    set_tags = list(set(tags))
    print(len(tags), len(set_tags))
    tag_counts = Counter(tags)
    print(tag_counts)
    counts = [tag_counts[x] for x in tag_counts]
    sub_tags_counts = {x: tag_counts[x] for x in tag_counts if tag_counts[x] >= 3}
    sub_tags_counts = dict(sorted(sub_tags_counts.items(), key=lambda item: item[1], reverse=True))
    first_12 = dict(islice(sub_tags_counts.items(), 12))

    topics = ['{} & {}'.format(k[0], k[1]) for k in first_12.keys()]  # Create readable labels
    frequencies = list(first_12.values())

    # Plot the horizontal bar chart
    # plt.figure(figsize=(10, 6))
    plt.barh(topics, frequencies, color='skyblue', edgecolor='black')
    plt.xlabel('Repetition of Topics', fontsize=12)
    plt.ylabel('Topics', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('topics_frequency_per_topic.png', dpi=600, bbox_inches='tight')

    plt.figure()
    number_counts = Counter(counts)

    keys = [1, 2, 3, 4, 5]
    values = [number_counts[i] for i in keys]

    plt.plot(keys, values, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Repetition per Topics', fontsize=12)
    plt.ylabel('Frequency of Topics', fontsize=12)
    plt.xticks(keys)
    plt.savefig('frequency_topics.png', dpi=600, bbox_inches='tight')
    #
