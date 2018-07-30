import json
import numpy as np
import pdb


def len_stat(mylist):
    intended_lens = range(20, 40, 5)
    for l in intended_lens:
        num = len(np.where(np.array(mylist) <= l)[0])
        print('%f of sentences is shorter than %d words' % (float(num)/len(mylist), l))


def count_word(sentences, word_count_threshold):
    word_counts = {}
    nsents = 0
    nwords = 0
    for sent in sentences:
        nsents += 1
        for w in sent.lower().split():
            nwords += 1
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    preserved_words = 0
    for k, v in word_counts.items():
        if word_counts[k] >= word_count_threshold:
            preserved_words += v

    print('the total amount of sentences is %d' % nsents)
    print 'counts >= %d: %f of words are preserved, %d out of %d unit words are preserved' % (word_count_threshold,  float(preserved_words)/nwords, len(vocab), len(word_counts))


json_file = '/scratch4/sunxm/caption_guided_saliency/DATA/HRI_Captions/HDD-VC_v0.1.2_with_split.json'
with open(json_file, 'r') as f:
    data = json.load(f)


sentences = data['sentences']

max_event = 0
max_maneuver = 0
max_merged = 0
len_event = []
len_maneuver = []
len_merged = []
Event = {}
caption_event_lines = []
caption_maneuver_lines = []
caption_merged_lines = []

for idx in range(len(sentences)):
    caption_event = sentences[idx]['caption_event']
    caption_event = caption_event.replace(',', ' ')
    caption_event = caption_event.replace('.', ' ')
    if len(caption_event.split()) > 2:
        caption_event_lines.append(caption_event.lower() + '\n')
        len_event.append(len(caption_event.split()))

    caption_maneuver = sentences[idx]['caption_maneuver']
    caption_maneuver = caption_maneuver.replace(',', ' ')
    caption_maneuver = caption_maneuver.replace('.', ' ')
    caption_maneuver_lines.append(caption_maneuver.lower() + '\n')
    len_maneuver.append(len(caption_maneuver.split()))

    caption_merged = sentences[idx]['caption_merged']
    caption_merged = caption_merged.replace(',', ' ')
    caption_merged = caption_merged.replace('.', ' ')
    caption_merged_lines.append(caption_merged.lower() + '\n')
    len_merged.append(len(caption_merged.split()))
    # extract split
    event = sentences[idx]['event']
    if event in Event.keys():
        Event[event] += 1
    else:
        Event[event] = 1

with open('caption_events.txt', 'w+') as f:
    f.writelines(caption_event_lines)

with open('caption_maneuver.txt', 'w+') as f:
    f.writelines(caption_maneuver_lines)

with open('caption_merged.txt', 'w+') as f:
    f.writelines(caption_merged_lines)