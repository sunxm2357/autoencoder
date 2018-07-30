# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
# import pdb
# import time
#
#
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tqdm
import os


def read_sequence(file_name, max_len=25, min_len=2):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    documents = []
    counts = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < max_len and len(tokens) > min_len:
            new_line = " ".join(tokens)
            if new_line not in documents:
                documents.append(new_line)
                counts.append(1)
            else:
                idx = documents.index(new_line)
                counts[idx] += 1
    return documents, counts

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
caption_type = 'maneuver'
caption_file = '../hri_data/caption_%s.txt' % caption_type
documents, counts = read_sequence(caption_file)
embedding_file = '%s_embeddings.npy' % caption_type
if os.path.exists(embedding_file):
    embeddings = np.load(embedding_file)
else:
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        batch_size = 32
        first=True
        for start_idx in tqdm.tqdm(range(0, len(documents), batch_size)):
            messages = documents[start_idx: min(start_idx+batch_size, len(documents))]
            message_embeddings = session.run(embed(messages))
            if first:
                embeddings = message_embeddings
                first = False
            else:
                embeddings = np.concatenate((embeddings, message_embeddings), axis=0)
            np.save(embedding_file, embeddings)

cls_num = 10
cluster = AgglomerativeClustering(n_clusters=cls_num, affinity='cosine', linkage='complete').fit(embeddings)
labels = cluster.labels_
output_file = 'caption_%s_cluster_%d.txt' % (caption_type, cls_num)
cluster_items = np.zeros((cls_num,))
for clss_id in range(cls_num):
    # print('-' * 50)
    for idx, sentence in enumerate(documents):
        if labels[idx] == clss_id:
        #     print(sentence, counts[idx])
            cluster_items[clss_id] += counts[idx]

# sort cluster_items
lines = []
idxs = np.argsort(cluster_items).tolist()[::-1]
for clss_id in idxs:
    print('-' * 20 + '%d/%d' %(cluster_items[clss_id], np.sum(counts)) + '-' * 20)
    lines.append('-' * 20 + '%d/%d' %(cluster_items[clss_id], np.sum(counts)) + '-' * 20 + '\n')
    for idx, sentence in enumerate(documents):
        if labels[idx] == clss_id:
            print(sentence, counts[idx])
            lines.append(sentence + '  %d\n' % counts[idx])

with open(output_file, 'w+') as f:
    f.writelines(lines)

