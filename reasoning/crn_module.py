import os
import pickle

import numpy as np


class ReasoningModule(object):
    def __init__(self):
        # Load parameters for CRN
        with open(os.path.dirname(os.path.realpath(__file__)) + '/crn_reasoning.pickle', 'rb') as f:
            model = pickle.load(f)
        self.centroids = model['centroid']
        self.num_clusters = self.centroids.shape[0]
        self.normalize_min = model['maxmin'][1]
        # max - min
        self.normalize_denom = model['maxmin'][0] - model['maxmin'][1]

    def reason(self, input):
        # Normalize the received input
        normalized_input = (input-self.normalize_min)/self.normalize_denom

        # Find the closest cluster centroid
        norm = np.linalg.norm(self.centroids - normalized_input, axis=1)
        T = np.exp(-norm)
        winner = np.argmax(T)

        return winner, self.centroids[winner]


if __name__ == '__main__':
    data = np.load('./test/latent.npy')
    label = np.load('./test/label.npy')

    num_data = data.shape[0]

    CRN = ReasoningModule()

    num_clusters_CRN = CRN.num_clusters
    print("Number of clusters: {}".format(num_clusters_CRN))

    cluster_count = [0 for i in range(num_clusters_CRN)]
    label_history = [[0, 0, 0] for i in range(num_clusters_CRN)]

    CRN_R = []
    for i in range(num_data):
        CRN_clus_id, CRN_latent = CRN.reason(data[i])
        cluster_count[CRN_clus_id] += 1
        label_history[CRN_clus_id][label[i]] += 1
        CRN_R.append(CRN_clus_id)

    # Label 0: Friction 3.0
    # Label 1: Friction 0.1
    # Label 2: Friction 0.5
    print("Label distribution in each cluster")
    for i in range(num_clusters_CRN):
        print("Cluster {} ({} samples): {}".format(i, cluster_count[i], label_history[i]))

    label_data = np.array(label_history)
    label_max = label_data.max(axis=1)
    print("Clustering purity: {} ({}/{})".format(label_max.sum()/num_data, label_max.sum(), num_data))
