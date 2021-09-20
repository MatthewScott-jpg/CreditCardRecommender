#!/usr/bin/env python

"""This module builds clusters and assigns labels to each observation"""

import math
import numpy as np
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

def cluster_builder(x, x_scaled):
    """Assigns cluster labels to points in the data

    Builds an optimized DBSCAN clustering algorithm for the scaled data,
    then returns the original data with cluster labels attatched

    Args:
        x: a preprocessed dataframe containing relevant information
        x_scaled: a scaled version of x
    """
    nneighbors = neighbors.NearestNeighbors(n_neighbors=4)
    neighbors_fit = nneighbors.fit(x_scaled)
    distances, indices = neighbors_fit.kneighbors(x_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    start = round(math.ceil(distances.mean()*10))/10
    end = round(math.ceil(distances.max()*10))/10

    best_eps,score = 0,0
    for eps_try in np.arange(start,end+.1,.1):
        temp_db = cluster.DBSCAN(eps=eps_try, min_samples=5).fit(x_scaled)
        labels = temp_db.labels_
        silhouette_avg = metrics.silhouette_score(x_scaled,labels)
        if silhouette_avg > score:
            score = silhouette_avg
            best_eps = eps_try

    db = cluster.DBSCAN(eps=best_eps, min_samples=5).fit(x_scaled)
    labels = db.labels_
    x['DBSCAN_Labels'] = labels
    print(f'DBSCAN with eps={best_eps} built')

    return x
