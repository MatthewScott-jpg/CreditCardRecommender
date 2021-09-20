#!/usr/bin/env python

"""clusterBuilder.py: Given a preprocessed dataframe, optimizes the DBSCAN algorithm based on silhouette scores and returns
a full-scale dataframe with cluster labels
"""

import pandas as pd
import numpy as np
import math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def clusterBuilder(X, X_scaled):
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    start = round(math.ceil(distances.mean()*10))/10
    end = round(math.ceil(distances.max()*10))/10

    best_eps,score = 0,0
    for eps_try in np.arange(start,end+.1,.1):
        temp_db = DBSCAN(eps=eps_try, min_samples=5).fit(X_scaled)
        labels = temp_db.labels_
        silhouette_avg = silhouette_score(X_scaled,labels)
        if silhouette_avg > score:
            score = silhouette_avg
            best_eps = eps_try

    db = DBSCAN(eps=best_eps, min_samples=5).fit(X_scaled)
    labels = db.labels_
    X['DBSCAN_Labels'] = labels
    
    return X