from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from typing import List, Dict
import os
import argparse
import random
import json
import sklearn

# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers

from data import read_instances,generate_batches,get_fake_sample
from util import load_pretrained_model
from main_model import MainEncoderDecoder
from probing_model import ProbingEncoderDecoder


class KMeansClustering:

    def cluster_hidden_states(hidden_states):
        # scoring_metrics_K_Means_Clustering = {'Silhouette Coefficient': make_scorer(sklearn.metrics.silhouette_score)}
           # , 'Davies-Bouldin Index': make_scorer(sklearn.metrics.davies_bouldin_score)}
        #number_of_clusters_hyperparameter = range(5, 13, 1)
        #param_grid = {'n_clusters': number_of_clusters_hyperparameter}
        km = KMeans(n_clusters = 8)
        #gridSearchCV = GridSearchCV(km, param_grid=param_grid, scoring=None, return_train_score=True,refit='Silhouette Coefficient')
        #print(hidden_states.numpy())
        gridSearchResults = km.fit(hidden_states.numpy())
        silhouette_score_for_k_means_clustering =  silhouette_score(hidden_states.numpy(), gridSearchResults.labels_, metric='euclidean')
        #print(gridSearchResults.cv_results_)
        #print(gridSearchResults.best_estimator_.labels_)
        return silhouette_score_for_k_means_clustering,gridSearchResults.labels_,km
        #print(km.labels_)