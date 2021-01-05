# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from ProcessData import ProcessData
import pandas as pd


# # functions

def generate_sim_matrix(trainSet, sim_metric, is_user=True):
    sim_options = {'name': sim_metric,
                   'user_based': is_user
                   }
    model = KNNBasic(sim_options=sim_options, verbose=False)
    model.fit(trainSet)
    simsMatrix = model.compute_similarities()
    return simsMatrix


def get_model(model_name, sim_options):
    if model_name == 'KNNBasic':
        model = KNNBasic(sim_options=sim_options, verbose=False)
    elif model_name == 'KNNWithMeans':
        model = KNNWithMeans(sim_options=sim_options, verbose=False)
    elif model_name == 'KNNWithZScore':
        model = KNNWithZScore(sim_options=sim_options, verbose=False)
    elif model_name == 'KNNBaseline':
        model = KNNBaseline(sim_options=sim_options, verbose=False)
    return model


def generate_sim_matrix(trainSet, model_name, sim_metric, is_user=True):
    sim_options = {'name': sim_metric,
                   'user_based': is_user
                   }
    model = get_model(model_name, sim_options)
    model.fit(trainSet)
    simsMatrix = model.compute_similarities()
    return simsMatrix


def get_hit_rates(trainSet, testSet, simsMatrix, num_nearest_neighbors=10):
    # Build up dict to lists of (int(movieID), predictedrating) pairs
    topN = defaultdict(list)
    k = num_nearest_neighbors
    for uiid in range(trainSet.n_users):
        # Get top N similar users to this one
        similarityRow = simsMatrix[uiid]

        similarUsers = []
        for innerID, score in enumerate(similarityRow):
            if (innerID != uiid):
                similarUsers.append( (innerID, score) )

        kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

        # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for similarUser in kNeighbors:
            innerID = similarUser[0]
            userSimilarityScore = similarUser[1]
            theirRatings = trainSet.ur[innerID]
            for rating in theirRatings:
                candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

        # Build a dictionary of stuff the user has already seen
        watched = {}
        for itemID, rating in trainSet.ur[uiid]:
            watched[itemID] = 1

        # Get top-rated items from similar users:
        pos = 0
        for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not itemID in watched:
                movieID = trainSet.to_raw_iid(itemID)
                topN[int(trainSet.to_raw_uid(uiid))].append( (int(movieID), 0.0) )
                pos += 1
                if (pos > 40):
                    break
    HR = RecommenderMetrics.HitRate(topN, testSet)
    CHR = RecommenderMetrics.CumulativeHitRate(topN, testSet, ratingCutoff=4.0)
    RHR = RecommenderMetrics.RatingHitRate(topN, testSet)
    ARHR = RecommenderMetrics.AverageReciprocalHitRank(topN, testSet)
    return HR, CHR, RHR, ARHR


# # load data

# +
items_path = '../data/goodbooks-10k-master/books.csv'
ratings_path = '../data/goodbooks-10k-master/ratings.csv'
itemID_column = 'book_id'
userID_column = 'user_id'
ratings_column = 'rating'
itemName_column = 'title'
rating_scale_min = 1
rating_scale_max = 5

# please check how large your ratings.csv is, the larger it is the longer it'll take to run!
# 5 million entries is far too much!
size_of_data = 100000
ratings = pd.read_csv(ratings_path)
ratings = ratings[:size_of_data]
items = pd.read_csv(items_path)

result = pd.merge(ratings, items[[itemID_column, itemName_column]], how='left', on=[itemID_column])
merged_data = result[[userID_column, itemID_column, itemName_column, ratings_column]]
# -

len(ratings['user_id'].unique())

# # process data

# +
from surprise import Dataset
from surprise import Reader

data = merged_data[['user_id', 'book_id', 'rating']]
reader = Reader(rating_scale=(0.5, 4.5))
data = Dataset.load_from_df(data, reader)

popularity_rankings = merged_data['book_id'].value_counts()
rankings = pd.Series(range(1, len(popularity_rankings) +1, 1), index = popularity_rankings.index)

processed_data = ProcessData(data, rankings)
# Train on leave-One-Out train set
trainSet = processed_data.GetLOOCVTrainSet()
testSet = processed_data.GetLOOCVTestSet()
# -

# # run test 

# +
user_scores_df = pd.DataFrame(columns = ["Hit Rate", 'Cumulative Hit Rate', 'Rating Hit Rate', 'Average Reciprocal Hit Rank'])
metrics = ['cosine', 'msd', 'pearson', 'pearson_baseline']

for metric in metrics:
    temp_list = []
    
    sim_matrix = generate_sim_matrix(trainSet, 'KNNBasic', metric, is_user=True)
    HR, CHR, RHR, ARHR = get_hit_rates(trainSet, testSet, sim_matrix, num_nearest_neighbors=10)
    
    temp_list.append(HR)
    temp_list.append(CHR)
    temp_list.append(RHR)
    temp_list.append(ARHR)
    user_scores_df.loc[metric] = temp_list

user_scores_df
# -

# with different models

# + active=""
# user_scores_df = pd.DataFrame(columns = ["Hit Rate", 'Cumulative Hit Rate', 'Rating Hit Rate', 'Average Reciprocal Hit Rank'])
#
# model_names = ['KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'KNNBaseline']
# # model_names = ['KNNBasic', 'KNNBaseline']
# metrics = ['cosine', 'msd', 'pearson', 'pearson_baseline']
#
# for model_name in model_names:
#     for metric in metrics:
#         temp_list = []
#         sim_matrix = generate_sim_matrix(trainSet, model_name, metric, is_user=True)
#         HR, CHR, RHR, ARHR = get_hit_rates(trainSet, testSet, sim_matrix, num_nearest_neighbors=10)
#         
#         temp_list.append(HR)
#         temp_list.append(CHR)
#         temp_list.append(RHR)
#         temp_list.append(ARHR)
#         user_scores_df.loc[model_name + " " + metric] = temp_list
#
# user_scores_df
# -


