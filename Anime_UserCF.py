# -*- coding: utf-8 -*-
from DataLoader import DataLoader
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
import pandas as pd

# +
items_path = '../data/anime-rec-data/anime.csv'
ratings_path = '../data/anime-rec-data/rating.csv'
itemID_column = 'anime_id'
userID_column = 'user_id'
ratings_column = 'rating'
itemName_column = 'name'
rating_scale_min = 1
rating_scale_max = 10

# please check how large your ratings.csv is, the larger it is the longer it'll take to run!
# 5 million entries is far too much!
size_of_data = 100000


# +
ratings = pd.read_csv(ratings_path)
print('shape of original ratings was: ', ratings.shape)
ratings = ratings[:size_of_data]
print('shape of ratings is now: ', ratings.shape)
items = pd.read_csv(items_path)

result = pd.merge(ratings, items[[itemID_column, itemName_column]], how='left', on=[itemID_column])
merged_data = result[[userID_column, itemID_column, itemName_column, ratings_column]]
# -

testUser = 30
k = 10

merged_data[merged_data['user_id'] == testUser].sort_values(by=['rating'], ascending =False)[:40]

# Load our data set and compute the user similarity matrix
ml = DataLoader(items_path, ratings_path, userID_column, itemID_column, ratings_column, itemName_column, size_of_data)
data = ml.loadData(rating_scale_min, rating_scale_max)

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

simsMatrix.shape

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testUser)
similarityRow = simsMatrix[testUserInnerID]

# removing the testUser from the similarityRow
similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

# find the k users largest similarities
kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# +
# Get the stuff the k users rated, and add up ratings for each item, weighted by user similarity

# candidates will hold all possible items(movies) and combined rating from all k users
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    # this will hold all the items they've rated and the ratings for each of those items
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
# -

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if itemID not in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getItemName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 8):
            break


# # create a new user and get recommendations
#
# here I'm loading a new user with ratings for selected books

# +
k=10
testItemID = 485
mockUserID = 0
max_rating = ratings['rating'].max()

selected_items = [485, 592, 1041, 479, 95, 4106]
selected_ratings = []
#can manually input the ratings per item
#selected_ratings = [5, 5, 5, 5, 5, 4]

# +
# this code will build new dataframe with the new items and ratings
# the default user will be 0

new_rows = {userID_column:[], itemID_column:[], ratings_column:[]}
if selected_ratings or len(selected_ratings) != 0:
    for num, values in enumerate(zip(selected_items, selected_ratings)):
        new_rows[userID_column].append(mockUserID)
        new_rows[itemID_column].append(values[0])
        new_rows[ratings_column].append(values[1])
else:
    for values in enumerate(selected_items):
        new_rows[userID_column].append(mockUserID)
        new_rows[itemID_column].append(values[1])
        new_rows[ratings_column].append(max_rating)

new_rows = pd.DataFrame(new_rows)
# -

new_rows

ml = DataLoader(items_path, ratings_path, userID_column, itemID_column, ratings_column, itemName_column, size_of_data)
data = ml.addUserLoadData(new_rows, rating_scale_min, rating_scale_max)

trainSet = data.build_full_trainset()


sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

simsMatrix.shape

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(mockUserID)
similarityRow = simsMatrix[testUserInnerID]

# removing the testUser from the similarityRow
similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

# find the k users largest similarities
kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# +
# Get the stuff the k users rated, and add up ratings for each item, weighted by user similarity

# candidates will hold all possible items(movies) and combined rating from all k users
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    # this will hold all the items they've rated and the ratings for each of those items
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
# -

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if itemID not in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getItemName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 8):
            break
