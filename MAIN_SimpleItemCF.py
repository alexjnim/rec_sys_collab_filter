# -*- coding: utf-8 -*-
from DataLoader import DataLoader
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
import pandas as pd


def item_based_rec_loader(data, testUser, no_recs):

    trainSet = data.build_full_trainset()
    
    # note that user_base: False here, thus we are telling KNN that 
    # we want to generate an item-item based similarity matrix
    sim_options = {'name': 'cosine',
                   'user_based': False
                   }
    
    
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainSet)

    similarity_matrix = model.compute_similarities()

    testUserInnerID = trainSet.to_inner_uid(testUser)
    # Get the top K items we rated
#     k = 10
#     testUserRatings = trainSet.ur[testUserInnerID]
#     kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

    #or look for items with rating > threshold
    kNeighbors = []
    testUserRatings = trainSet.ur[testUserInnerID]
    for rating in testUserRatings:
       if rating[1] > 4.0:
           kNeighbors.append(rating)
    
    # Get similar items to stuff we liked (weighted by rating)
    candidates = defaultdict(float)
    for itemID, rating in kNeighbors:
        similarity_row = similarity_matrix[itemID]
        for innerID, score in enumerate(similarity_row):
            candidates[innerID] += score * (rating / 5.0)

    # Build a dictionary of stuff the user has already seen
    excluded = {}
    for itemID, rating in trainSet.ur[testUserInnerID]:
        excluded[itemID] = 1
    
    # Build a dictionary for results
    results = {'book': [], 'rating_sum': []}
    
    # Get top-rated items from similar users:
    print('\n')
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in excluded:
            bookID = trainSet.to_raw_iid(itemID)
#             print(ml.getItemName(int(bookID)), ratingSum)
            results['book'].append(ml.getItemName(int(bookID)))
            results['rating_sum'].append(ratingSum)
            pos += 1
            if (pos > no_recs -1):
                break
    return pd.DataFrame(results)

items_path = '../data/goodbooks-10k-master/books.csv'
ratings_path = '../data/goodbooks-10k-master/ratings.csv'
itemID_column = 'book_id'
userID_column = 'user_id'
ratings_column = 'rating'
itemName_column = 'title'
rating_scale_min = 1
rating_scale_max = 5

# +
# here we don't need to worry too much about the size of ratings.csv even though its 5 million rows!
# with item-based CF, the dimensions of the similarity matrix will be the size of number of unique items, here 10,000
ratings = pd.read_csv(ratings_path)
items = pd.read_csv(items_path)

result = pd.merge(ratings, items[[itemID_column, itemName_column]], how='left', on=[itemID_column])
merged_data = result[[userID_column, itemID_column, itemName_column, ratings_column]]
# -

testUser = 78
merged_data[merged_data['user_id'] == testUser].sort_values(by=['rating'], ascending =False).head(20)

# Load our data set and compute the user similarity matrix
ml = DataLoader(items_path, ratings_path, userID_column, itemID_column, ratings_column, itemName_column, size_of_data)
data = ml.loadData(rating_scale_min, rating_scale_max)

item_based_rec_loader(data, testUser, 10)

# # select a single item and find items similar to that
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

result = pd.merge(new_rows, items[[itemID_column, itemName_column]], how='left', on=[itemID_column])
result

# +
data = ml.addUserLoadData(new_rows, rating_scale_min, rating_scale_max)

item_based_rec_loader(data, mockUserID, 10)
# -


