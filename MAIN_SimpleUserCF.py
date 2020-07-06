# -*- coding: utf-8 -*-
from DataLoader import DataLoader
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
import pandas as pd


# +
def user_based_rec_loader(data, testUser, no_recs):
    trainSet = data.build_full_trainset()
    sim_options = {'name': 'cosine',
               'user_based': True
               }
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainSet)

    similarity_matrix = model.compute_similarities()

    testUserInnerID = trainSet.to_inner_uid(testUser)
    similiarty_row = similarity_matrix[testUserInnerID]

    # removing the testUser from the similiarty_row
    similarUsers = []
    for innerID, score in enumerate(similiarty_row):
        if (innerID != testUserInnerID):
            similarUsers.append( (innerID, score) )
#     # find the k users largest similarities
#     k = 10
#     kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

#     or can tune for ratings > threshold
    kNeighbors = []
    for rating in similarUsers:
       if rating[1] > 0.8:
           kNeighbors.append(rating)

    
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

    # Build a dictionary of stuff the user has already seen
    excluded = {}
    for itemID, rating in trainSet.ur[testUserInnerID]:
        excluded[itemID] = 1
   
    # Build a dictionary for results
    results = {'book_title': [], 'rating_sum': []}
            
    # Get top-rated items from similar users:
    print('\n')
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if itemID not in excluded:
            bookID = trainSet.to_raw_iid(itemID)
#             print(ml.getItemName(int(bookID)), ratingSum)
            results['book_title'].append(ml.getItemName(int(bookID)))
            results['rating_sum'].append(ratingSum)
            pos += 1
            if (pos > no_recs -1):
                break
                
    return pd.DataFrame(results)
# -

# # change project variables

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

# +
ratings = pd.read_csv(ratings_path)
print('shape of original ratings was: ', ratings.shape)
ratings = ratings[:size_of_data]
print('shape of ratings is now: ', ratings.shape)
items = pd.read_csv(items_path)

# items[[itemID_column, itemName_column]] as we are only trying to view the item name here

result = pd.merge(ratings, items[[itemID_column, itemName_column]], how='left', on=[itemID_column])
merged_data = result[[userID_column, itemID_column, itemName_column, ratings_column]]
# -

# # choose a user and the numeber of recommendations wanted

testUser = 78

merged_data[merged_data['user_id'] == testUser].sort_values(by=['rating'], ascending =False).head(20)

# # load data

# Load our data set and compute the user similarity matrix
ml = DataLoader(items_path, ratings_path, userID_column, itemID_column, ratings_column, itemName_column, size_of_data)
data = ml.loadData(rating_scale_min, rating_scale_max)

# # run for recommendations

user_based_rec_loader(data, testUser, 10)

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

# # load new user and recommendations

data = ml.addUserLoadData(new_rows, rating_scale_min, rating_scale_max)

user_based_rec_loader(data, mockUserID, 10)


