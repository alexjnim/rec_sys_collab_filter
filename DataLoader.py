import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np
import pandas as pd

class DataLoader:

    def __init__(self, items_path, ratings_path, userID_column, itemID_column, ratings_column, itemName_column, size_of_data):
        self.itemsPath = items_path
        self.ratingsPath = ratings_path
        self.itemID_column = itemID_column
        self.userID_column = userID_column
        self.ratings_column = ratings_column
        self.itemName_column = itemName_column
        self.size_of_data = size_of_data

    def loadData(self, rating_scale_min, rating_scale_max):
        self.items_df = pd.read_csv(self.itemsPath)

        self.ratings_df = pd.read_csv(self.ratingsPath)
        self.ratings_df = self.ratings_df[:self.size_of_data]
        self.ratings_df = self.ratings_df[[self.userID_column, self.itemID_column, self.ratings_column]]

        reader = Reader(rating_scale=(rating_scale_min, rating_scale_max))
        self.ratingsDataset = Dataset.load_from_df(self.ratings_df, reader)

        return self.ratingsDataset

    def addUserLoadData(self, new_rows, rating_scale_min, rating_scale_max):
        self.items_df = pd.read_csv(self.itemsPath)

        self.ratings_df = pd.read_csv(self.ratingsPath)
        self.ratings_df = self.ratings_df[:self.size_of_data]
        self.ratings_df = pd.concat([self.ratings_df, new_rows], ignore_index=True)

        self.ratings_df = self.ratings_df[[self.userID_column, self.itemID_column, self.ratings_column]]

        reader = Reader(rating_scale=(rating_scale_min, rating_scale_max))
        self.ratingsDataset = Dataset.load_from_df(self.ratings_df, reader)

        return self.ratingsDataset

    def getUserRatings(self, user):
        userRatings = self.ratings_df[self.ratings_df[self.userID_column] == 1][[self.itemID_column, self.ratings_column]]
        return userRatings

    def getPopularityRanks(self):
        popularity_rankings = self.ratings_df[self.itemID_column].value_counts()
        rankings = pd.Series(range(1, len(popularity_rankings) +1, 1), index = popularity_rankings.index)
        return rankings

    def getItemName(self, itemID):
        if itemID in list(self.items_df[self.itemID_column]):
            return self.items_df[self.itemName_column][self.items_df[self.itemID_column] == itemID].iloc[0]
        else:
            return "Not available"

    def getItemID(self, itemName):
        if itemName in list(self.items_df[self.itemName_column]):
            return self.items_df[self.itemID_column][self.items_df[self.itemName_column] == itemName].iloc[0]
        else:
            return "Not available"
