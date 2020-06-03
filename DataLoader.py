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

    def getGenres(self):
        #this will store the genres for each film
        genres = defaultdict(list)
        #this will store the keys for each genre
        genreIDs = {}
        #this will track the maximum number of IDs
        maxGenreID = 0

        for i in range(len(self.items_df)):
            row = self.items_df.iloc[i]
            itemID = row[0]
            genreList = row[2].split('|')
            genreIDList = []
            for genre in genreList:
                #if the genre is already listed, it will append to genreIDList
                if genre in genreIDs:
                    genreID = genreIDs[genre]
                #if the genre isn't listed yet, it will be listed and assigned a genreID before being appended
                else:
                    genreID = maxGenreID
                    genreIDs[genre] = genreID
                    maxGenreID += 1
                genreIDList.append(genreID)
            genres[itemID] = genreIDList

        # this will effectively do one-hot-encoding to all the genres
        # and return a list with 0's and 1's for each itemID
        for (itemID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[itemID] = bitfield

        return genres

    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)

        for i in range(len(self.items_df)):
            row = self.items_df.iloc[i]
            itemID = row[0]
            title = row[1]
            m = p.search(title)
            year = m.group(1)
            if year:
                years[itemID] = int(year)

        return years

    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                itemID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[itemID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes

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
