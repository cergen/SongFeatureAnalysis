#import libraries
from turtle import shape
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# data provided by musicoset link:https://marianaossilva.github.io/DSW2019/

# read data
songsDf = pd.read_csv('musicoset_data\musicoset_metadata\musicoset_metadata\songs.csv', header=None, skiprows=1, sep='\t')
songsDf.columns = ['song_id', 'name', 'billboard', 'artist', 'song_popularity', 'explicit', 'song_type']

accFeaDf = pd.read_csv("musicoset_data\musicoset_songfeatures\musicoset_songfeatures/acoustic_features.csv", header=None, skiprows=1, sep='\t')
accFeaDf.columns = ['song_id', 'duration_ms', 'key', 'mode', 'time_signature', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',\
    'speechiness', 'valence', 'tempo']

artistsDf = pd.read_csv("musicoset_data\musicoset_metadata\musicoset_metadata/artists.csv", header=None, skiprows=1, sep='\t')
artistsDf.columns = ['artist_id', 'name', 'followers', 'artist_popularity', 'artist_type', 'main_genre', 'genres', 'image_url']
artistsDf.drop(['artist_type', 'image_url'], axis=1, inplace=True)

#split artist id & name for songsDf
idArtistsDf = songsDf['artist'].str.split(':', n=1, expand=True)
idArtistsDf = idArtistsDf.replace("[{,',}]", "", regex=True)
idArtistsDf.columns = ['artist_id', 'artist_name']
songsDf = pd.concat([songsDf, idArtistsDf], axis=1)

#Data Overview
songsDf.head()
accFeaDf.head()
artistsDf.head()
songsDf.info()
accFeaDf.info()
songsDf.describe()
accFeaDf.describe()

#merge dataframes with primary_key 'song_id'
df = pd.merge(left=songsDf, right=accFeaDf, how='outer', on='song_id')
df.head()
df.describe()
df.info()

#merge dataframes with primary_key 'artist_id' to get genre
allDf = pd.merge(left=df, right=artistsDf, how='outer', on='artist_id')
allDf.head()
allDf.describe()
allDf.info()

#distribution of song popularity
sns.set_style('whitegrid')
sns.displot(allDf['song_popularity'])
plt.show()

#log transform the skewed data
allDf['log2_song_pop'] = np.log2(allDf['song_popularity'])
allDf['log10_song_pop'] = np.log10(allDf['song_popularity'])
sns.displot(allDf['log10_song_pop'])
sns.displot(allDf['log2_song_pop'])
plt.show()

#pairplot to check for associations
sns.pairplot(data=allDf, y_vars=['log10_song_pop'], x_vars=['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',\
    'speechiness', 'valence'])
plt.show()

#from the visualization a pattern isn't present between song features and song popularity....

sns.pairplot(data=allDf, vars=['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence'])
plt.show()

sns.lmplot(x='energy', y='loudness', data=allDf)
plt.show()

#seems like there is a positive association between energy and loudness.... let's check
X = accFeaDf['energy']
y = accFeaDf['loudness']

sns.displot(X)
plt.show()
sns.displot(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.4)

lm = LinearRegression()
lm.fit(X_train, y_train)