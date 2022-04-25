#import libraries
from turtle import shape
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

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

#merge dataframes with primary_key 'artist_id' to get genre
allDf = pd.merge(left=df, right=artistsDf, how='outer', on='artist_id')

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
# sns.pairplot(data=allDf, y_vars=['log10_song_pop'], x_vars=['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',\
#     'speechiness', 'valence'])
# plt.show()

#from the visualization a pattern isn't present between song features and song popularity....

sns.pairplot(data=allDf, vars=['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence'])
plt.show()

#check correlation matrix
sns.heatmap(accFeaDf.corr())
plt.show()

#seems like something is going on with energy and loudness
sns.lmplot(x='energy', y='loudness', data=allDf)
plt.show()

#seems like there is a positive association between energy and loudness.... let's check
#reshape data for single predictor model
X = accFeaDf['energy'].values.reshape(-1,1)
y = accFeaDf['loudness'].values.reshape(-1,1)

#check shape
X.shape
y.shape

#split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.4)

#identify and fit the model
lm = LinearRegression()
lm.fit(X=X_train, y=y_train)

#check intercept and coefficient
print(lm.intercept_)
print(lm.coef_)

#check results
y_pred = lm.predict(X_test)

results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten(),})
results

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='r', linewidth=2)
plt.show()

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))