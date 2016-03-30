import numpy as np
import pandas as pd

data = pd.read_csv('data/tags.csv')

# Convert tags to upper
data['tag'] = data['tag'].str.upper()

# GRoup movies n tags to remove duplicate tags
tmp = data.groupby(['movieId','tag'], as_index=False)['userId'].count()

mean_counts = tmp.groupby(['movieId']).userId.mean().reset_index()

merge_tmp_means = pd.merge(tmp, mean_counts, on='movieId',suffixes=('_count','_mean'))
merge_tmp_means.head()
tmp_x = merge_tmp_means[merge_tmp_means.userId_count >= merge_tmp_means.userId_mean]

# combine tags into one
tmp = tmp_x.groupby('movieId')['tag'].apply(','.join).reset_index()

import nltk
from nltk.corpus import stopwords

tmp['tag'] = tmp['tag'].apply(lambda x: x.split(',') ) # Tokenize


tmp['tag'] = tmp['tag'].apply(lambda x: [word for word in x if word not in stopwords.words('english')] ) # REmove stop words
# Lemmanize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

iod = tmp['tag'].apply(lambda  x: set([ wordnet_lemmatizer.lemmatize(word.decode('utf-8').strip()) for word in x]) )

tmp['tag'] = iod


tmp.to_csv('data/tags_cleaned.csv', index=False)



movies= pd.read_csv('data/movies.csv')
movies.head()
import re
def funct(x):
    if re.search(r'\((\d{4})\)',x) is None:
        return 0000
    else:
        return re.search(r'\((\d{4})\)',x).group(1)

movies['year'] = movies.title.map(funct)

movies.head()
movies['title_count'] = movies.title.map(len)
def vowel_count(x):
    vowels = "aeiou"
    vowel_sum=0
    for v in vowels:
         vowel_sum+=x.lower().count(v)
    return vowel_sum
movies['title_vowels'] = movies.title.map(vowel_count)

movies.head()


ratings = pd.read_csv('data/ratings.csv')

ratings.head()

ratings = ratings.groupby(['movieId'], as_index=False).rating.mean()

movie_rating = pd.merge(movies, ratings, on = 'movieId')
movie_rating.head()

final = pd.merge(movie_rating, tmp, on='movieId' )

final['tag'] = final.tag.map(list)
final.head()

s = final['tag'].apply(pd.Series,1).stack()
s.index = s.index.droplevel(-1)
s.name = 'tag'
s.head()
del final['tag']
final_2= pd.merge(s.reset_index(), final, left_on='level_0', right_on='movieId')
final_2.head()
final_2 = final_2.drop(['level_0', 'level_1'], axis=1)

final_2.head()

final_2.to_csv('data/final_data.csv', index=False,encoding='utf-8')
final_2.drop(['genres','title','movieId'], axis=1).to_csv('data/final_hadoop_data.csv', index=False,encoding='utf-8')



genres= ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
for i in genres:
    tmp = final_2[final_2.genres.str.contains(i)].drop(['genres','title','movieId'], axis=1)
    tmp = pd.get_dummies(tmp, columns=['tag'])
    tmp.to_csv('genres/final_'+i+".csv", index=False,encoding='utf-8')


import pandas2arff
reload(pandas2arff)
t=pd.get_dummies(final_2.drop(['genres','title','movieId'], axis=1), columns=['tag'])
pandas2arff.pandas2arff(t,"foo.arff")