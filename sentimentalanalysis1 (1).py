#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = input("Enter the word: ")
unique = a.capitalize()
unique2 = a.lower()
hashtag = '#'+unique2
u_refs=[]
for i in range(0,7):
    lst=input("Enter the word refs: ")
    u_refs.append(lst)
custom_stopwords = ['RT', hashtag]
xaxis = unique + ' 10 Tweet Moving Average Polarity'
top = unique + ' analysis'


# # 1. Authenticate to Twitter

# In[2]:


import tweepy as tw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[3]:


consumer_key = 'UDxW5m3yt2EcXUTP3vaeS6NQU'
consumer_secret = '7ewX7yk5NZLnfFC4M3ZvAzH0jOpqHOje04xF7LYBrxzx3hd5V2'
access_token = '1396143018306985992-Le27CYDdsR5GiAEQxwRgJFfthYv2U0'
access_token_secret = 'NpZkQsD7Ah0hZNexXCyAKW8DEWLbYOy9exZswYrmRRrlf'


# In[4]:


# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# # 2. Get Tweets

# In[5]:


query = tw.Cursor(api.search, q=hashtag).items(1000)
tweets = [{'Tweets':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]


# In[6]:


df = pd.DataFrame.from_dict(tweets)


# In[9]:


def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) !=-1:
            flag = 1
    return flag

df[unique] = df['Tweets'].apply(lambda x: identify_subject(x,u_refs))


# # 3. Preprocesses

# In[11]:


#import stopwords
import nltk
from nltk.corpus import stopwords

#import textblob
from textblob import Word, TextBlob


# In[12]:


nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
#custom_stopwords = ['RT', '#DogeCoin']


# In[13]:


def preprocess_tweets(tweet, custom_stopwords):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]', '')
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in stop_words)
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in custom_stopwords)
    preprocessed_tweet = " ".join([Word(word).lemmatize() for word in preprocessed_tweet.split()])
    return(preprocessed_tweet)
df['Processed Tweet'] = df['Tweets'].apply(lambda x: preprocess_tweets(x, custom_stopwords))


# # 4. Calculate Sentiment

# In[14]:


df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])


# In[16]:


display(df[df[unique]==1][[unique, 'polarity', 'subjectivity']].groupby(unique).agg([np.mean, np.max, np.min, np.median]))


# # 5. Visualise

# In[17]:


unique2 = df[df[unique]==1][['Timestamp', 'polarity']]
unique2 = unique2.sort_values(by='Timestamp', ascending=True)
unique2['MA Polarity'] = unique2.polarity.rolling(10, min_periods=3).mean()


# In[18]:


#unique2.head()


# In[19]:


plt.plot(unique2['Timestamp'], unique2['MA Polarity'])
plt.title("\n".join([xaxis]))
plt.suptitle('/n'.join([top]), y=0.98)
plt.show()

