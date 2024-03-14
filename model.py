import pickle
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob


def create_features(df, map1, map2, map3, map4, map5, map6, map7, map8, map9, map10, map11, map12, map13, map14, map15):
    # Creating new features based on the title and description
    df['title_chars'] = df['title'].str.len()  # Number of characters in the title
    df['description_words'] = df['description'].str.count(' ').add(1)  # Number of words in the description
    
    # Calculating the age of the video in seconds
    df['video_age_seconds'] = (pd.to_datetime(df['trending_date'], utc = True) - pd.to_datetime(df['publishedAt'])).dt.total_seconds().astype('int')
    
    # Converting the duration from seconds to 5-minute intervals
    df['duration_5minutes'] = df['duration_seconds']/(60*5)
    df['duration_5minutes'] = df['duration_5minutes'].apply(np.floor)

    # Converting boolean features to binary
    df['ratings_disabled'] = np.where(df['ratings_disabled'] == True, 1, 0)
    df['comments_disabled'] = np.where(df['comments_disabled'] == True, 1, 0)
    df['has_thumbnail'] = np.where(df['has_thumbnail'] == True, 1, 0)
    
    # More features based on the title
    df['title_words'] = df['title'].str.count(' ').add(1)  # Number of words in the title
    df['title_avg_chars_word'] = df['title_chars']/df['title_words']  # Average number of characters per word in the title
    df[['title_sentiment_polarity', 'title_sentiment_subjectivity']] = df['title'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))  # Sentiment analysis of the title
    df['title_uppercases'] = df['title'].str.findall(r'[A-Z]').str.len()/df['title_chars']  # Ratio of uppercase letters in the title
    df['title_lowercases'] = df['title'].str.findall(r'[a-z]').str.len()/df['title_chars']  # Ratio of lowercase letters in the title
    df['title_hashtag_count'] = df['title'].str.count('#')  # Number of hashtags in the title
    df['title_hashtag_count'] = df['title_hashtag_count'].fillna(0)  # Fill NA values with 0
    
    # Features based on the tags
    df['has_tag'] = np.where(df['tags'] == '[None]', 0, 1) 
    df['tags_count'] = df['tags'].str.count('|').add(1)  
    
    # Extracting the month from the published date
    df['published_at_month'] = pd.DatetimeIndex(pd.to_datetime(df['publishedAt'])).month
    
    # Popularity features based on the category and channel
    df['category_popularity'] = df['categoryId'].map(map1)
    df['channel_popularity'] = df['channelTitle'].map(map2)
    
    # Extracting the weekday from the trending date
    df['trending_weekday'] = pd.to_datetime(df['trending_date']).dt.dayofweek.add(1)
    
    # More features based on the description
    df['description_chars'] = df['description'].str.len()  
    df['description_chars'] = df['description_chars'].fillna(0)  
    df['description_words'] = df['description'].str.count(' ').add(1) 
    df['description_words'] = df['description_words'].fillna(0)  
    df['description_avg_chars_word'] = df['description_chars']/df['description_words']  
    df[['description_sentiment_polarity', 'description_sentiment_subjectivity']] = df['description'].astype('str').apply(lambda Text2: pd.Series(TextBlob(Text2).sentiment))  
    df['description_link'] = np.where(df['description'].str.contains("http://", regex = False), 1, 0)  
    df['description_exclamation'] = df['description'].str.count('!') 
    df['description_exclamation'] = df['description_exclamation'].fillna(0) 
    df['description_hashtag_count'] = df['description'].str.count('#') 
    df['description_hashtag_count'] = df['description_hashtag_count'].fillna(0)  
    
    # Calculating the age of the video in seconds again
    df['video_age_seconds'] = (pd.to_datetime(df['trending_date'], utc = True) - pd.to_datetime(df['publishedAt'])).dt.total_seconds().astype('int')
    
    # Mean duration features based on the category and channel
    df['category_mean_duration_seconds'] = df['categoryId'].map(map3)
    df['channel_mean_duration_seconds'] = df['channelTitle'].map(map4)
    
    # Mean title characters features based on the category and channel
    df['category_mean_title_chars'] =  df['categoryId'].map(map5)
    df['channel_mean_title_chars'] =  df['channelTitle'].map(map6)
    
    # Mean description words features based on the category and channel
    df['category_mean_description_words'] =  df['categoryId'].map(map7)
    df['channel_mean_description_words'] =  df['channelTitle'].map(map8)
    
    # Mean video age features based on the category and channel
    df['category_mean_video_age_seconds'] =  df['categoryId'].map(map9)
    df['channel_mean_video_age_seconds'] =  df['channelTitle'].map(map10)
    
    # Mean target features based on the category and channel
    df['category_mean_target_since_august'] =  df['categoryId'].map(map11)
    df['channel_mean_target_since_august'] =  df['channelTitle'].map(map12)
    
    # Mean likes ratio features based on the category and channel
    df['category_mean_likes_ratio_since_august'] =  df['categoryId'].map(map13)
    df['channel_mean_likes_ratio_since_august'] =  df['channelTitle'].map(map14)
    
    # Converting the duration from seconds to 5-minute intervals again
    df['duration_5minutes'] = df['duration_5minutes']/(60*5)
    df['duration_5minutes'] = df['duration_5minutes'].apply(np.floor).astype(str)
    
    # Mean target feature based on the duration
    df['duration_5minutes_mean_target'] =  df['duration_5minutes'].map(map15)

    return df



def main(df):
    # Load the maps
    with open('maps.pkl', 'rb') as f:
        map1, map2, map3, map4, map5, map6, map7, map8, map9, map10, map11, map12, map13, map14, map15 = pickle.load(f)

        # Load the model
        model = joblib.load('model.pkl')

        # Create features and select columns
        new = create_features(df, map1, map2, map3, map4, map5, map6, map7, map8, map9, map10, map11, map12, map13, map14, map15)
        new = new[['duration_seconds', 'title_chars', 'title_words', 'title_avg_chars_word', 'title_sentiment_polarity', 'title_sentiment_subjectivity',
                    'title_uppercases', 'title_lowercases', 'tags_count', 'published_at_month', 'category_popularity', 'channel_popularity','description_chars', 
                    'description_words', 'description_avg_chars_word', 'description_sentiment_polarity', 'description_sentiment_subjectivity','description_exclamation',
                    'description_hashtag_count', 'video_age_seconds', 'category_mean_duration_seconds', 'channel_mean_duration_seconds', 'category_mean_title_chars',
                    'channel_mean_title_chars','category_mean_description_words', 'channel_mean_description_words', 'category_mean_video_age_seconds',
                    'channel_mean_video_age_seconds', 'category_mean_target_since_august', 'channel_mean_target_since_august', 'category_mean_likes_ratio_since_august',
                    'channel_mean_likes_ratio_since_august']]

        # Make prediction
        prediction = model.predict(new)
        return prediction
