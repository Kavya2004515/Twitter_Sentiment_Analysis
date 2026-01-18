# =====================================
# Twitter Sentiment Analysis
# HEX SOFTWARES - Project 3
# =====================================

import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading tweets dataset...")

# -------------------------------------
# STEP 1: Load Dataset
# -------------------------------------

df = pd.read_csv("tweets.csv")

# Dataset columns:
# 0 = id, 1 = topic, 2 = sentiment, 3 = text
df.columns = ["id", "topic", "sentiment", "text"]

print("Dataset loaded successfully!")
print(df.head())

# -------------------------------------
# STEP 2: Clean Tweet Text
# -------------------------------------

def clean_tweet(tweet):
    tweet = str(tweet)  # ✅ FIX: handles NaN / float values
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"[^a-z\s]", "", tweet)
    return tweet

df["clean_text"] = df["text"].apply(clean_tweet)

# -------------------------------------
# STEP 3: Sentiment Analysis
# -------------------------------------

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["predicted_sentiment"] = df["clean_text"].apply(get_sentiment)

# -------------------------------------
# STEP 4: Show Sample Output
# -------------------------------------

print("\nSample Results:\n")
print(df[["text", "predicted_sentiment"]].head(10))

# -------------------------------------
# STEP 5: Visualization
# -------------------------------------

plt.figure(figsize=(6, 4))
sns.countplot(x="predicted_sentiment", data=df)
plt.title("Twitter Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

# -------------------------------------
# STEP 6: User Input Test
# -------------------------------------

while True:
    tweet = input("\nEnter a tweet (type exit to stop): ")
    if tweet.lower() == "exit":
        print("Program ended.")
        break

    cleaned = clean_tweet(tweet)
    print("Sentiment:", get_sentiment(cleaned))
