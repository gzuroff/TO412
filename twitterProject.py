import tweepy
import pandas as pd
import numpy as np
from scipy import stats 
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import re
import nltk
from operator import itemgetter

def getFollowers(api, username):
    user = api.get_user(username)
    return user.followers_count

def getMention(tweet):
    x = re.findall(r'@(\w+)', tweet)
    return x[0]

#Functions for neural net sentiment analysis
def prepareSentence(s):
    stemmer = LancasterStemmer()
    ignore_words = set(stopwords.words('english'))
    regpattern = re.compile('[\W_]+" "')
    s = re.sub('[^A-Za-z ]+', '', s)
    words = nltk.word_tokenize(s.lower())
    return [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
#Converts sentence to bag of words
def toBOW(sentance, words):
    bag = []
    for word in words:
        bag.append(1) if word in sentance else bag.append(0)
    return bag

key = "Pfv5AqNYvf2dpTAqOiJLRXbsk"
secret = "MZ8jQBF39lpukzQw0mCQ2xEIS1M1XdqZIl7V5MsevhspspIFQs"

comps = ["wendys", "burgerking", "mcdonalds", "arbys"]
auth = tweepy.OAuthHandler(key, secret)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
tweets = {}
df = pd.DataFrame(columns = ["text", "created", "re", "fav", "followC", "sn"])
words = []

for comp in comps:
    df = pd.DataFrame(columns = ["text", "created", "re", "fav", "followC", "sn"])
    compTweets = tweepy.Cursor(api.user_timeline, comp)
    for page in compTweets.pages():
        for tweet in page:
            words.extend(prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")))
            #print(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r",""))
            df = df.append({
                "text": tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r",""),
                "textClean": prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")),
                "created": tweet.created_at.hour,
                "re": tweet.retweet_count,
                "fav": tweet.favorite_count,
                "followC": tweet.user.followers_count,
                "sn": tweet.user.screen_name
            }, ignore_index=True)
    #TODO: basic linear regression model
    df.loc[:,"sn"] = pd.Categorical(df.sn).codes
    tweets[comp] = df
trainComps = ["jetblue", "innocent", "mlbgifs", "digiornopizza", "generalelectric", "charmin", "mit", "linkedinhelp"]
trainData = {}

for comp in trainComps:
    compTweets = tweepy.Cursor(api.user_timeline, comp)
    df = pd.DataFrame(columns = ["text", "re", "fav"])
    for page in compTweets.pages():
        for tweet in page:
            words.extend(prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")))
            df = df.append({
                "text": prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")),
                "re": tweet.retweet_count,
                "fav": tweet.favorite_count
            }, ignore_index=True)
    trainData[comp] = df

#Creating words for bag of words
distinctWords = set(words)
lower_threshold = 10
upper_threshold = 350
counts = []
final_words = []
for word in distinctWords:
    counts.append(words.count(word))
    if words.count(word) > lower_threshold and words.count(word) < upper_threshold: 
        final_words.append(word)
inputs = {}
outputs = {}
trainInputs = []
trainOutputs = []
for comp in comps:
    iputs = []
    oputs = []
    for index, row in tweets[comp].iterrows():
        bag = toBOW(row["textClean"], final_words)
        iputs.append(bag)
        score = row["fav"] + row["re"]
        oputs.append(score)
    inputs[comp] = iputs
    outputs[comp] = oputs
for comp in trainComps:
    for index, row in trainData[comp].iterrows():
        bag = toBOW(row["text"], final_words)
        trainInputs.append(bag)
        score = row["fav"] + row["re"]
        trainOutputs.append(score)

trainOutputs = stats.zscore(trainOutputs)
outputs = stats.zscore(outputs)
print("training nn")
nnet = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=(int(len(final_words)*0.5),int(len(final_words)*0.25)),solver='adam', max_iter=400)
nnet.fit(trainInputs, trainOutputs)
print("Finished Training NN")
#Add sentiment score to fast food comps dataframes
for comp in comps:
    tweets[comp]["sent"] = nnet.predict(inputs[comp])

#Parse tweets for info like @s, #s, pics, links
ats = []
atsFollowers = []
punds = []
pics = []
links = []

for comp in comps:
    for index, tweet in tweets[comp].iterrows():
        ats.append(1 if "@" in tweet['text'] else 0)
        atsFollowers.append(
            0 if "@" not in tweet['text'] else getFollowers(api, getMention(tweet['text']))
        )
        