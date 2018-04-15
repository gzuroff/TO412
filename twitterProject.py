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
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
analyzer = SentimentIntensityAnalyzer()

checkFor = ["McDonalds", "BurgerKing", "Arbys", "tacobell", "kfc", "DennysDiner", "Subway", "Wendys"]
checkFor = [x.lower() for x in checkFor]

def getFollowers(api, username):
    try:
        user = api.get_user(username)
        return user.followers_count
    except:
        return 0

def getMention(tweet):
    x = re.findall(r'@(\w+)', tweet)
    return 1 if x[0].lower() in checkFor else 0

#Functions for neural net sentiment analysis
def prepareSentence(s):
    stemmer = LancasterStemmer()
    ignore_words = set(stopwords.words('english'))
    regpattern = re.compile('[\W_]+" "')
    s = re.sub('[^A-Za-z ]+', '', s)
    words = nltk.word_tokenize(s.lower())
    return [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
#Converts sentence to bag of words, represents each word as a 1 if it is present in the tweet, 0 if not
def toBOW(sentance, words):
    bag = []
    for word in words:
        bag.append(1) if word in sentance else bag.append(0)
    return bag

key = "Pfv5AqNYvf2dpTAqOiJLRXbsk"
secret = "MZ8jQBF39lpukzQw0mCQ2xEIS1M1XdqZIl7V5MsevhspspIFQs"

comps = ["wendys", "burgerking", "mcdonalds", "arbys"]
trainComps = ["jetblue", "innocent", "mlbgifs", "digiornopizza", "generalelectric", "charmin", "mit", "linkedinhelp"]
auth = tweepy.OAuthHandler(key, secret)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
tweets = {}
df = pd.DataFrame(columns = ["text", "created", "re", "fav", "followC", "sn"])
words = []
try:
    print("trying to read comps CSVs")
    for comp in comps:
        f_in = comp + ".csv"
        df = pd.read_csv(f_in)
        tweets[comp] = df
except:
    print("Fetching Comps tweets from API")
    for comp in comps:
        print("Fectching " + comp + "'s tweets")
        df = pd.DataFrame(columns = ["text", "created", "re", "fav", "followC", "sn", "pic", "polarity", "subjectivity", "analyzer"])
        compTweets = tweepy.Cursor(api.user_timeline, comp)
        for page in compTweets.pages():
            for tweet in page:
                analysis = TextBlob(tweet.text)
                df = df.append({
                    "text": tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r",""),
                    "textClean": prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")),
                    "created": tweet.created_at.hour,
                    "week": pd.to_datetime(tweet.created_at).week,
                    "year": tweet.created_at.year,
                    "re": tweet.retweet_count,
                    "fav": tweet.favorite_count,
                    "followC": tweet.user.followers_count,
                    "inReply": 1 if tweet.in_reply_to_status_id is not None else 0,
                    "sn": tweet.user.screen_name,
                    "pic": 1 if 'media' in tweet.entities else 0,
                    "polarity": analysis.sentiment.polarity,
                    "subjectivity": analysis.sentiment.subjectivity,
                    "analyzer": analyzer.polarity_scores(tweet.text)["compound"],
                    "atCompany": getMention(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")) if "@" in tweet.text is not None else 0
                }, ignore_index=True)
        #TODO: basic linear regression model
        df.loc[:,"sn"] = pd.Categorical(df.sn).codes
        tweets[comp] = df
        f_out = comp + ".csv"
        tweets[comp].to_csv(f_out)
trainData = {}
try:
    print("trying to read top 8 comps tweets from CSV")
    for comp in trainComps:
        f_in = comp + ".csv"
        df = pd.read_csv(f_in)
        trainData[comp] = df
except:
    print("Fetching top 8 Comps tweets from API")
    for comp in trainComps:
        print("Fectching " + comp + "'s tweets")
        compTweets = tweepy.Cursor(api.user_timeline, comp)
        df = pd.DataFrame(columns = ["text", "re", "fav"])
        for page in compTweets.pages():
            for tweet in page:
                df = df.append({
                    "text": prepareSentence(tweet.text.rstrip().replace("\n", "").replace(",","").replace("\r","")),
                    "re": tweet.retweet_count,
                    "fav": tweet.favorite_count
                }, ignore_index=True)
        trainData[comp] = df
        f_out = comp + ".csv"
        trainData[comp].to_csv(f_out)
print("All tweets collected")

#Parse tweets for info like @s, #s links
print("Parsing tweets for metadata")

for comp in comps:
    ats = []
    pounds = []
    links = []
    print("Parsing " + comp)
    for index, tweet in tweets[comp].iterrows():
        words.extend(tweet["textClean"])
        if "@" in tweet["text"]:
            ats.append(1)
            #atsFollowers.append(getFollowers(api, getMention(tweet["text"])))
        else:
            ats.append(0)
            #atsFollowers.append(0)
        pounds.append(1 if "#" in tweet['text'] else 0)
        links.append(1 if len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet['text'])) > 0 else 0)
    tweets[comp]["ats"] = ats
    #tweets[comp]["atsFollowers"] = atsFollowers
    tweets[comp]["pounds"] = pounds
    tweets[comp]["links"] = links
#Addd words from top 8 comps tweets to words list
for comp in trainComps:
    print("Parsing " + comp)
    for index, tweet in trainData[comp].iterrows():
        words.extend(prepareSentence(tweet["text"].rstrip().replace("\n", "").replace(",","").replace("\r","")))
print("Finished parsing tweets",
    "\nSarting BagOfWords")
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
        score = row["fav"] + (2 * row["re"])
        oputs.append(score)
    inputs[comp] = iputs
    outputs[comp] = oputs
for comp in trainComps:
    for index, row in trainData[comp].iterrows():
        bag = toBOW(row["text"], final_words)
        trainInputs.append(bag)
        score = row["fav"] + (2 * row["re"])
        trainOutputs.append(score)
trainOutputs = stats.zscore(trainOutputs)
for comp in comps:
    outputs[comp] = stats.zscore(outputs[comp])
print("Finished BagOfWords")
filename = 'finalized_model.sav'
try:
    print("Trying to load nnet")
    nnet = pickle.load(open(filename, 'rb'))
except:
    print("training nn")
    nnet = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=(int(len(final_words)*0.5),int(len(final_words)*0.25)),solver='adam', max_iter=400)
    nnet.fit(trainInputs, trainOutputs)
    pickle.dump(nnet, open(filename, 'wb'))
    print("Finished Training NN")
#Add sentiment score to fast food comps dataframes
print("computing sentiment")
for comp in comps:
    tweets[comp]["sent"] = nnet.predict(inputs[comp])
    f_out = comp + ".csv"
    tweets[comp].to_csv(f_out)

