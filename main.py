#Sentiment analysis with Native Bayes Model


import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import joblib
import os



stop_words = stopwords.words('english')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = 'v'
        else:
            pos = 'a'
            
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
                
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

"""   
def save_model(model, dir='/home/metalist/Documents/Summer_School'):
    path = os.path.join(dir, 'model.joblib')
    return joblib.dump(model, path)
    """
        
        
positive_tweets = twitter_samples.strings('positive_tweets.json')
#print(positive_tweets)
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
#print(text.)
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

print(tweet_tokens[0])
print('#################################################')
print('Positive tweet example : ', positive_tweets[0])

print(pos_tag(tweet_tokens[0]))
print('#################################################')
print(lemmatize_sentence(tweet_tokens[0]))
print('#################################################')
print(remove_noise(tweet_tokens[0], stop_words))
print('#################################################')
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
print(positive_tweet_tokens[500])
print('-------------------------------------------------')
print(positive_cleaned_tokens_list[500])

print('#################################################')
all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))
print('#################################################')

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]
# print(positive_dataset)
negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]
# print(negative_dataset)
dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


# Building and testing the model
classifier = NaiveBayesClassifier.train(train_data)
print('#################################################')
print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))
# Building and testing the model

dir = '/home/metalist/Documents/Summer_School'
path = os.path.join(dir, 'model.joblib')
joblib.dump(classifier, path)
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################




print('#################################################')
custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
custom_tokens = remove_noise(word_tokenize(custom_tweet))
print(classifier.classify(dict([token, True] for token in custom_tokens)))
p_custom_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
p_custom_tokens = remove_noise(word_tokenize(p_custom_tweet))
print(classifier.classify(dict([token, True] for token in p_custom_tokens)))

s_custom_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'
s_custom_tokens = remove_noise(word_tokenize(s_custom_tweet))
print(classifier.classify(dict([token, True] for token in s_custom_tokens)))
