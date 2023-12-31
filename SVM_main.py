#SVM model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")
testData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")

print(trainData.sample(frac=1).head(5))

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])

import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Label'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

#results
print("Training time: %fs; Prediction time: %fs" %
      (time_linear_train, time_linear_predict))

report = classification_report(testData['Label'], prediction_linear, output_dict=True)

print('positive: ', report['pos'])
print('negative: ', report['neg'])

review = """I ordered just once from TerribleCo, they screwed up, never used the app again."""

review_vector = vectorizer.transform([review]) # vectorizing
print('review class: ', classifier_linear.predict(review_vector))

dir = '/home/metalist/Documents/Summer_School'
path = os.path.join(dir, 'SVM_model.joblib')
joblib.dump(vectorizer, path)