import pandas as pd
import re
import nltk
from joblib import dump
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

n_jobs = -1  # Use all CPU cores for cross validation
scoring = {'Accuracy': 'accuracy', 'F-Measure': 'f1_micro'}

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# Load training csv and preprocess by removing irrelevant chars/urls
print('Loading and preprocessing csv files...')
df_train = pd.read_csv('data/train.csv', usecols=[0, 2])
X_train = df_train['Comment'].values.tolist()
y_train = df_train['Insult'].values.tolist()
regex = re.compile('[^a-zA-Z]')
X_train = [regex.sub(' ', comment) for comment in X_train]
X_train = [re.sub(r'http\S+', '', comment.lower()) for comment in X_train]
# Do the same for test csv
df_test = pd.read_csv('data/impermium_verification_labels.csv', usecols=[1, 3])
X_test = df_test['Comment'].values.tolist()
y_test = df_test['Insult'].values.tolist()
X_test = [regex.sub(' ', comment) for comment in X_test]
X_test = [re.sub(r'http\S+', '', comment.lower()) for comment in X_test]


# Lemmatization class using nltk lemmatizer, this is probably a bad idea! Lemmatization can lower accuracy in
# sentiment analysis tasks
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, texts):
        return [self.wnl.lemmatize(t) for t in word_tokenize(texts)]


# Vectorize with NLTK lemmatizer
print('Vectorizing with CountVectorizer')
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stopwords.words('english'),
                             lowercase=True, ngram_range=(2, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
# Naive Bayes classification and prediction
print('Training and predicting with Multinomial Naive Bayes')
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_vect, y_train)
print('Accuracy for Multinomial naive bayes classifier on lemmatized data:', nb.score(X_test_vect, y_test), 'F1 score:',
      f1_score(y_test, nb.predict(X_test_vect)))

# POS tagging
print('POS tagging the corpus')
nltk.download('averaged_perceptron_tagger')

X_pos_train = [nltk.pos_tag(nltk.word_tokenize(tweet)) for tweet in X_train]


# Function to transform a given dataset to its POS (part of speech) representation using verbs, adjectives, nouns,
# adverbs in relation to each sample's length (fraction).
def pos_transform(data):
    X_pos = []  # New training set list
    for tweet in data:
        pos_list = []
        VERB_COUNT = 0
        ADJECTIVE_COUNT = 0
        ADVERB_COUNT = 0
        NOUN_COUNT = 0
        for tup in tweet:
            if tup[1].startswith('VB'):
                VERB_COUNT += 1
            elif tup[1].startswith('NN'):
                NOUN_COUNT += 1
            elif tup[1].startswith('RB'):
                ADVERB_COUNT += 1
            elif tup[1].startswith('JJ'):
                ADJECTIVE_COUNT += 1
        for x in [VERB_COUNT, ADJECTIVE_COUNT, ADVERB_COUNT, NOUN_COUNT]:
            pos_list.append(x / len(tweet))
        X_pos.append(pos_list)
    return X_pos


X_pos_train = pos_transform(X_pos_train)
X_pos_test = [nltk.pos_tag(nltk.word_tokenize(tweet)) for tweet in X_test]
X_pos_test = pos_transform(X_pos_test)
# Classify using SVM
svm = SVC(probability=True)
print('Training SVM classifier on POS features')
svm.fit(X_pos_train, y_train)
print('Evaluating SVM POS on test set')
print('Accuracy for SVM POS classifier:', svm.score(X_pos_test, y_test), 'F1 score:',
      f1_score(y_test, svm.predict(X_pos_test)))

# Classify using random forest
forest = RandomForestClassifier()
print('Training random forest classifier on POS features')
forest.fit(X_pos_train, y_train)
print('Evaluating random forest classifier on POS features')
print('Accuracy for random forest POS classifier:', forest.score(X_pos_test, y_test), 'F1 score:',
      f1_score(y_test, forest.predict(X_pos_test)))

# Classify using TF-IDF features
tfidf = TfidfVectorizer(max_features=4000)
print('Training tf-idf model on all 3 classifiers')
X_tfidf_train = tfidf.fit_transform(X_train)
dump(tfidf, 'tfidf_vectorizer.joblib')
X_tfidf_test = tfidf.transform(X_test)
nb = MultinomialNB()
nb.fit(X_tfidf_train, y_train)
svm.fit(X_tfidf_train, y_train)
dump(svm, 'svm_classifier.joblib')
forest.fit(X_tfidf_train, y_train)
print('Accuracy for random forest tf idf classifier:', forest.score(X_tfidf_test, y_test), 'F1 score:',
      f1_score(y_test, forest.predict(X_tfidf_test)))
print('Accuracy for SVM tf idf classifier:', svm.score(X_tfidf_test, y_test), 'F1 score:',
      f1_score(y_test, svm.predict(X_tfidf_test)))
print('Accuracy for multinomial naive bayes tf idf classifier:', nb.score(X_tfidf_test, y_test), 'F1 score:',
      f1_score(y_test, nb.predict(X_tfidf_test)))
