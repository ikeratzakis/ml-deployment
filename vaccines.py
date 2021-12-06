import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import numpy as np
from joblib import dump
from nltk.corpus import stopwords
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

nltk.download('stopwords')
stopwords = stopwords.words('english') + ['vaccine', 'vaccines', 'vaccination']


# Load dataset. 0 is neutral, 1 is anti-vax, 2 is pro-vax
def load_dataset(subset):
    path = ''
    if subset == 'train':
        path = 'datasets/vaccine_train_set.csv'
    elif subset == 'test':
        path = 'datasets/vaccine_validation_set.csv'

    df = pd.read_csv(path)
    X = df['tweet']
    y = df['label']
    return X, y


def plot_wordcloud(data, labels):
    tweet_dict = {0: [], 1: [], 2: []}
    for tweet, sentiment in zip(data, labels):
        tweet_dict[sentiment].append(tweet)
    print(tweet_dict[0])
    print('Generating wordclouds...')
    pro_cloud = WordCloud(stopwords=stopwords).generate(''.join(tweet for tweet in tweet_dict[0]))
    print('Next cloud')
    anti_cloud = WordCloud(stopwords=stopwords).generate(''.join(tweet for tweet in tweet_dict[1]))
    print('Next cloud')
    neutral_cloud = WordCloud(stopwords=stopwords).generate(''.join(tweet for tweet in tweet_dict[2]))
    fig, axes = plt.subplots(1, 3)
    axes[0].axis('off'), axes[1].axis('off'), axes[2].axis('off')
    axes[0].imshow(pro_cloud, interpolation='bilinear'), axes[0].set_title('Pro-vax tweets'),
    axes[1].imshow(anti_cloud, interpolation='bilinear'), axes[1].set_title('Antivax tweets')
    axes[2].imshow(neutral_cloud, interpolation='bilinear'), axes[2].set_title('Neutral tweets')
    plt.show()


def preprocess(data):
    clean_corpus = []
    # Remove all https and non letters from the corpus

    for tweet in data:
        result = re.sub(r'http\S+', '', tweet)
        clean_tweet = re.sub('[^a-zA-Z]+', ' ', result)
        clean_corpus.append(clean_tweet)
    return clean_corpus


# Vectorize data
def vectorize(subset, data, tfidf_vectorizer=None):
    if subset == 'train':
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(data)
        return X, tfidf
    elif subset == 'test':
        X = tfidf_vectorizer.transform(data)
        return X


# Train classifier and plot learning curve
def train_classifier(data, labels):
    clf = LogisticRegression(max_iter=200, n_jobs=-1, solver='liblinear', dual=True)
    plt.title('Learning curve (logistic regression)')
    plt.xlabel('Learning examples')
    plt.ylabel('Score')

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X=data, y=labels, cv=5, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.legend(loc='best')
    clf.fit(data, labels)
    return clf


# Evaluate on validation data and plot results
def evaluate_classifier(clf, data, labels):
    predictions = clf.predict(data)
    print('Classification report:', classification_report(y_true=labels, y_pred=predictions))
    # Plot confusion matrix
    class_labels = ['neutral', 'antivax', 'provax']
    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()
    plt.title('Vaccination sentiment classifier')
    plt.show()


def main():
    X_train, y_train = load_dataset(subset='train')
    X_test, y_test = load_dataset(subset='test')

    # Preprocessing and wordcloud
    print('Preprocessing data...')
    X_train = preprocess(X_train)
    print(X_train[2])
    X_test = preprocess(X_test)
    plot_wordcloud(X_train, y_train)

    # Vectorize
    X_train, tfidf = vectorize(subset='train', data=X_train)

    X_test = vectorize(subset='test', data=X_test, tfidf_vectorizer=tfidf)

    # Train and evaluate classifier
    print('Training classifier...')
    clf = train_classifier(data=X_train, labels=y_train)
    dump(tfidf, 'tfidf.joblib')
    dump(clf, 'log_regr_clf.joblib')
    evaluate_classifier(clf, data=X_test, labels=y_test)


main()
