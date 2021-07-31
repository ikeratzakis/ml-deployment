import flask
from flask import Flask, request
from joblib import load

app = Flask(__name__, template_folder='templates')

# Load dumped classifier/vectorizer
clf = load('svm_classifier.joblib')
tfidf_vect = load('tfidf_vectorizer.joblib')


# Renders the main webpage
@app.route('/')
def index():
    return flask.render_template('index.html')


# Predict API
@app.route('/predict', methods=['POST'])
def predict():
    # Transform to tf-idf representation
    comment = [request.form['comment']]
    vect_comment = tfidf_vect.transform(comment)
    # Make prediction and return results
    prediction = clf.predict_proba(vect_comment)
    insult_prob = '{:.4f}'.format(100 * float(prediction[0][1]))
    return flask.render_template('index.html',
                                 value=comment[0],
                                 prediction_text='Probability this is an insult: ' + insult_prob + '%')


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
