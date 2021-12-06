import flask
from flask import Flask, request
from joblib import load

app = Flask(__name__, template_folder='templates')

# Load dumped classifier/vectorizer
clf = load('log_regr_clf.joblib')
tfidf_vect = load('tfidf.joblib')


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
    neutral_prob = '{:.4f}'.format(100 * float(prediction[0][0]))
    anti_prob = '{:.4f}'.format(100 * float(prediction[0][1]))
    pro_prob = '{:.4f}'.format(100 * float(prediction[0][2]))
    return flask.render_template('index.html',
                                 value=comment[0],
                                 prediction_text='Probability this is a neutral comment: ' + neutral_prob + '%\n' +
                                 'Probability this is an anti-vax comment: ' + anti_prob + '%\n' +
                                 'Probability this is a pro-vax comment: ' + pro_prob + '%')


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)