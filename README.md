# ml-deployment
Demonstration of a deployed machine learning model using Flask. Website is https://ikera.pythonanywhere.com/. Dataset is from https://www.kaggle.com/c/detecting-insults-in-social-commentary/data. The task is to detect whether a comment contains an insult or not. The deployed classifier has around 68% accuracy and is not particularly fine-tuned, since this project was done mostly to show how a machine learning model can be used inside a website. CSS and (modified) HTML is from https://github.com/MaajidKhan/DeployMLModel-Flask
# How to use
If you want to see the project in action, visit https://ikera.pythonanywhere.com/. To run from source, you'll need sklearn, flask, nltk and pandas.  Download the dataset from the Kaggle link above and after putting it in the project's folder, assuming you don't want to train the model by scratch, run deployment.py. By default, this will host a local Flask server on http://127.0.0.1:5000/. 
# Classifier info
Three base models were explored: a nltk LemmaTokenized corpus, a POS tagged corpus and a tf-idf transformed corpus. The best result was obtained using a SVM on the tf-idf transformed corpus. The classifier estimates the probability that the given comment was an insult.
# Deployment info
The file templates/index.html contains the website's layout, styled with some CSS in the static/css folder. In deployment.py a simple flask.render_template method is called inside predict() to route the results of the prediction to the website.
