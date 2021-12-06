# vaccine-sentiment
Demonstration of a deployed machine learning model using Flask. The task is to analyse a tweet about the COVID vaccine and classify its sentiment. The deployed classifier has around 73% accuracy (I've also uploaded an image of the confusion matrix) and is not particularly fine-tuned, since this is just a prototype to show how a machine learning model can be deployed inside a website. CSS and (modified) HTML is from https://github.com/MaajidKhan/DeployMLModel-Flask
# How to use
If you want to see the project in action, visit https://ikera.pythonanywhere.com/. I'm not sure if I can distribute the dataset yet, since it was obtained from a university course page. 
# Classifier info (vacc_deployment.py)
The prototype classifier uses a Logistic Regression model, applied on a tf-idf vectorized twitter corpus. There are 3 possible classes: neutral, provax, anti-vax.
# Deployment info
The file templates/index.html contains the website's layout, styled with some CSS in the static/css folder. In vacc_deployment.py a simple flask.render_template method is called inside predict() to route the results of the prediction to the website.
