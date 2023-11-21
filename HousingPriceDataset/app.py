import pickle
import sklearn
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
randomForestModel=pickle.load(open('regModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    a2D = np.array(list(data.values())).reshape(1, -1)
    output = randomForestModel.predict(a2D)
    return jsonify({'result': output.tolist()})

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    print(final_input)
    output = randomForestModel.predict(final_input)
    return render_template("result.html", result=output[0])



if __name__=="__main__":
    app.run(debug=True)