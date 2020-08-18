import numpy as np
from flask import Flask, request
import pickle

from flask_restful import Api
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


model = pickle.load(open('Pickle_LR_Model.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def results():
    data = request.get_json(force=True)
    print(list(data.values()))
    d = [float(num) for num in list(data.values())]
    print(d)
    prediction = model.predict([np.array(d)])
    output = prediction[0][0]*1000
    return {"price":output}

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
