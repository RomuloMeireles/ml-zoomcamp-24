import pickle
from flask import Flask, request, jsonify

# PREFIX = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework"
# files = {
#     "model": f"{PREFIX}/model1.bin",
#     "dv": f"{PREFIX}/dv.bin"
# }

# for file_name, url in files.items():
#     response = requests.get(url)
#     response.raise_for_status()
#     with open(f"{file_name}.bin", "wb") as f:
#         f.write(response.content)
#     print(f"Downloaded {file_name}.bin")

model = "model.bin"
dv = "dv.bin"

with open(model, "rb") as model_file, open(dv, "rb") as dv_file:
    model = pickle.load(model_file)
    dv = pickle.load(dv_file)
    
app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]  
    
    result = {
        'subscription-probability': float(y_pred),
    }
    
    return jsonify(result)
    
    
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9698)