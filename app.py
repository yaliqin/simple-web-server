from model.DAM import model_interface,load_model
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

model,graph,sess = load_model()

@app.route('/')
def hello():
    return "Simple Web Server!"


@app.route('/answer', methods=["POST"])
def get_answer():
    input = request.get_json()
    question = input['question']
    return jsonify({"answer": model_interface(question,graph, model,sess)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
