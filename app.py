from flask import Flask, request, jsonify
from inference import generate_response

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    response = generate_response(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
