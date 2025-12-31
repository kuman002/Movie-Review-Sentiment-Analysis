from flask import Flask, request, jsonify, render_template
from src.utilis import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    prob, sentiment = predict(text)

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(prob), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
