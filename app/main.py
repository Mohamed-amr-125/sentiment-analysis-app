from flask import Blueprint, render_template, request, jsonify
from model.sentiment_model import predict_sentiment

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    result = predict_sentiment(review)
    # Convert probabilities to percentages
    result = {key: f"{value * 100:.2f}%" for key, value in result.items()}
    return jsonify(result)