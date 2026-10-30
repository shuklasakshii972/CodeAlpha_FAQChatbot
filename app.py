
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# FAQ Knowledge Base
FAQS = [
    {"question": "What is artificial intelligence?", "answer": "Artificial Intelligence (AI) enables computers to perform tasks that normally require human intelligence."},
    {"question": "What is machine learning?", "answer": "Machine Learning allows computers to learn patterns from data and make predictions."},
    {"question": "What is deep learning?", "answer": "Deep Learning uses multi-layered neural networks to learn complex patterns."},
    {"question": "What is natural language processing?", "answer": "NLP helps computers understand and process human language."},
    {"question": "What is a chatbot?", "answer": "A chatbot is software that communicates with users and provides automated responses."},
    {"question": "What is Python?", "answer": "Python is a high-level programming language known for its simple syntax."},
    {"question": "What is Flask?", "answer": "Flask is a lightweight Python framework used to build web applications."},
    {"question": "What is an API?", "answer": "An API allows different software applications to communicate with each other."},
    {"question": "What is a dataset?", "answer": "A dataset is an organized collection of data used for analysis and model training."},
    {"question": "What is supervised learning?", "answer": "Supervised learning trains models using labelled data."},
    {"question": "What is unsupervised learning?", "answer": "Unsupervised learning finds patterns in data without labelled outputs."},
    {"question": "What is classification?", "answer": "Classification predicts categories or classes from input data."},
    {"question": "What is regression?", "answer": "Regression predicts continuous numerical values."},
    {"question": "What is overfitting?", "answer": "Overfitting occurs when a model learns training data too closely and performs poorly on unseen data."},
    {"question": "What is underfitting?", "answer": "Underfitting occurs when a model is too simple to learn patterns in data."},
    {"question": "What is a neural network?", "answer": "A neural network is a machine learning model inspired by the human brain."},
    {"question": "What is tokenization?", "answer": "Tokenization splits text into smaller units such as words or sentences."},
    {"question": "What is TF-IDF?", "answer": "TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures the importance of words in documents."},
    {"question": "What is cosine similarity?", "answer": "Cosine similarity measures how similar two vectors are."},
    {"question": "How does this chatbot work?", "answer": "The chatbot uses TF-IDF and cosine similarity to find the closest matching FAQ and return its answer."},
    {"question": "What is data preprocessing?", "answer": "Data preprocessing cleans and prepares raw data for analysis."},
    {"question": "What is computer vision?", "answer": "Computer vision enables computers to understand images and videos."},
    {"question": "What is generative AI?", "answer": "Generative AI creates new content such as text, images and code."},
    {"question": "What is a database?", "answer": "A database is an organized collection of information."},
    {"question": "What is cloud computing?", "answer": "Cloud computing provides computing resources over the internet."}
]

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Prepare FAQ questions
questions = [clean_text(faq["question"]) for faq in FAQS]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(questions)


# Find the most similar FAQ
def find_answer(question):

    cleaned = clean_text(question)

    if not cleaned:
        return "Please type a question.", 0

    user_vector = vectorizer.transform([cleaned])

    scores = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    index = scores.argmax()
    score = float(scores[index])

    if score < 0.16:
        return (
            "Sorry, I couldn't find a matching answer. "
            "Try asking about AI, machine learning, NLP or Python.",
            score
        )

    return FAQS[index]["answer"], score


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        faqs=FAQS
    )


# Chatbot API
@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please type a question.",
            "score": 0
        })

    answer, score = find_answer(question)

    return jsonify({
        "answer": answer,
        "score": round(score, 3)
    })


# Health check
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "faq_count": len(FAQS)
    })


if __name__ == "__main__":
    app.run(debug=True)
