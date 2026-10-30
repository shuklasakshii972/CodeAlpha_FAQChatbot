# CodeAlpha FAQ Chatbot

A web-based FAQ chatbot that answers questions about AI and computer-science concepts by matching the user's question to a predefined FAQ dataset.

## Features
- Chat-style web interface
- 25 question-answer pairs in the knowledge base
- Text preprocessing (lowercasing, punctuation removal and whitespace normalization)
- TF-IDF vectorization with English stop-word removal and unigrams/bigrams
- Cosine similarity to retrieve the closest FAQ
- Similarity score and fallback for questions with no close match
- Flask JSON endpoint and responsive interface

## Technologies
Python, Flask, Scikit-learn, TF-IDF, cosine similarity, HTML, CSS and JavaScript.

## Setup
Python 3.9 or newer is recommended.

### Windows
```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## How it works
1. The application loads the FAQ question-answer dataset.
2. Text preprocessing converts questions to lowercase and removes punctuation/extra spaces.
3. Scikit-learn's `TfidfVectorizer` converts FAQ questions into numerical vectors.
4. A user's question is preprocessed and transformed using the same vectorizer.
5. Cosine similarity compares the user vector with every FAQ vector.
6. The highest-scoring FAQ answer is returned if the score is at least 0.16; otherwise, a fallback response is shown.

## Add your own FAQs
Edit the `FAQS` list near the top of `app.py`. Each item must contain a `question` and an `answer`. Restart the Flask server after editing.

## Test
Try questions such as:
- What is machine learning?
- Explain TF-IDF
- How does this chatbot work?
- What is overfitting?
- What is NLP?



## CodeAlpha task mapping
This project collects FAQs, preprocesses text, matches user questions with TF-IDF and cosine similarity, and displays the best matching answer in a chat interface.
