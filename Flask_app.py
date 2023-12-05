from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re

app = Flask(__name__)

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your preprocessed data or preprocess it here
# For simplicity, we'll load a sample DataFrame
df = pd.DataFrame({
    'ProcessedColumn1': ['text one', 'another sentence', 'sentence three'],
    'ProcessedColumn2': ['sentence 1', 'a different sentence', 'this is sentence 3']
})

# Tokenize and Calculate Cosine Similarity
vectorizer = CountVectorizer().fit(df['ProcessedColumn1'] + ' ' + df['ProcessedColumn2'])
vectorized_text1 = vectorizer.transform(df['ProcessedColumn1']).toarray()
vectorized_text2 = vectorizer.transform(df['ProcessedColumn2']).toarray()

cosine_sim = cosine_similarity(vectorized_text1, vectorized_text2)

# Set a Threshold and Rate Similarity
threshold = 0.8
df['SimilarityRating'] = cosine_sim.mean(axis=1)

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    sentence1 = data.get('sentence1', '')
    sentence2 = data.get('sentence2', '')

    # Preprocess the input sentences
    preprocessed_sentence1 = preprocess_text(sentence1)
    preprocessed_sentence2 = preprocess_text(sentence2)

    # Tokenize and Calculate Cosine Similarity
    vectorized_input1 = vectorizer.transform([preprocessed_sentence1]).toarray()
    vectorized_input2 = vectorizer.transform([preprocessed_sentence2]).toarray()

    cosine_similarity_score = cosine_similarity(vectorized_input1, vectorized_input2)[0, 0]

    # Return the similarity rating
    return jsonify({'similarity_rating': cosine_similarity_score})

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove numerical digits
    text = re.sub(r'\d+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

if __name__ == '__main__':
    app.run(debug=True)
