# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import os
import nltk
from flask import Flask, render_template, request
import text2emotion as te
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import contractions
from googletrans import Translator
from langdetect import detect, DetectorFactory


# Map POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    DetectorFactory.seed = 0
    translator = Translator()
    text = request.form['text']


    file = request.files.get('file')

    if file:
        # Ensure the upload folder exists
        app.config['UPLOAD_FOLDER'] = r"C:\Users\rishu\VrishabhWork\Projects\SentimentAnalysisbot\temp_storage_data"
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file temporarily and read its content
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Remove the temporary file
        os.remove(file_path)

    if not text.strip():
        return "Please provide text either by typing or uploading a file.", 400
    
    text = contractions.fix(text)
    detected_language = detect(text)
    if detected_language != 'en':
        text = translator.translate(text, src=detected_language, dest='en').text
    # Preprocess text
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    processed_text = " ".join(lemmatized_words)

    # Get Text2Emotion scores
    emotions_t2e = te.get_emotion(processed_text)

    # Get VADER sentiment scores
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(processed_text)

    # Normalize VADER scores to match Text2Emotion scale
    total_vader = vader_scores['pos'] + vader_scores['neg'] + vader_scores['neu']
    normalized_vader = {key: value / total_vader for key, value in vader_scores.items()}

    # Map VADER to Text2Emotion categories
    emotions_vader = {
        "Happy": normalized_vader['pos'] * 0.75,
        "Sad": normalized_vader['neg'],
        "Angry": normalized_vader['neg'] * 0.5,  # Partial mapping to anger
        "Surprise": emotions_t2e['Surprise'],  # Use the surprise value from Text2Emotion
        "Fear": normalized_vader['neg'] * 0.5,  # Partial mapping to fear
    }

    # Merge the two emotion models
    combined_emotions = {}
    for emotion in emotions_t2e:
        # Combine the emotion from both models (average the two scores)
        combined_emotions[emotion] = (emotions_t2e[emotion] + emotions_vader.get(emotion, 0)) / 2

    # Convert dict to keys and values for rendering
    emotions_keys = list(combined_emotions.keys())
    emotions_values = list(combined_emotions.values())

    # Returning all nessesary data to the front-end
    return render_template('home.html', text=text, emotions=combined_emotions, 
                       emotions_keys=emotions_keys, emotions_values=emotions_values,
                       emotions_t2e=emotions_t2e, emotions_vader=emotions_vader)



if __name__ == "__main__":
    app.run(debug=True)
