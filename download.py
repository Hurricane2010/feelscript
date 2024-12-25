import nltk
import os

# Define the directory for storing NLTK data
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

# Ensure the directory exists
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Now download the resources to the specified directory
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

print("NLTK resources downloaded to the specified directory.")
