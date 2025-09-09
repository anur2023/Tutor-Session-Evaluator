import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
import contractions

# nltk.download("punkt")
# nltk.download("wordnet")

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def normalize_text(self, text):
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def expand_contractions(self, text):
        return contractions.fix(text)

    def lemmatize_text(self, text):
        words = nltk.word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])

    def preprocess(self, text):
        text = self.normalize_text(text)
        text = self.expand_contractions(text)
        text = self.lemmatize_text(text)
        return text
