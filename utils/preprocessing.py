import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def load_slang_dict(json_path='utils/slangwords.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)
    return slang_dict

slang_dict = load_slang_dict()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r"RT[\s]+", '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(words):
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    custom_stop = set(['iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'])
    stop_words.update(custom_stop)
    return [word for word in words if word not in stop_words]

def stemmingText(words):  # ✅ menerima list, bukan string
    return [stemmer.stem(word) for word in words]

def toSentence(words):
    return ' '.join(words)

def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    filtered = filteringText(tokens)
    stemmed = stemmingText(filtered)
    final_text = toSentence(stemmed)
    return final_text
