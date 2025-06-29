import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load slang dictionary
def load_slang_dict(json_path='utils/slangwords.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)
    return slang_dict

slang_dict = load_slang_dict()

# Buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# List kata negasi yang perlu dipertahankan dan dideteksi
NEGATION_WORDS = {'tidak', 'bukan', 'jangan', 'tak', 'belum'}

# Preprocessing steps
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # hapus mention
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # hapus hashtag
    text = re.sub(r"RT[\s]+", '', text)         # hapus retweet
    text = re.sub(r"http\S+", '', text)         # hapus link
    text = re.sub(r'[0-9]+', '', text)          # hapus angka
    text = re.sub(r'[^\w\s]', '', text)         # hapus simbol
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

def tokenizingText(text):
    return word_tokenize(text)

# Gabungkan kata negasi dengan kata setelahnya
def handle_negations(tokens):
    result = []
    skip_next = False
    for i, word in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if word in NEGATION_WORDS and i + 1 < len(tokens):
            combined = f"{word}_{tokens[i + 1]}"
            result.append(combined)
            skip_next = True
        else:
            result.append(word)
    return result

# Filtering stopword tapi pertahankan kata negasi
def filteringText(words):
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    custom_stop = set([
        'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 
        'loh', 'kah', 'woi', 'woii', 'woy'
    ])

    # Jangan buang kata negasi!
    stop_words.update(custom_stop)
    stop_words.difference_update(NEGATION_WORDS)

    return [word for word in words if word not in stop_words]

def stemmingText(words):
    return [stemmer.stem(word) for word in words]

def toSentence(words):
    return ' '.join(words)

# Fungsi utama
def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    tokens = handle_negations(tokens)       # ⬅️ gabungkan kata negasi
    filtered = filteringText(tokens)
    stemmed = stemmingText(filtered)
    final_text = toSentence(stemmed)
    return final_text
