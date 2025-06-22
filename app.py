from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from utils.preprocessing import (
    cleaningText,
    casefoldingText,
    fix_slangwords,
    tokenizingText,
    filteringText,
    stemmingText,
    toSentence
)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

DATA_PATH = os.path.join(UPLOAD_FOLDER, 'latest_data.csv')
VECTORIZER_PATH = os.path.join(MODEL_FOLDER, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'naive_bayes_model.pkl')


@app.route("/admin/upload", methods=["POST"])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Hanya File CSV yang diterima'}), 400

    file.save(DATA_PATH)
    return jsonify({'message': 'Berhasil upload datasets'})


@app.route("/admin/preview", methods=["GET"])
def preview_dataset():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No dataset uploaded yet'}), 400
    
    df = pd.read_csv(DATA_PATH)
    return jsonify({
        'head': df.head().to_dict(orient='records'),
        'tail': df.tail().to_dict(orient='records')
    })


@app.route("/admin/preprocessed", methods=["GET"])
def preview_preprocessed():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No dataset uploaded'}), 400

    df = pd.read_csv(DATA_PATH).head(5)  # 👈 batasi di awal!

    df['text_clean'] = df['full_text'].apply(cleaningText)
    df['text_casefoldingText'] = df['text_clean'].apply(casefoldingText)
    df['text_slangwords'] = df['text_casefoldingText'].apply(fix_slangwords)
    df['text_token'] = df['text_slangwords'].apply(tokenizingText)
    df['text_stop'] = df['text_token'].apply(filteringText)
    df['text_steming'] = df['text_stop'].apply(stemmingText)
    df['text_final'] = df['text_steming'].apply(toSentence)

    preview = df[[
        'full_text',
        'text_clean',
        'text_casefoldingText',
        'text_slangwords',
        'text_token',
        'text_stop',
        'text_steming',
        'text_final'
    ]]

    return jsonify(preview.to_dict(orient='records'))



@app.route("/admin/train", methods=["POST"])
def train_model():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'Tidak ada dataset yang di upload'}), 400

    import time
    start_all = time.time()

    # Ambil rasio split dari request (default 0.8)
    split_ratio = request.json.get("split_ratio", 0.8)
    try:
        split_ratio = float(split_ratio)
        assert 0.5 <= split_ratio <= 0.95  # Validasi batas
    except:
        return jsonify({"error": "Nilai split_ratio tidak valid (range 0.5 - 0.95)"}), 400

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    if 'full_text' not in df.columns or 'emotion' not in df.columns:
        return jsonify({'error': 'Dataset must contain "full_text" and "emotion" columns'}), 400

    # --- Preprocessing ---
    start = time.time()
    df['text_clean'] = df['full_text'].apply(cleaningText)
    df['text_casefoldingText'] = df['text_clean'].apply(casefoldingText)
    df['text_slangwords'] = df['text_casefoldingText'].apply(fix_slangwords)
    df['text_token'] = df['text_slangwords'].apply(tokenizingText)
    df['text_stop'] = df['text_token'].apply(filteringText)
    df['text_steming'] = df['text_stop'].apply(toSentence)
    df['text_final'] = df['text_steming']
    print("✅ Preprocessing done in", time.time() - start, "seconds")

    df = df.dropna(subset=['text_final', 'emotion'])

    X = df['text_final']
    y = df['emotion']

    # --- TF-IDF Vectorizer ---
    start = time.time()
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)
    print("✅ TF-IDF done in", time.time() - start, "seconds")

    # --- Splitting dataset dengan rasio dinamis ---
    test_size = 1 - split_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=test_size, random_state=42
    )

    # --- Training Model ---
    start = time.time()
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("✅ Training done in", time.time() - start, "seconds")

    # --- Evaluasi ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_labels = model.classes_.tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # --- Simpan model dan vectorizer ---
    joblib.dump(tfidf, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("✅ All done in", time.time() - start_all, "seconds")

    return jsonify({
        'message': f'Training model berhasil dan tersimpan dengan split data {int(split_ratio * 100)}:{int((1 - split_ratio) * 100)}',
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'report': report,
        'confusion_matrix': {
            'labels': cm_labels,
            'matrix': cm.tolist()
        }
    })


@app.route("/predict", methods=["POST"])
def predict_emotion():
    try:
        text_input = request.json.get("text", "")
        if not isinstance(text_input, str) or not text_input.strip():
            return jsonify({"error": "Teks input tidak valid"}), 400

        # --- Preprocessing ---
        text_clean = cleaningText(text_input)
        text_casefolding = casefoldingText(text_clean)
        text_slang = fix_slangwords(text_casefolding)
        text_token = tokenizingText(text_slang)
        text_stop = filteringText(text_token)
        text_steming = stemmingText(text_stop)
        text_final = toSentence(text_steming)

        if not text_final:
            return jsonify({"error": "Pra-pemrosesan gagal menghasilkan teks yang valid"}), 500

        # --- Load Model & Vectorizer ---
        tfidf = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)

        X_input = tfidf.transform([text_final])

        # --- TF-IDF Detail ---
        tfidf_vector = X_input.toarray()[0]
        feature_names = tfidf.get_feature_names_out()
        tfidf_detail = []
        tfidf_dict = {}
        word_set = set(text_steming)

        for word in word_set:
            if word in tfidf.vocabulary_:
                index = tfidf.vocabulary_[word]
                tf = X_input[:, index].sum()
                idf = tfidf.idf_[index]
                tfidf_value = tf * idf

                tfidf_dict[word] = tfidf_value

                tfidf_detail.append({
                    "word": word,
                    "tf": round(tf, 6),
                    "idf": round(idf, 6),
                    "tfidf": round(tfidf_value, 6)
                })

        # --- Log Prior Detail ---
        total_doc = sum(model.class_count_)
        prior_detail = {
            label: {
                "count": int(model.class_count_[i]),
                "total": int(total_doc),
                "prior": model.class_count_[i] / total_doc,
                "log_prior": float(model.class_log_prior_[i])
            }
            for i, label in enumerate(model.classes_)
        }

        # --- Step-by-step Naive Bayes ---
        step_by_step = {}
        for i, label in enumerate(model.classes_):
            log_prior = model.class_log_prior_[i]
            log_likelihood_sum = 0
            total_words_in_class = model.feature_count_[i].sum()
            vocab_size = model.feature_count_.shape[1]

            word_details = []
            for word in tfidf_dict:
                index = tfidf.vocabulary_.get(word)
                tfidf_val = tfidf_dict[word]
                if index is not None:
                    count_w_class = model.feature_count_[i][index]
                    log_likelihood = model.feature_log_prob_[i][index]
                    log_contrib = log_likelihood * tfidf_val
                else:
                    count_w_class = 0
                    log_likelihood = -100  # asumsi smoothing ekstrem
                    log_contrib = log_likelihood * tfidf_val

                log_likelihood_sum += log_contrib

                word_details.append({
                    "word": word,
                    "tfidf": round(tfidf_val, 6),
                    "count_w_class": int(count_w_class),
                    "total_words_in_class": int(total_words_in_class),
                    "vocab_size": int(vocab_size),
                    f"log(P(w|{label}))": round(log_likelihood, 6),
                    f"log(P(w|{label})) * tfidf": round(log_contrib, 6),
                    f"rumus_log(P(w|{label}))": f"log(({int(count_w_class)} + 1) / ({int(total_words_in_class)} + {int(vocab_size)}))"
                })

            total_log_score = log_prior + log_likelihood_sum
            step_by_step[label] = {
                "log_prior": round(log_prior, 6),
                "log_likelihood_sum": round(log_likelihood_sum, 6),
                "total_log_score": round(total_log_score, 6),
                "details": word_details
            }

        # --- Probabilitas dari total_log_score ---
        import numpy as np
        log_scores = [step_by_step[label]["total_log_score"] for label in model.classes_]
        exp_scores = np.exp(log_scores)
        sum_exp_scores = np.sum(exp_scores)

        final_probabilities = {
            label: float(exp_score / sum_exp_scores)
            for label, exp_score in zip(model.classes_, exp_scores)
        }

        probability_detail = {
            label: {
                "log_score": float(step_by_step[label]["total_log_score"]),
                "exp_log_score": float(exp_score),
                "sum_exp_scores": float(sum_exp_scores),
                "final_probability": float(exp_score / sum_exp_scores)
            }
            for label, exp_score in zip(model.classes_, exp_scores)
        }

        # --- Ambil label dengan probabilitas tertinggi ---
        pred_label = max(final_probabilities.items(), key=lambda x: x[1])[0]

        return jsonify({
            "text": text_input,
            "emotion": pred_label,
            "confidence": float(final_probabilities[pred_label]),
            "probabilities": final_probabilities,
            "probability_detail": probability_detail,
            "tfidf": tfidf_dict,
            "tfidf_detail": tfidf_detail,
            "step_by_step": step_by_step,
            "prior_detail": prior_detail
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5001)
