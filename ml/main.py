import pandas as pd
import numpy as np
import pickle
import joblib
import os
import re
import math
import warnings

# Import library yang diperlukan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore')

# [FIX] Menambahkan path khusus agar NLTK bisa menemukan data di Azure App Service
nltk.data.path.append("/home/site/wwwroot/nltk_data")


# Mendapatkan path absolut dari direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def replace_nan_with_none(data): #
    if isinstance(data, dict): #
        return {k: replace_nan_with_none(v) for k, v in data.items()} #
    elif isinstance(data, list): #
        return [replace_nan_with_none(elem) for elem in data] #
    elif isinstance(data, float) and math.isnan(data): #
        return None #
    else:
        return data #

class CVJobMatcher:
    def __init__(self): #
        self.tfidf_vectorizer = TfidfVectorizer( #
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.stemmer = PorterStemmer() #
        self.stop_words = set(stopwords.words('english')) #
        self.job_data = None #
        self.job_vectors = None #
        self.model = None #
        self.tokenizer = None #

    def extract_text_from_pdf(self, pdf_path): #
        try:
            with open(pdf_path, 'rb') as file: #
                pdf_reader = PyPDF2.PdfReader(file) #
                text = "" #
                for page in pdf_reader.pages: #
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                return text.strip() #
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return "" #

    def preprocess_text(self, text): #
        if not isinstance(text, str):
            return ""
        text = text.lower() #
        text = re.sub(r'[^a-zA-Z\s]', '', text) #
        text = re.sub(r'\s+', ' ', text).strip() #
        tokens = word_tokenize(text) #
        processed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2] #
        return ' '.join(processed_tokens) #
        
    def find_matching_jobs(self, cv_text, top_k=10): #
        # Fungsi ini sekarang mengembalikan dictionary untuk debugging
        if self.job_data is None or self.job_vectors is None:
            return {"jobs": [], "debug_reason": "Data pekerjaan atau vektor model tidak dimuat."}

        try:
            original_text_len = len(cv_text)
            processed_cv = self.preprocess_text(cv_text) #
            processed_text_len = len(processed_cv)

            if not processed_cv:
                reason = f"Teks CV menjadi kosong setelah diproses. Panjang teks asli: {original_text_len}."
                return {"jobs": [], "debug_reason": reason}

            cv_vector = self.tfidf_vectorizer.transform([processed_cv]) #
            similarities = cosine_similarity(cv_vector, self.job_vectors).flatten() #
            
            if np.max(similarities) == 0.0:
                 reason = (f"CV berhasil diproses (panjang: {processed_text_len}), "
                           f"namun tidak ada kecocokan sama sekali (semua skor similaritas 0). "
                           f"Coba gunakan teks CV yang lebih relevan dengan deskripsi pekerjaan.")
                 return {"jobs": [], "debug_reason": reason}

            top_indices = similarities.argsort()[-top_k:][::-1] #

            results = [] #
            for idx in top_indices: #
                job_info = self.job_data.iloc[idx].to_dict() #
                job_info['similarity_score'] = similarities[idx] #
                results.append(job_info) #

            cleaned_results = replace_nan_with_none(results) #
            return {"jobs": cleaned_results, "debug_reason": "Pencocokan berhasil."}

        except Exception as e:
            return {"jobs": [], "debug_reason": f"Terjadi exception: {str(e)}"}

    def load_job_dataset(self, dataset_path): #
        try:
            abs_dataset_path = os.path.join(SCRIPT_DIR, dataset_path)
            self.job_data = pd.read_csv(abs_dataset_path) #
            self.job_data['combined_text'] = ( #
                self.job_data['Title'].fillna('') + ' ' +
                self.job_data['Company'].fillna('') + ' ' +
                self.job_data['Job Description'].fillna('') + ' ' +
                self.job_data['Job Requirements'].fillna('') + ' ' +
                self.job_data['Location'].fillna('')
            )
            self.job_data['processed_text'] = self.job_data['combined_text'].apply(self.preprocess_text) #
            print(f"Loaded {len(self.job_data)} job records from {abs_dataset_path}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False #

    def load_model(self, model_name): #
        try:
            models_folder = os.path.join(SCRIPT_DIR, "models")
            base_path = os.path.join(models_folder, model_name) #
            print(f"Loading model from base path: {base_path}")
            required_files = {
                "tfidf_vectorizer": f"{base_path}_tfidf_vectorizer.pkl",
                "job_data": f"{base_path}_job_data.pkl",
                "job_vectors": f"{base_path}_job_vectors.pkl"
            }
            for name, file_path in required_files.items():
                if not os.path.exists(file_path):
                    print(f"Error: Required model file not found: {file_path}")
                    return False #
            self.tfidf_vectorizer = joblib.load(required_files["tfidf_vectorizer"]) #
            self.job_data = joblib.load(required_files["job_data"]) #
            self.job_vectors = joblib.load(required_files["job_vectors"]) #
            print(f"Model '{model_name}' loaded successfully from {models_folder}")
            return True #
        except Exception as e:
            print(f"Error loading model: {e}")
            return False #