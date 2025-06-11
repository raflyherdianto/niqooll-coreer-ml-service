import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import math
warnings.filterwarnings('ignore')

# Mendapatkan path absolut dari direktori tempat script ini berada (yaitu, folder 'ml/')
# Ini memastikan path akan selalu benar, tidak peduli dari mana script dijalankan.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def replace_nan_with_none(data):
    """
    Recursively replaces NaN float values with None in dictionaries and lists.
    This makes the data JSON serializable.
    """
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(elem) for elem in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data

class CVJobMatcher:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.job_data = None
        self.job_vectors = None
        self.model = None
        self.tokenizer = None
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Ekstrak teks dari file PDF CV
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    # Tambahkan spasi antar halaman untuk memastikan kata tidak menyatu
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    def preprocess_text(self, text):
        """
        Preprocessing teks dengan cleaning dan normalisasi
        """
        if not isinstance(text, str):
            return "" # Kembalikan string kosong jika input bukan string
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        
        processed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(processed_tokens)
    
    def load_job_dataset(self, dataset_path):
        """
        Load dataset lowongan kerja
        Expected columns: ['Title', 'Company', 'Location', 'Country', 'Job Description', 'Job Requirements', 'Link']
        """
        try:
            # Menggunakan SCRIPT_DIR untuk membuat path dataset menjadi absolut
            abs_dataset_path = os.path.join(SCRIPT_DIR, dataset_path)
            self.job_data = pd.read_csv(abs_dataset_path)
            
            expected_columns = ['Title', 'Company', 'Location', 'Country', 'Job Description', 'Job Requirements', 'Link']
            if not all(col in self.job_data.columns for col in expected_columns):
                 print(f"Warning: Dataset tidak memiliki semua kolom yang diharapkan. Kolom yang ada: {list(self.job_data.columns)}")

            # Gabungkan kolom teks untuk membuat satu representasi teks per pekerjaan
            self.job_data['combined_text'] = (
                self.job_data['Title'].fillna('') + ' ' +
                self.job_data['Company'].fillna('') + ' ' +
                self.job_data['Job Description'].fillna('') + ' ' +
                self.job_data['Job Requirements'].fillna('') + ' ' +
                self.job_data['Location'].fillna('')
            )
            
            self.job_data['processed_text'] = self.job_data['combined_text'].apply(self.preprocess_text)
            
            print(f"Loaded {len(self.job_data)} job records from {abs_dataset_path}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def build_job_vectors(self):
        """
        Build TF-IDF vectors untuk dataset lowongan kerja
        """
        if self.job_data is None:
            print("Please load job dataset first")
            return False
        
        job_texts = self.job_data['processed_text'].tolist()
        self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
        print(f"Built job vectors with shape: {self.job_vectors.shape}")
        return True
            
    def find_matching_jobs(self, cv_text, top_k=10):
        """
        Cari lowongan kerja yang paling cocok dengan CV
        """
        if self.job_data is None or self.job_vectors is None:
            print("Job data and vectors are not loaded. Cannot find matches.")
            return []
        
        try:
            processed_cv = self.preprocess_text(cv_text)
            if not processed_cv:
                print("CV text is empty after preprocessing.")
                return []

            cv_vector = self.tfidf_vectorizer.transform([processed_cv])
            similarities = cosine_similarity(cv_vector, self.job_vectors).flatten()
            
            # Ambil indeks dan skor dari K teratas
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                job_info = self.job_data.iloc[idx].to_dict()
                job_info['similarity_score'] = similarities[idx]
                results.append(job_info)
            
            # Pastikan hasil bersih dari NaN
            cleaned_results = replace_nan_with_none(results)
            return cleaned_results
            
        except Exception as e:
            print(f"Error finding matching jobs: {e}")
            return []
    
    def save_model(self, model_name):
        """
        Simpan model dalam folder 'ml/models/'
        """
        try:
            # Path ke folder models di dalam folder 'ml'
            models_folder = os.path.join(SCRIPT_DIR, "models")
            if not os.path.exists(models_folder):
                os.makedirs(models_folder)
                print(f"Created folder: {models_folder}")
            
            base_path = os.path.join(models_folder, model_name)
            
            joblib.dump(self.tfidf_vectorizer, f"{base_path}_tfidf_vectorizer.pkl")
            joblib.dump(self.job_data, f"{base_path}_job_data.pkl")
            joblib.dump(self.job_vectors, f"{base_path}_job_vectors.pkl")
            
            print(f"Model components saved successfully to {models_folder}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name):
        """
        Load model dari folder 'ml/models/'
        """
        try:
            # Path ke folder models di dalam folder 'ml'
            models_folder = os.path.join(SCRIPT_DIR, "models")
            base_path = os.path.join(models_folder, model_name)
            
            print(f"Loading model from base path: {base_path}")

            required_files = {
                "tfidf_vectorizer": f"{base_path}_tfidf_vectorizer.pkl",
                "job_data": f"{base_path}_job_data.pkl", 
                "job_vectors": f"{base_path}_job_vectors.pkl"
            }
            
            for name, file_path in required_files.items():
                if not os.path.exists(file_path):
                    print(f"Error: Required model file not found: {file_path}")
                    return False
            
            self.tfidf_vectorizer = joblib.load(required_files["tfidf_vectorizer"])
            self.job_data = joblib.load(required_files["job_data"])
            self.job_vectors = joblib.load(required_files["job_vectors"])
            
            print(f"Model '{model_name}' loaded successfully from {models_folder}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
