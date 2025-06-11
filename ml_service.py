import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from main import CVJobMatcher, replace_nan_with_none 

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Konfigurasi Logging ---
# Menggunakan logger bawaan Flask lebih baik daripada print() untuk produksi
logging.basicConfig(level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Pastikan folder uploads ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    app.logger.info(f"Folder '{UPLOAD_FOLDER}' telah dibuat.")

# --- Inisialisasi Model ---
# Model dimuat hanya sekali saat server dimulai untuk efisiensi.
app.logger.info("Memulai proses pemuatan model pencocokan CV...")
matcher = CVJobMatcher()

# Memuat model dari path yang sudah diperbaiki di main.py
# Nama model 'cv_job_matcher_model' mengacu pada file-file seperti 'cv_job_matcher_model_tfidf_vectorizer.pkl'
if matcher.load_model("cv_job_matcher_model"):
    app.logger.info("Model berhasil dimuat!")
else:
    app.logger.error("GAGAL memuat model. Pastikan file model ada di folder 'ml/models/'. Aplikasi mungkin tidak berfungsi dengan benar.")

# --- Definisi Endpoint API ---
@app.route('/match', methods=['POST'])
def match_cv_endpoint():
    """
    Endpoint untuk menerima file CV (PDF), memprosesnya, dan mengembalikan
    pekerjaan yang cocok.
    """
    app.logger.info("Endpoint '/match' dipanggil.")
    
    # 1. Validasi permintaan: Pastikan file ada dalam request
    if 'cv_pdf' not in request.files:
        app.logger.warning("Permintaan gagal: 'cv_pdf' tidak ditemukan.")
        return jsonify({"error": "File 'cv_pdf' tidak ditemukan dalam permintaan"}), 400

    file = request.files['cv_pdf']

    # Jika pengguna tidak memilih file, browser mungkin mengirim part kosong
    if file.filename == '':
        app.logger.warning("Permintaan gagal: Tidak ada file yang dipilih.")
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    # 2. Proses File
    filepath = None
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            app.logger.info(f"File '{filename}' berhasil disimpan di '{filepath}'.")

            # 3. Ekstrak teks dan cari pekerjaan yang cocok
            app.logger.info(f"Memproses file: {filepath}")
            cv_text = matcher.extract_text_from_pdf(filepath)

            if not cv_text:
                app.logger.error(f"Gagal mengekstrak teks dari PDF: {filename}")
                return jsonify({"error": "Gagal mengekstrak teks dari PDF"}), 500

            # Cari 10 pekerjaan teratas
            matching_jobs = matcher.find_matching_jobs(cv_text, top_k=10)

            # Bersihkan nilai NaN agar bisa menjadi JSON yang valid
            cleaned_matching_jobs = replace_nan_with_none(matching_jobs)

            # 4. Kirim hasil kembali sebagai JSON
            app.logger.info(f"Menemukan {len(cleaned_matching_jobs)} pekerjaan yang cocok untuk '{filename}'.")
            return jsonify({
                "status": "success",
                "message": "Pekerjaan berhasil dicocokkan",
                "data": cleaned_matching_jobs
            })

        except Exception as e:
            app.logger.error(f"Terjadi error saat memproses file: {e}", exc_info=True)
            return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

        finally:
            # 5. Hapus file sementara setelah diproses
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                app.logger.info(f"File sementara dihapus: {filepath}")

    app.logger.warning("Permintaan gagal: File tidak valid atau tidak diproses.")
    return jsonify({"error": "File tidak valid"}), 400
