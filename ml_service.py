import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ml.main import CVJobMatcher, replace_nan_with_none # [FIX] Path impor yang benar

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# [FIX] Pastikan folder uploads ada (aman untuk multiple workers)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.logger.info(f"Folder '{app.config['UPLOAD_FOLDER']}' sudah siap.")


# --- Inisialisasi Model ---
app.logger.info("Memulai proses pemuatan model pencocokan CV...")
matcher = CVJobMatcher()
if matcher.load_model("cv_job_matcher_model"): #
    app.logger.info("Model berhasil dimuat!")
else:
    app.logger.error("GAGAL memuat model.")

# --- Definisi Endpoint API ---
@app.route('/match', methods=['POST']) #
def match_cv_endpoint():
    app.logger.info("Endpoint '/match' dipanggil.")
    if 'cv_pdf' not in request.files: #
        return jsonify({"error": "File 'cv_pdf' tidak ditemukan dalam permintaan"}), 400
    file = request.files['cv_pdf'] #
    if file.filename == '': #
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    filepath = None
    if file:
        filename = secure_filename(file.filename) #
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) #
        try:
            file.save(filepath) #
            app.logger.info(f"File '{filename}' berhasil disimpan sementara.")
            cv_text = matcher.extract_text_from_pdf(filepath) #

            if not cv_text: #
                return jsonify({
                    "status": "error",
                    "message": "Gagal mengekstrak teks dari PDF.",
                    "data": [],
                    "debug_info": "Fungsi extract_text_from_pdf mengembalikan string kosong."
                }), 500

            # Menangani respons dictionary dari find_matching_jobs
            match_result = matcher.find_matching_jobs(cv_text, top_k=10) #
            
            jobs = match_result.get("jobs", [])
            debug_reason = match_result.get("debug_reason", "Tidak ada pesan debug.")
            
            app.logger.info(f"Debug Info: {debug_reason}")
            app.logger.info(f"Menemukan {len(jobs)} pekerjaan yang cocok untuk '{filename}'.")
            
            return jsonify({
                "status": "success",
                "message": "Pekerjaan berhasil dicocokkan",
                "data": jobs, #
                "debug_info": debug_reason
            })

        except Exception as e:
            app.logger.error(f"Terjadi error: {e}", exc_info=True)
            return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500
        finally:
            if filepath and os.path.exists(filepath):
                os.remove(filepath) #
                app.logger.info(f"File sementara dihapus: {filepath}")

    return jsonify({"error": "File tidak valid"}), 400

# ENDPOINT UNTUK TESTING
@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    app.logger.info(f"Endpoint /test dipanggil dengan metode: {request.method}")
    return jsonify({
        "status": "success",
        "message": "Endpoint tes berfungsi!",
        "metode_yang_digunakan": request.method
    })