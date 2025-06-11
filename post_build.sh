#!/bin/bash

# Script ini akan dijalankan setelah proses build Oryx selesai
echo "Menjalankan post-build script..."

# Mengunduh data NLTK yang diperlukan ke dalam direktori yang akan dipaketkan
python -m nltk.downloader -d /home/site/wwwroot/nltk_data stopwords punkt

echo "Proses download data NLTK selesai."