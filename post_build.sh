#!/bin/bash

# Script ini akan dijalankan setelah proses build Oryx selesai
echo "Menjalankan post-build script..."

# Menambahkan punkt_tab ke daftar download
python -m nltk.downloader -d /home/site/wwwroot/nltk_data stopwords punkt punkt_tab

echo "Proses download data NLTK selesai."