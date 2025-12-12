# 🔊 Aplikasi Klasifikasi Suara - Embung A ITERA

Aplikasi web untuk klasifikasi suara menggunakan model Transformer dengan Transfer Learning.

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan Transfer Learning menggunakan model Transformer untuk klasifikasi suara di Embung A ITERA. Model dilatih menggunakan dataset UrbanSound8K dan data primer dari Embung A ITERA.

## 🚀 Cara Deploy ke Streamlit Cloud

### Persiapan

1. **Simpan model dan file yang diperlukan:**
   ```
   project_folder/
   ├── app.py
   ├── requirements.txt
   ├── transformer_final_model.pth
   ├── README.md
   └── .streamlit/
       └── config.toml (opsional)
   ```

2. **Update nilai GLOBAL_MEAN dan GLOBAL_STD di app.py:**
   - Buka `app.py`
   - Cari baris:
     ```python
     GLOBAL_MEAN = 0.0  # Ganti dengan nilai dari training
     GLOBAL_STD = 1.0   # Ganti dengan nilai dari training
     ```
   - Ganti dengan nilai aktual dari training model Anda

### Deploy ke Streamlit Cloud

1. **Push ke GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Audio Classification App"
   git branch -M main
   git remote add origin https://github.com/username/repo-name.git
   git push -u origin main
   ```

2. **Upload model file:**
   - Karena file `.pth` besar, gunakan Git LFS:
     ```bash
     git lfs install
     git lfs track "*.pth"
     git add .gitattributes
     git add transformer_final_model.pth
     git commit -m "Add model file"
     git push
     ```
   
   - **Alternatif:** Upload model ke Google Drive/Dropbox dan download saat runtime

3. **Deploy di Streamlit Cloud:**
   - Buka https://share.streamlit.io/
   - Klik "New app"
   - Pilih repository GitHub Anda
   - Set main file: `app.py`
   - Klik "Deploy"

### Deploy Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## 📦 Struktur File

- `app.py` - File utama aplikasi Streamlit
- `requirements.txt` - Daftar dependencies
- `transformer_final_model.pth` - Bobot model yang sudah dilatih
- `README.md` - Dokumentasi proyek

## 🎯 Fitur Aplikasi

1. **Upload Audio File** - Support format .wav, .mp3, .ogg
2. **Real-time Classification** - Klasifikasi audio dengan confidence score
3. **Visualisasi:**
   - Audio Waveform
   - Log-Mel Spectrogram
   - Probability Distribution
4. **Top 3 Predictions** - Menampilkan 3 prediksi teratas dengan probabilitas

## 🔧 Konfigurasi Model

- **Sample Rate:** 22050 Hz
- **Duration:** 2.95 detik
- **Mel Bins:** 128
- **Model Dimension:** 128
- **Attention Heads:** 4
- **Encoder Layers:** 2

## 📊 Kelas yang Dapat Dikenali

1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music
11. Custom Sound (Embung A ITERA)

## 🛠️ Troubleshooting

### Error: Model file not found
Pastikan file `transformer_final_model.pth` ada di folder yang sama dengan `app.py`

### Error: librosa installation
Jika ada masalah dengan librosa:
```bash
pip install librosa --upgrade
pip install numba --upgrade
```

### Memory Error saat deploy
Jika model terlalu besar:
1. Gunakan model quantization
2. Atau upload model ke cloud storage dan download saat runtime

## 📝 Cara Mendapatkan GLOBAL_MEAN dan GLOBAL_STD

Dari kode training Anda, tambahkan kode ini setelah fold terakhir:
```python
# Simpan mean dan std
np.save("global_mean.npy", GLOBAL_MEAN)
np.save("global_std.npy", GLOBAL_STD)
print(f"GLOBAL_MEAN: {GLOBAL_MEAN}")
print(f"GLOBAL_STD: {GLOBAL_STD}")
```

Kemudian copy nilai tersebut ke `app.py`

## 🔗 Alternatif: Download Model dari Cloud

Jika file model terlalu besar untuk GitHub, gunakan cara ini:

```python
import gdown

@st.cache_resource
def load_model():
    # Download dari Google Drive
    url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
    output = 'transformer_final_model.pth'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    # Load model
    device = torch.device("cpu")
    model = TransformerClassifier(...).to(device)
    model.load_state_dict(torch.load(output, map_location=device))
    model.eval()
    return model, device
```

Tambahkan `gdown` ke requirements.txt

## 👨‍💻 Developer

Institut Teknologi Sumatera (ITERA)
Deep Learning Project - Klasifikasi Suara Embung A

## 📄 License

MIT License