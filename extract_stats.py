"""
Script untuk mengekstrak GLOBAL_MEAN dan GLOBAL_STD dari data training
Jalankan script ini setelah training untuk mendapatkan nilai yang perlu
di-update ke app.py
"""

import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm

# Konstanta (sama dengan training)
SAMPLE_RATE = 22050
SOUND_DURATION = 2.95
HOP_LENGTH = 512
WINDOW_LENGTH = 512
N_MELS = 128

def extract_features_for_stats(df, audio_path, custom_folder):
    """Ekstrak fitur untuk menghitung mean dan std"""
    X = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        # Tentukan jalur file
        if row['fold'] <= 10:
            file_path = os.path.join(audio_path, f"fold{row['fold']}", row["slice_file_name"])
        else:
            file_path = os.path.join(custom_folder, row["slice_file_name"])
        
        try:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=SOUND_DURATION)
            mel_spec = librosa.feature.melspectrogram(
                y=signal, sr=sr,
                hop_length=HOP_LENGTH, 
                win_length=WINDOW_LENGTH,
                n_mels=N_MELS
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = librosa.util.fix_length(log_mel, size=128, axis=1)
            X.append(log_mel.T)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(X)

def main():
    print("="*70)
    print("EKSTRAKSI GLOBAL_MEAN DAN GLOBAL_STD UNTUK DEPLOYMENT")
    print("="*70)
    
    # METODE 1: Jika Anda punya file .npy dari training
    print("\n🔍 Metode 1: Loading dari file .npy")
    try:
        X = np.load("X_gabungan.npy")
        print(f"✅ Berhasil load X_gabungan.npy")
        print(f"   Shape: {X.shape}")
        
        # Hitung statistik untuk data training (fold 1-10)
        # Asumsikan fold terakhir adalah fold 10
        mean = X.mean()
        std = X.std()
        
        print(f"\n📊 HASIL PERHITUNGAN:")
        print(f"   GLOBAL_MEAN = {mean}")
        print(f"   GLOBAL_STD  = {std}")
        
        # Simpan ke file
        np.save("global_mean.npy", mean)
        np.save("global_std.npy", std)
        print(f"\n💾 Nilai disimpan ke 'global_mean.npy' dan 'global_std.npy'")
        
        # Cetak untuk copy-paste
        print("\n" + "="*70)
        print("📋 COPY-PASTE KE app.py:")
        print("="*70)
        print(f"GLOBAL_MEAN = {mean}")
        print(f"GLOBAL_STD = {std}")
        print("="*70)
        
        return True
        
    except FileNotFoundError:
        print("❌ File X_gabungan.npy tidak ditemukan")
        print("   Lanjut ke Metode 2...\n")
    
    # METODE 2: Ekstrak ulang dari audio files
    print("\n🔍 Metode 2: Ekstrak dari audio files")
    print("⚠️  Ini akan memakan waktu lama!\n")
    
    response = input("Lanjutkan ekstraksi? (y/n): ")
    if response.lower() != 'y':
        print("Dibatalkan.")
        return False
    
    # Path (sesuaikan dengan struktur Anda)
    metadata_path = input("Path ke UrbanSound8K.csv: ").strip() or "UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_path = input("Path ke folder audio US8K: ").strip() or "UrbanSound8K/audio/"
    custom_folder = input("Path ke folder custom audio: ").strip() or "/content/drive/MyDrive/Dataset Deep Learning/wavs/"
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
        print(f"✅ Loaded metadata: {len(df)} samples")
    except FileNotFoundError:
        print(f"❌ File {metadata_path} tidak ditemukan!")
        return False
    
    # Ekstrak fitur
    print("\n🔄 Mengekstrak fitur...")
    X = extract_features_for_stats(df, audio_path, custom_folder)
    
    if len(X) > 0:
        # Hitung statistik
        mean = X.mean()
        std = X.std()
        
        print(f"\n📊 HASIL PERHITUNGAN:")
        print(f"   GLOBAL_MEAN = {mean}")
        print(f"   GLOBAL_STD  = {std}")
        
        # Simpan
        np.save("X_gabungan.npy", X)
        np.save("global_mean.npy", mean)
        np.save("global_std.npy", std)
        print(f"\n💾 Hasil disimpan")
        
        # Cetak untuk copy-paste
        print("\n" + "="*70)
        print("📋 COPY-PASTE KE app.py:")
        print("="*70)
        print(f"GLOBAL_MEAN = {mean}")
        print(f"GLOBAL_STD = {std}")
        print("="*70)
        
        return True
    else:
        print("❌ Tidak ada data yang berhasil diekstrak")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Selesai! Sekarang update nilai di app.py")
    else:
        print("\n❌ Gagal mengekstrak statistik")