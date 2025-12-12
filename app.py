import streamlit as st
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
import matplotlib.pyplot as plt
import librosa.display

# ==================== KONFIGURASI ====================
SAMPLE_RATE = 22050
SOUND_DURATION = 2.95
HOP_LENGTH = 512
WINDOW_LENGTH = 512
N_MELS = 128
MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
NUM_CLASSES = 11  # 10 kelas US8K + 1 kelas Custom_Sound

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music', 'Custom_Sound'
]

# Mean dan Std dari training (ganti dengan nilai aktual dari model Anda)
GLOBAL_MEAN = -35.268856048583984  # Ganti dengan nilai dari training
GLOBAL_STD = 19.743492126464844   # Ganti dengan nilai dari training

# ==================== MODEL ARCHITECTURE ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ==================== FUNGSI HELPER ====================
@st.cache_resource
def load_model():
    """Load model yang sudah dilatih"""
    device = torch.device("cpu")
    model = TransformerClassifier(
        input_dim=N_MELS,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('transformer_final_model.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ File model 'transformer_final_model.pth' tidak ditemukan!")
        st.stop()

def extract_features(audio_file, mean, std):
    """Ekstraksi fitur dari file audio"""
    try:
        # Load audio
        signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=SOUND_DURATION)
        
        # Ekstraksi Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            n_mels=N_MELS
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = librosa.util.fix_length(log_mel, size=128, axis=1)
        
        # Transpose dan normalisasi
        log_mel_T = log_mel.T
        log_mel_normalized = (log_mel_T - mean) / std
        
        return log_mel_normalized, signal, sr, log_mel
    except Exception as e:
        st.error(f"Error saat memproses audio: {e}")
        return None, None, None, None

def predict_audio(model, features, device):
    """Prediksi kelas audio"""
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class_id, probabilities

# ==================== STREAMLIT UI ====================
def main():
    # Page config
    st.set_page_config(
        page_title="Klasifikasi Suara - Embung A ITERA",
        page_icon="🔊",
        layout="wide"
    )
    
    # Header
    st.title("🔊 Klasifikasi Suara Menggunakan Transformer")
    st.markdown("### Implementasi Transfer Learning untuk Klasifikasi Suara di Embung A ITERA")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4CAF50/FFFFFF?text=ITERA", use_column_width=True)
        st.markdown("## 📋 Informasi Proyek")
        st.info("""
        **Model:** Transformer Encoder
        
        **Dataset:**
        - UrbanSound8K (10 kelas)
        - Data Primer Embung A ITERA
        
        **Kelas yang Dapat Dikenali:**
        - Air Conditioner
        - Car Horn
        - Children Playing
        - Dog Bark
        - Drilling
        - Engine Idling
        - Gun Shot
        - Jackhammer
        - Siren
        - Street Music
        - Custom Sound (Embung A)
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Spesifikasi Model")
        st.write(f"- **Sample Rate:** {SAMPLE_RATE} Hz")
        st.write(f"- **Duration:** {SOUND_DURATION}s")
        st.write(f"- **Mel Bins:** {N_MELS}")
        st.write(f"- **Model Dim:** {MODEL_DIM}")
        st.write(f"- **Attention Heads:** {NUM_HEADS}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📁 Upload File Audio")
        uploaded_file = st.file_uploader(
            "Pilih file audio (.wav, .mp3, .ogg)",
            type=['wav', 'mp3', 'ogg']
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            # Tombol prediksi
            if st.button("🚀 Klasifikasi Audio", type="primary"):
                with st.spinner("Memproses audio..."):
                    # Load model
                    model, device = load_model()
                    
                    # Ekstraksi fitur
                    features, signal, sr, log_mel = extract_features(
                        uploaded_file, GLOBAL_MEAN, GLOBAL_STD
                    )
                    
                    if features is not None:
                        # Prediksi
                        predicted_id, probabilities = predict_audio(model, features, device)
                        predicted_class = CLASS_NAMES[predicted_id]
                        confidence = probabilities[0, predicted_id].item()
                        
                        # Simpan hasil ke session state
                        st.session_state['prediction'] = predicted_class
                        st.session_state['confidence'] = confidence
                        st.session_state['probabilities'] = probabilities
                        st.session_state['signal'] = signal
                        st.session_state['sr'] = sr
                        st.session_state['log_mel'] = log_mel
    
    with col2:
        st.markdown("## 📊 Hasil Klasifikasi")
        
        if 'prediction' in st.session_state:
            # Display hasil prediksi
            st.success("✅ Klasifikasi Berhasil!")
            
            # Hasil utama
            st.markdown(f"### Prediksi Kelas: **{st.session_state['prediction']}**")
            st.markdown(f"### Confidence: **{st.session_state['confidence']:.2%}**")
            
            # Progress bar untuk confidence
            st.progress(st.session_state['confidence'])
            
            st.markdown("---")
            
            # Top 3 prediksi
            st.markdown("#### 🏆 Top 3 Prediksi:")
            probs = st.session_state['probabilities']
            top_k = min(3, NUM_CLASSES)
            top_p, top_class = probs.topk(top_k, dim=1)
            
            for i in range(top_k):
                class_idx = top_class[0][i].item()
                class_name = CLASS_NAMES[class_idx]
                prob = top_p[0][i].item()
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{i+1}. {class_name}**")
                with col_b:
                    st.write(f"{prob:.2%}")
                st.progress(prob)
        else:
            st.info("👆 Upload file audio dan klik tombol klasifikasi untuk melihat hasil")
    
    # Visualisasi
    if 'signal' in st.session_state:
        st.markdown("---")
        st.markdown("## 📈 Visualisasi Audio")
        
        tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🎵 Mel Spectrogram", "📊 Probability Distribution"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(
                y=st.session_state['signal'],
                sr=st.session_state['sr'],
                ax=ax
            )
            ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(
                st.session_state['log_mel'],
                sr=st.session_state['sr'],
                hop_length=HOP_LENGTH,
                x_axis='time',
                y_axis='mel',
                ax=ax,
                cmap='viridis'
            )
            ax.set_title('Log-Mel Spectrogram', fontsize=14, fontweight='bold')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
        
        with tab3:
            probs_np = st.session_state['probabilities'][0].cpu().numpy()
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(CLASS_NAMES, probs_np, color='skyblue')
            
            # Highlight prediksi tertinggi
            max_idx = np.argmax(probs_np)
            bars[max_idx].set_color('green')
            
            ax.set_xlabel('Probability', fontweight='bold')
            ax.set_title('Probability Distribution Across Classes', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Tambahkan nilai di setiap bar
            for i, (bar, prob) in enumerate(zip(bars, probs_np)):
                ax.text(prob + 0.01, i, f'{prob:.3f}', va='center')
            
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Developed for Deep Learning Project - ITERA</p>
        <p>Model: Transformer Encoder | Dataset: UrbanSound8K + Custom Data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()