# 🎵 AI Vinahouse Remix Studio Pro

Công cụ tạo Remix Vinahouse chuyên nghiệp với công nghệ Deep Learning 2026.

![Version](https://img.shields.io/badge/version-2.0.0-purple)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![React](https://img.shields.io/badge/react-18+-cyan)
![CUDA](https://img.shields.io/badge/cuda-optional-green)

## ✨ Tính Năng Chính

### 🎧 Tách Nhạc (Source Separation)
- **Demucs v4** (htdemucs) - Meta AI
- **MDX23C** - MDX Challenge winner
- **UVR5** - Tách vocal chất lượng cao
- Tách 4 stems: Vocals, Drums, Bass, Other

### 🚀 Fast Training - Kho Model AI
**11 models có sẵn:**

| Model | Loại | Size | VRAM | Mô tả |
|-------|------|------|------|-------|
| HT-Demucs v4 | Separation | 150MB | 4GB | Meta AI - Tách nhạc tốt nhất |
| Demucs MDX23C | Separation | 180MB | 5GB | MDX Challenge winner |
| UVR5-MDX-VIP | Separation | 250MB | 6GB | Tách vocal xuất sắc |
| RVC v2 Base | Voice | 190MB | 4GB | Chuyển giọng hát AI |
| RVC v2 Vinahouse | Voice | 200MB | 4GB | Fine-tune cho Vinahouse |
| RAVE Vinahouse | Generation | 50MB | 2GB | Đã train sẵn Vinahouse |
| RAVE v2 | Generation | 60MB | 3GB | Real-time generation |
| MusicGen Small | Generation | 300MB | 6GB | Text-to-Music (300M) |
| MusicGen Medium | Generation | 1.2GB | 10GB | Text-to-Music (1.5B) |
| So-VITS-SVC 4.1 | Voice | 220MB | 6GB | Singing Voice Conversion |
| AudioCraft Enhance | Enhancement | 500MB | 8GB | Nâng cao chất lượng âm thanh |

### 🧠 AI Training (DiT)
- Training Diffusion Transformer từ đầu
- Dataset processing cho DJ sets dài 2-5 giờ
- Hierarchical chunking (30s segments)
- Spectrogram caching

### 🔐 Watermark
- Embed watermark siêu âm (19kHz)
- Bảo vệ bản quyền sản phẩm
- Detect watermark từ file audio

## 🚀 Cài Đặt

### Yêu cầu
- Python 3.10+
- Node.js 18+
- CUDA (optional, để train model)

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Server chạy tại: http://localhost:8002

### Frontend
```bash
npm install
npm run dev
```
App chạy tại: http://localhost:3000

## 📖 Hướng Dẫn Sử Dụng

### 1. Tạo Remix
1. Upload file nhạc (MP3/WAV/FLAC)
2. Chọn preset (Vinahouse Classic, Modern, Bounce...)
3. Điều chỉnh BPM, Energy, Bass Boost
4. Nhấn "🔥 Tạo Vinahouse Remix"
5. Xuất file MP3/WAV

### 2. Fast Training
1. Vào tab "🧠 Huấn Luyện"
2. Chọn phương pháp: QLoRA (nhanh nhất) hoặc LoRA
3. Chọn model từ Kho Model AI
4. Nhập đường dẫn dataset
5. Nhấn "🚀 Bắt Đầu Training"

### 3. Training DiT
- Dành cho user nâng cao
- Cần GPU 16GB+ VRAM
- Dataset 1000+ bài

## 🛠️ Tech Stack

### Backend
- **FastAPI** - REST API framework
- **PyTorch** - Deep learning
- **Demucs** - Source separation
- **Librosa** - Audio processing
- **Pedalboard** - VST-grade effects

### Frontend
- **React 18** - UI framework
- **TailwindCSS** - Styling
- **TypeScript** - Type safety

## 📁 Cấu Trúc Project

```
ai-remix-app/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── routers/
│   │   └── training.py      # Training API
│   ├── services/
│   │   ├── source_separator.py
│   │   ├── genre_detector.py
│   │   ├── audio_analyzer.py
│   │   └── remix_generator.py
│   └── training/
│       ├── train_script.py
│       ├── data_prep_pipeline.py
│       └── emotional_tagger.py
├── src/
│   ├── App.tsx
│   └── components/
│       ├── FastTrainingPanel.tsx
│       ├── TrainingPanel.tsx
│       └── ...
└── README.md
```

## ⌨️ Phím Tắt

| Phím | Chức năng |
|------|-----------|
| Space | Play/Pause |
| R | Tạo Remix |
| S | Tách nhạc |
| E | Xuất file |
| 1-5 | Chuyển tab |

## 📄 License

MIT License - Tự do sử dụng cho mục đích cá nhân và thương mại.

## 🙏 Credits

- Meta AI - Demucs, MusicGen
- RVC Community - Voice Conversion
- ACIDS - RAVE
- Various open-source contributors

---

Made with ❤️ for Vinahouse producers