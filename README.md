# 🎵 AI Auto-Remix Music Application

Ứng dụng AI tự động remix nhạc sử dụng Machine Learning để phân tích và tạo remix chất lượng cao.

![AI Remix Studio](https://img.shields.io/badge/AI-Remix%20Studio-purple)
![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

## ✨ Features

### 🎤 Source Separation
- Tách audio thành 4 stems: **Vocals, Drums, Bass, Other**
- Sử dụng Demucs (state-of-the-art source separation)

### 🎼 Audio Analysis
- **BPM Detection** - Phát hiện tempo tự động
- **Key Detection** - Phát hiện key (C Major, A Minor, etc.)
- **Genre Classification** - Phân loại dòng nhạc (EDM, Pop, Hip-hop, House, etc.)
- **Energy Analysis** - Phân tích năng lượng bài hát
- **Section Detection** - Phát hiện cấu trúc (Intro, Verse, Chorus, Outro)

### 🔧 Remix Controls
- **Genre Selector** - Chọn style remix (EDM, House, Trap, Lo-fi, etc.)
- **BPM Slider** - Điều chỉnh tempo (60-200 BPM)
- **Energy Level** - Điều chỉnh năng lượng
- **Effects**: Reverb, Delay, Filter

### 🎨 UI Features
- Drag & Drop upload
- Real-time Waveform visualization
- Stem Player (nghe từng track riêng)
- Preview Player
- Export WAV/MP3
- Remix History

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- (Optional) NVIDIA GPU cho ML acceleration

### 1. Clone & Install Frontend

```bash
cd ai-remix-app
npm install
```

### 2. Install Python Backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start Backend Server

```bash
cd backend
python main.py
# hoặc
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### 4. Start Frontend

```bash
cd ..
npm run dev
```

Frontend runs at: http://localhost:5173

## 📁 Project Structure

```
ai-remix-app/
├── src/
│   ├── components/
│   │   ├── Dashboard.tsx       # Main dashboard
│   │   ├── UploadArea.tsx      # File upload
│   │   ├── WaveformDisplay.tsx # Audio visualization
│   │   ├── StemPlayer.tsx      # Individual stems
│   │   ├── RemixControls.tsx   # Remix parameters
│   │   ├── AudioAnalysis.tsx   # Analysis results
│   │   ├── ProcessingStatus.tsx# Progress indicator
│   │   └── RemixHistory.tsx    # Past remixes
│   ├── store/
│   │   └── remixStore.ts       # Zustand state
│   ├── services/
│   │   └── api.ts              # Backend API
│   └── App.tsx
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Python deps
│   └── services/
│       ├── source_separator.py # Stem separation
│       ├── genre_detector.py   # Genre classification
│       ├── audio_analyzer.py   # BPM/Key detection
│       └── remix_generator.py  # Remix creation
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload audio file |
| POST | `/api/analyze/{job_id}` | Analyze audio |
| POST | `/api/separate/{job_id}` | Separate stems |
| POST | `/api/remix/{job_id}` | Generate remix |
| GET | `/api/status/{job_id}` | Get job status |
| GET | `/api/download/stem/{job_id}/{stem}` | Download stem |
| GET | `/api/download/remix/{job_id}` | Download remix |

## 🎛️ Genre Presets

| Genre | Target BPM | Characteristics |
|-------|------------|-----------------|
| EDM | 128-150 | High energy, drop style |
| House | 120-130 | Steady beat, bass boost |
| Trap | 130-170 | Heavy drums, bass |
| Hip-hop | 80-115 | Vocal forward |
| Lo-fi | 60-90 | Mellow, soft |
| Techno | 125-150 | Minimal, driving |
| Dubstep | 140-150 | Heavy bass, drops |

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI Framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **Zustand** - State management
- **Vite** - Build tool

### Backend
- **FastAPI** - Python web framework
- **Librosa** - Audio analysis
- **Demucs** - Source separation
- **PyTorch** - ML inference
- **SoundFile** - Audio I/O

## 📝 Notes

### GPU Support
- Install CUDA-enabled PyTorch for faster processing:
```bash
pip install torch+cuda -f https://download.pytorch.org/whl/torch_stable.html
```

### Fallback Mode
- Nếu Demucs không available, app sẽ dùng frequency-based separation (chất lượng thấp hơn)

### Known Limitations
- Source separation quality depends on Demucs model
- Large files (>50MB) may take time to process
- Genre detection is heuristic-based (not ML model)

## 📄 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

Made with ❤️ using AI/ML