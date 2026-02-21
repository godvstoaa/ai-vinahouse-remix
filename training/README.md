# AI Remix Model Training Module

Module đào tạo mô hình Diffusion Transformer (DiT) để tạo Vinahouse Remix từ nhạc gốc.

## 📋 Yêu cầu hệ thống

| Phần cứng | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| GPU | RTX 4070 (8GB VRAM) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| SSD | 100GB free | 500GB+ (cho data) |

## 🚀 Cài đặt

```bash
# Cài đặt dependencies training
pip install -r requirements_train.txt
```

## 📁 Cấu trúc thư mục

```
ai-remix-app/
├── config/
│   └── train_config.yaml      # Cấu hình training
├── training/
│   ├── emotional_tagger.py    # Phân tích cảm xúc nhạc Nonstop
│   ├── data_prep_pipeline.py  # Cắt/chuẩn bị data
│   ├── dit_model_architecture.py  # Kiến trúc DiT
│   └── train_script.py        # Script training chính
├── datasets/
│   ├── processed/             # Audio segments đã cắt
│   ├── metadata/              # JSON metadata
│   └── cache/                 # Mel spectrograms cache
└── checkpoints/               # Model checkpoints
```

## 🔧 Sử dụng

### Bước 1: Cấu hình đường dẫn data

Chỉnh sửa `config/train_config.yaml`:

```yaml
data:
  vinahouse_dir: "D:/Music/Vinahouse/"  # Đường dẫn tới 10k bài Vinahouse
  nonstop_dir: "D:/Music/Nonstop/"      # Đường dẫn tới 2k bài Nonstop
```

### Bước 2: Chạy Emotional Tagger (cho Nonstop)

```bash
cd ai-remix-app

# Phân tích năng lượng cảm xúc từ Nonstop tracks
python -m training.emotional_tagger D:/Music/Nonstop/ --output datasets/metadata/
```

Output: Files JSON chứa metadata với energy events (build, drop, breakdown).

### Bước 3: Chạy Data Prep Pipeline

```bash
# Cắt 10k bài thành segments 30 giây
python -m training.data_prep_pipeline D:/Music/Vinahouse/ --workers 4
```

**Lưu ýVRAM:**
- Process sẽ không load toàn bộ data vào RAM
- Dùng streaming để xử lý từng file
- Kiểm tra RAM trong Task Manager

### Bước 4: Chạy Training

```bash
# Bắt đầu training
accelerate launch -m training.train_script --config config/train_config.yaml

# Hoặc đơn giản:
python -m training.train_script
```

**Theo dõi training:**
```bash
tensorboard --logdir checkpoints/logs/
```

Mở http://localhost:6006 để xem loss curve.

## ⚙️ Cấu hình tối ưu 8GB VRAM

Trong `config/train_config.yaml`:

```yaml
mixed_precision: "fp16"              # BẮT BUỘC
gradient_accumulation_steps: 16      # Tích lũy 16 bước

training:
  batch_size: 2                      # GIỮ THẤP!
  
model:
  hidden_size: 768                   # DiT-Base
  num_layers: 12
```

**VRAM usage kỳ vọng:** ~7.1GB / 8.0GB

## 📊 Metrics

### Acceptance Criteria

| Metric | Target |
|--------|--------|
| VRAM usage | < 7.5GB |
| Loss after 10 epochs | Decreasing |
| Drop detection accuracy | > 80% |

### Monitor với TensorBoard

- `train/loss`: Phải giảm đều
- `train/learning_rate`: Warmup rồi giảm
- `system/gpu_memory_gb`: Không vượt 7.5GB

## 🔥 Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm `batch_size` xuống 1
2. Tăng `gradient_accumulation_steps` lên 32
3. Giảm `hidden_size` xuống 384 (DiT-Small)

### Loss không giảm

1. Kiểm tra learning rate (có thể quá cao)
2. Kiểm tra data quality
3. Tăng `warmup_steps`

### File không load được

1. Kiểm tra đường dẫn trong config
2. Đảm bảo file audio không bị corrupt
3. Chạy với `--max_files 10` để test

## 📝 File outputs

Sau khi training:

```
checkpoints/
├── checkpoint_1000.pt    # Checkpoint bước 1000
├── checkpoint_2000.pt
├── final_model.pt        # Model cuối cùng
└── logs/                 # TensorBoard logs
```

## 🎯 Next Steps

1. **Fine-tune**: Điều chỉnh hyperparameters dựa trên loss curve
2. **Generate samples**: Dùng model để tạo test remix
3. **Integrate**: Kết nối với backend API

---

**Hardware Tips:**
- Tắt Windows Update trước khi train 7-14 ngày
- Đảm bảo máy mát mẻ (điều hòa 24°C)
- Dùng UPS nếu điện không ổn định