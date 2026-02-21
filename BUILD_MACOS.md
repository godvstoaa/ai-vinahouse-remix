# 🍎 Hướng dẫn Build cho macOS

## 📋 Yêu cầu hệ thống

### Trên macOS:
- **macOS 10.13+** (High Sierra trở lên)
- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **Node.js 18+**: `brew install node`
- **Python 3.10+**: `brew install python@3.10`

## 🚀 Các bước build

### 1. Clone/Copy project về macOS

```bash
# Nếu từ git
git clone <repository-url>
cd ai-remix-app

# Hoặc copy folder trực tiếp
```

### 2. Cài đặt dependencies

```bash
# Install Node dependencies (bao gồm Electron)
npm install

# Install Python dependencies
cd backend
pip3 install -r requirements.txt
cd ..
```

### 3. Build cho macOS

```bash
# Build cho Mac (universal: cả Intel và Apple Silicon)
npm run electron:build:mac

# Hoặc build riêng cho từng architecture:
# Apple Silicon (M1/M2/M3)
npm run electron:build:mac-arm

# Intel (x64)
npm run electron:build:mac-x64

# Chỉ build DMG
npm run electron:build:dmg

# Chỉ build ZIP
npm run electron:build:zip
```

### 4. Kiểm tra output

Sau khi build xong, file cài đặt sẽ ở:
```
release/
├── AI Vinahouse Remix-1.0.0-arm64.dmg      # Apple Silicon
├── AI Vinahouse Remix-1.0.0-x64.dmg        # Intel
├── AI Vinahouse Remix-1.0.0-arm64.zip      # ZIP cho Apple Silicon
└── AI Vinahouse Remix-1.0.0-x64.zip        # ZIP cho Intel
```

## 📦 Cấu trúc file cài đặt

File DMG sẽ bao gồm:
- Ứng dụng AI Vinahouse Remix.app
- Backend Python với các dependencies
- Pre-trained models (nếu có trong folder `pretrained/`)

## 🔧 Tùy chỉnh build

### Thay đổi icon
1. Tạo icon file `.icns`:
   - Chuẩn bị icon 1024x1024 px (PNG)
   - Dùng tool như `iconutil` hoặc online converter
   
2. Đặt file vào: `build/icon.icns`

### Thay đổi thông tin app
Chỉnh sửa trong `electron-builder.json`:
```json
{
    "appId": "com.yourcompany.appname",
    "productName": "Tên App của bạn"
}
```

### Thêm pre-trained models
Đặt các file `.pt`, `.pth`, `.bin` vào folder `pretrained/` trước khi build.

## 🐛 Khắc phục lỗi thường gặp

### Lỗi: "electron-builder not found"
```bash
npm install electron-builder --save-dev
```

### Lỗi: "Python not found"
```bash
# Tạo virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Lỗi: "Code signing failed"
- Cần Apple Developer account để ký code
- Hoặc build không ký: thêm `"identity": null` vào config mac

### Lỗi: "App is damaged and can't be opened"
```bash
# Remove quarantine attribute
xattr -cr "AI Vinahouse Remix.app"
```

## 📤 Phân phối

### Upload lên GitHub Releases
1. Tạo release mới trên GitHub
2. Upload file DMG/ZIP
3. Users có thể download và cài đặt

### Notarization (yêu cầu Apple Developer)
```bash
# Notarize app
xcrun notarytool submit "release/AI Vinahouse Remix-1.0.0.dmg" \
    --apple-id "your@email.com" \
    --password "@keychain:AC_PASSWORD" \
    --team-id "TEAM_ID" \
    --wait
```

## 💻 Chạy development mode

```bash
# Chạy frontend + backend + Electron
npm run electron:dev

# Hoặc chạy riêng:
# Terminal 1: Frontend
npm run dev

# Terminal 2: Backend
cd backend && python main.py

# Terminal 3: Electron
electron .
```

## 📊 Kích thước file dự kiến

| Architecture | DMG Size | ZIP Size |
|-------------|----------|----------|
| arm64 (M1/M2) | ~500MB | ~450MB |
| x64 (Intel) | ~520MB | ~470MB |

*Size có thể lớn hơn nếu bao gồm pre-trained models*

## 🎯 Tối ưu kích thước

1. **Loại bỏ file không cần thiết**:
   - Xóa `node_modules/.cache`
   - Xóa `__pycache__`
   
2. **Sử dụng production build**:
   ```bash
   npm run build  # Vite sẽ minify code
   ```

3. **Nén assets**:
   - Image: WebP format
   - Audio: MP3 thay vì WAV

## ✅ Checklist trước khi build

- [ ] Test đầy đủ trên development mode
- [ ] Update version trong package.json
- [ ] Kiểm tra icon app
- [ ] Đảm bảo backend hoạt động
- [ ] Xóa cache và file tạm
- [ ] Có đủ pre-trained models (nếu cần)

---

**Lưu ý**: Build trên macOS sẽ cho kết quả tốt nhất. Cross-compile từ Windows/Linux có thể gặp vấn đề với code signing và notarization.