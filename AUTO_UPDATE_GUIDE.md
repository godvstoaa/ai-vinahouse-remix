
## 📋 Tổng quan

Hệ thống Auto-Update cho phép:
- ✅ Tự động kiểm tra update khi mở app
- ✅ Thông báo khi có version mới
- ✅ Download update trong background
- ✅ Cài đặt update khi user đồng ý
- ✅ Hiển thị progress bar khi download

---

## 🚀 Cách sử dụng

### 1. Trên App (Mac)

Khi mở app, hệ thống sẽ tự động kiểm tra update. Nếu có update mới:

1. **Notification hiện lên** với thông tin version mới
2. Click **"Download"** để tải
3. Xem **progress bar** khi đang download
4. Click **"Restart & Install"** để cài đặt

### 2. Nút Check Update

Trong header của app có nút **"Check Update"**:
- Click để kiểm tra update thủ công
- Hiển thị version hiện tại

---

## 🔧 Workflow cho Developer

### Bước 1: Chỉnh sửa code trên PC
```bash
# Chỉnh sửa các file cần thiết
# Ví dụ: sửa component, thêm tính năng, fix bug...
```

### Bước 2: Tăng version
```json
// package.json
{
    "version": "1.0.0" → "1.0.1"
}
```

### Bước 3: Build version mới
```bash
npm run electron:build:mac-arm
```

### Bước 4: Upload lên server
Copy file DMG từ folder `release/` lên:
- GitHub Releases (recommended)
- Hoặc server riêng

---

## 📦 Cấu hình Update Server

### Option 1: GitHub Releases (Recommended)

1. Tạo repo trên GitHub
2. Push code lên
3. Tạo Release mới
4. Upload file DMG/ZIP

Cập nhật `electron-builder.json`:
```json
{
    "publish": {
        "provider": "github",
        "owner": "your-username",
        "repo": "ai-vinahouse-remix"
    }
}
```

### Option 2: Server riêng

```json
{
    "publish": {
        "provider": "generic",
        "url": "https://your-server.com/updates/"
    }
}
```

Cấu trúc folder trên server:
```
/updates/
├── latest-mac.yml          # Metadata
├── AI Vinahouse Remix-1.0.1-arm64.dmg
├── AI Vinahouse Remix-1.0.1-arm64.zip
└── ...
```

---

## 📁 Files liên quan

| File | Mô tả |
|------|-------|
| `electron/main.js` | Auto-updater logic |
| `electron/preload.js` | IPC bridge |
| `src/components/UpdateButton.tsx` | UI component |
| `electron-builder.json` | Build config |
| `package.json` | Version number |

---

## 🔐 Code Signing (Optional nhưng recommended)

Để app được macOS chấp nhận, cần sign:

1. **Apple Developer Certificate**
2. **Notarization** với Apple

```bash
# Sign app
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" "release/AI Vinahouse Remix.app"

# Notarize
xcrun notarytool submit "release/AI Vinahouse Remix.dmg" --apple-id "your@email.com" --password "app-specific-password" --team-id "TEAMID"
```

---

## ⚡ Quick Commands

```bash
# Build cho Mac M4 (Apple Silicon)
npm run electron:build:mac-arm

# Build cho Mac Intel
npm run electron:build:mac-x64

# Build cả hai
npm run electron:build:mac
```

---

## 🎯 Quy trình cập nhật hoàn chỉnh

```
┌─────────────────────────────────────────────────────────┐
│  PC (Windows)                                           │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 1. Chỉnh sửa code                                   ││
│  │ 2. Tăng version trong package.json                  ││
│  │ 3. npm run electron:build:mac-arm                   ││
│  │ 4. Upload DMG lên GitHub Releases                   ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Mac M4 (User)                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 1. Mở app                                           ││
│  │ 2. App tự động check update                         ││
│  │ 3. Thông báo: "Update Available!"                   ││
│  │ 4. User click "Download"                            ││
│  │ 5. Progress bar hiện % download                     ││
│  │ 6. Click "Restart & Install"                        ││
│  │ 7. App restart với version mới! 🎉                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## ❓ FAQ

### Q: Update không hoạt động trong development?
**A:** Đúng vậy. Auto-update chỉ hoạt động khi app đã được build và sign.

### Q: Làm sao test update?
**A:**
1. Build version 1.0.0 và cài đặt
2. Tăng version lên 1.0.1, build lại
3. Upload lên server
4. Mở app 1.0.0, sẽ thấy thông báo update

### Q: Có thể force update không?
**A:** Có thể cấu hình `autoUpdater.autoDownload = true` và `autoUpdater.autoInstallOnAppQuit = true`

---

## 🎉 Done!

Hệ thống Auto-Update đã sẵn sàng. Mỗi khi bạn build version mới và upload, tất cả users sẽ nhận thông báo cập nhật!