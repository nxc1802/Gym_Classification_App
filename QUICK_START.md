# 🚀 Quick Start Guide

## Cách chạy ứng dụng

### 🎯 Production Mode (Khuyến nghị)
```bash
python app.py
```
- Không restart tự động
- Ổn định khi xử lý video
- Phù hợp cho demo và sử dụng thực tế

### 🔧 Development Mode  
```bash
python run_dev.py
```
- Restart tự động khi code thay đổi
- Dùng cho việc phát triển và debug
- ⚠️ Có thể bị restart khi xử lý video

### 🧪 Test cài đặt
```bash
python test_app.py
```
- Kiểm tra imports và cấu trúc thư mục
- Không cần model weights
- Test nhanh trước khi chạy app chính

## 📱 Truy cập ứng dụng

Sau khi chạy app, mở trình duyệt:
- **Local**: http://localhost:5000
- **Network**: http://0.0.0.0:5000

## 🛑 Dừng ứng dụng

Nhấn `Ctrl+C` trong terminal để dừng server.

## ⚠️ Lưu ý quan trọng

- Đảm bảo có đủ model weights trong `models/weights/`
- Sử dụng **Production Mode** khi demo hoặc xử lý video quan trọng
- Development Mode chỉ dùng khi cần debug code 