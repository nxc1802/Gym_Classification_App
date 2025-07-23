# 🏋️ Gym Exercise Recognition Web App

Ứng dụng web nhận diện bài tập gym sử dụng ensemble của 4 models AI: ST-GCN, Transformer và Swin3D.

## ✨ Tính năng

- 📹 **Upload video** từ máy tính hoặc **YouTube URL**
- 🤖 **4 models AI** hoạt động song song:
  - **ST-GCN**: Graph Convolutional Network cho skeleton data
  - **Transformer 12rel**: Transformer với relative features
  - **Transformer Angle**: Transformer với angle features  
  - **Swin3D**: Video transformer
- 🎯 **Ensemble prediction** kết hợp kết quả từ tất cả models
- ⏱️ **Thời gian xử lý** của từng model
- 📊 **Top 3 predictions** với confidence scores
- 🎨 **UI đẹp và responsive**

## 🏋️ Bài tập được hỗ trợ (22 loại)

```
barbell biceps curl, lateral raise, push-up, bench press,
chest fly machine, deadlift, decline bench press, hammer curl,
hip thrust, incline bench press, lat pulldown, leg extension,
leg raises, plank, pull up, romanian deadlift, russian twist,
shoulder press, squat, t bar row, tricep pushdown, tricep dips
```

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị model weights

Copy các file weights vào thư mục `models/weights/`:

```
models/weights/
├── best_stgcn.weights.h5
├── Transformer_12rel_4_bs16_sl32.keras
├── Transformer_12rel_4_angle3_branch_bs16_sl32.keras
└── best_swin3d_b_22k.pth
```

**Lưu ý**: Đảm bảo tên file chính xác như trên.

### 5. Chạy ứng dụng
```bash
cd gym_demo_app
python app.py
```

Mở trình duyệt và truy cập: `http://localhost:5000`

## 🎬 Cách sử dụng

### Upload Video File
1. Click "Choose File" và chọn video từ máy tính
2. Click "Analyze Video"
3. Chờ xử lý và xem kết quả

### YouTube URL
1. Paste URL YouTube vào ô "YouTube URL"  
2. Click "Analyze Video"
3. Chờ download và xử lý

### Kết quả hiển thị
- **Ensemble Result**: Kết quả cuối cùng từ việc kết hợp 4 models
- **Individual Results**: Kết quả riêng lẻ từ từng model
- **Processing Times**: Thời gian xử lý của từng model
- **Top 3 Predictions**: 3 bài tập có confidence cao nhất

## 🔧 Cấu trúc dự án

```
gym_demo_app/
├── app.py                    # Flask backend chính
├── requirements.txt          # Dependencies
├── README.md                # Hướng dẫn này
│
├── templates/
│   └── index.html           # Frontend HTML
│
├── static/
│   ├── css/style.css        # Styles
│   ├── js/script.js         # JavaScript
│   └── uploads/             # Thư mục upload tạm
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py      # Load và quản lý models
│   ├── video_processor.py   # Xử lý video chính
│   └── weights/             # Model weights (cần copy)
│
└── utils/
    ├── __init__.py
    ├── mediapipe_extractor.py  # Extract pose từ video
    └── youtube_downloader.py   # Download YouTube videos
```

## 📋 Requirements

### Phần cứng tối thiểu
- **RAM**: 8GB+ (khuyến nghị 16GB)
- **GPU**: Không bắt buộc nhưng tăng tốc đáng kể
- **Disk**: 5GB trống (cho models và dependencies)

### Phần mềm
- **Python**: 3.8+
- **FFmpeg**: Cần thiết cho xử lý video (tự động với av package)

## ⚡ Performance

### Thời gian xử lý (video 30s):
- **MediaPipe extraction**: ~5-10s
- **ST-GCN**: ~2-5s  
- **Transformer models**: ~3-8s
- **Swin3D**: ~10-20s (tùy GPU)
- **Total**: ~20-40s

### Tối ưu hóa:
- Sử dụng GPU nếu có
- Video ngắn hơn xử lý nhanh hơn
- Độ phân giải thấp hơn cũng giúp tăng tốc

## 🛠️ Troubleshooting

### Lỗi model weights
```
RuntimeError: No models could be loaded
```
**Giải pháp**: Kiểm tra file weights trong `models/weights/` có đúng tên không.

### Lỗi YouTube download
```
YouTube download failed
```
**Giải pháp**: 
- Kiểm tra kết nối internet
- Thử URL khác
- Video có thể bị restricted

### Lỗi memory
```
Out of memory
```
**Giải pháp**:
- Sử dụng video ngắn hơn (<2 phút)
- Restart ứng dụng
- Tăng RAM nếu có thể

### Lỗi MediaPipe
```
Cannot open video file
```
**Giải pháp**:
- Kiểm tra format video (MP4 được khuyến nghị)
- Thử convert video sang format khác

## 🔄 API Endpoints

### `POST /upload`
Upload và xử lý video

**Request:**
- `video`: File video (multipart/form-data)
- `youtube_url`: URL YouTube (form field)

**Response:**
```json
{
  "success": true,
  "results": {
    "ensemble_predictions": [
      {"exercise": "push-up", "probability": 0.85},
      {"exercise": "plank", "probability": 0.12},
      {"exercise": "bench press", "probability": 0.03}
    ],
    "individual_predictions": {
      "stgcn": [...],
      "transformer_12rel": [...],
      "transformer_angle": [...],
      "swin3d": [...]
    },
    "processing_times": {
      "mediapipe": 8.5,
      "stgcn": 3.2,
      "transformer_12rel": 4.1,
      "transformer_angle": 5.8,
      "swin3d": 15.3
    },
    "total_processing_time": 37.2,
    "windows_processed": 12
  }
}
```

### `GET /health`
Health check endpoint

## 📝 Ghi chú kỹ thuật

### Xử lý video:
- Video được chia thành windows 32 frames
- Padding được áp dụng cho window cuối nếu thiếu frames
- MediaPipe trích xuất 33 pose landmarks cho mỗi frame

### Ensemble method:
- Simple averaging của probabilities từ tất cả models
- Có thể nâng cấp thành meta-learning trong tương lai

### Formats hỗ trợ:
- **Video**: MP4, AVI, MOV, MKV, WebM, WMV, FLV, M4V, 3GP
- **YouTube**: Tất cả URLs youtube.com và youtu.be

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Hãy tạo pull request hoặc báo lỗi qua Issues.

---

Made with ❤️ for Gym enthusiasts and AI researchers 