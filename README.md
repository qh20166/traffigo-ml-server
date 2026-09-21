# TraffiGo ML Server

Server Machine Learning cho đồ án tốt nghiệp **HK253-DATN-055 — "Xây dựng giải pháp gom cụm tình trạng giao thông ở TP.HCM"** (Trường Đại học Bách Khoa — ĐHQG TP.HCM).

Dự đoán tốc độ lưu thông, mức độ kẹt xe, rủi ro sự cố và gợi ý tuyến đường theo từng đoạn đường — hiện thực đúng bộ API Machine Learning mô tả trong báo cáo đồ án, phục vụ ứng dụng di động **TraffiGo** (Android).

## Thành phần

| Tệp | Vai trò |
|---|---|
| `app.py` | REST API FastAPI — 7 endpoint `/api/ml/*` + `/health` + bộ refresher dữ liệu live |
| `train_model.py` | Pipeline huấn luyện: đọc snapshot, làm sạch, join thời tiết, huấn luyện Random Forest, xuất model |
| `models/speed_rf.joblib` | Mô hình Random Forest đã huấn luyện (nén level 9, ~74 MB) |
| `models/cluster_model.json` | Mô hình gom cụm K-Means k=6 của đồ án (dùng chung với app Android) |
| `models/meta.json` | Thông tin đặc trưng + chỉ số đánh giá của lần train gần nhất |
| `hcmc_weather_daily.json` | Dữ liệu thời tiết ngày TP.HCM (Open-Meteo Archive) dùng khi train |

## Số liệu mô hình

| Chỉ số | Random Forest | Baseline (dự đoán = freeFlow) |
|---|---|---|
| MAE | **1,84 km/h** | 8,25 km/h |
| RMSE | 2,52 km/h | — |
| R² | 0,941 | — |

- Tập huấn luyện: 171 snapshot TomTom (09/03–15/04/2026), 149.023 mẫu / 917 đoạn đường, 100% có dữ liệu thời tiết
- Mô hình gom cụm: K-Means k=6 trên nhánh Feature Selection (Silhouette 0,4183; DBI 0,8063; Calinski–Harabasz 147.574,6)

## API

| Endpoint | Method | Chức năng |
|---|---|---|
| `/api/ml/predict` | POST | Dự đoán giao thông cho một đoạn đường |
| `/api/ml/predict-batch` | POST | Dự đoán nhiều đoạn đường |
| `/api/ml/recommend-route` | POST | Đề xuất tuyến đường tối ưu |
| `/api/ml/compare-routes` | POST | So sánh hai tuyến đường |
| `/api/ml/detect-incident` | POST | Phát hiện sự cố giao thông |
| `/api/ml/detect-incidents` | POST | Phát hiện nhiều sự cố |
| `/api/ml/traffic-status` | POST | Tổng hợp trạng thái giao thông |
| `/api/ml/live-status` | GET | Trạng thái bộ cập nhật dữ liệu live |
| `/health` | GET | Sức khỏe server + chỉ số mô hình |

Tài liệu tương tác tự sinh tại `/docs` (Swagger UI).

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/ml/predict -H "Content-Type: application/json" ^
  -d "{\"segment\": {\"currentSpeed\": 25, \"freeFlowSpeed\": 60, \"hourOfDay\": 18, \"dayOfWeek\": 6}}"
```

Kết quả: `predictedSpeed`, `trafficLevel`, `clusterId`, `severityRank`, `incident.risk` — trong đó `clusterId` được gán bằng chính mô hình K-Means đang nhúng trong ứng dụng Android.

## Bộ cập nhật dữ liệu live

Server chạy luồng nền gọi TomTom Flow API định kỳ (mặc định **mỗi 15 phút**, 50 đoạn giám sát) và lưu snapshot vào bộ nhớ, phục vụ `/api/ml/live-status`. Với 8 khóa API xoay vòng, tổng tiêu thụ ≈ 600 lời gọi/khóa/ngày — an toàn trong hạn mức free (2.500/khóa/ngày).

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TRAFFIGO_REFRESH_DISABLE` | (trống = bật) | Đặt `1` để tắt refresher |
| `TRAFFIGO_REFRESH_INTERVAL` | 900 | Chu kỳ quét live (giây) |
| `TRAFFIGO_REFRESH_MAX` | 50 | Số đoạn giám sát |
| `TOMTOM_KEYS` | đọc từ `../local.properties` | Khóa API xoay vòng, phân tách bằng dấu phẩy |

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python train_model.py     # huấn luyện lại (tùy chọn — models/ đã có sẵn)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## Tích hợp ứng dụng Android

Ứng dụng TraffiGo tự tìm server theo thứ tự: URL người dùng cấu hình (nhấn giữ chip trạng thái server trên bản đồ) → `http://10.0.2.2:8000` (emulator) → `http://127.0.0.1:8000` (khi dùng `adb reverse tcp:8000 tcp:8000` hoặc trình giả lập MuMu/LDPlayer). Server offline thì ứng dụng tự fallback về suy luận on-device — không gián đoạn tính năng.

## Deploy Render.com

1. Push repo này lên GitHub (file model 74 MB nằm dưới giới hạn 100 MB của GitHub, không cần Git LFS)
2. Render → New Web Service → chọn repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Lưu ý free tier ngủ sau 15 phút không traffic — dùng cron-job.org ping `/health` mỗi 10 phút

## Xử lý sự cố

| Hiện tượng | Xử lý |
|---|---|
| `error while loading model` khi start | Chưa có model → chạy `python train_model.py` |
| Refresher báo FAIL hết | Hết quota ngày các khóa TomTom → đợi reset hoặc thêm khóa mới (loại Mobile/Native) |
| Load model chậm ~4s lúc start | Bình thường — đang giải nén level 9 |

## Tác giả

Trần Quang Huy — 2211288, Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách Khoa — ĐHQG TP.HCM.
