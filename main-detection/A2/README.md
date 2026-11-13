# A2

CNN 1D nhị phân mở rộng E1 bằng đệm trái, mặt nạ bước hợp lệ và 16 đặc trưng không thứ nguyên tính từ một cửa sổ. Đầu vào có 157 kênh qua 32 bước. Các đoạn cắt bổ sung lấy từ chế độ nguồn để huấn luyện trên các đoạn có ít lịch sử khỏe hoặc lỗi đã xuất hiện. Với cấu hình mặc định, đầu ra đầu tiên có ở khoảng 0,04 s sau warm-up.

Chuẩn bị dữ liệu, bản clone Hoque và kịch bản theo [hướng dẫn chung](../README.md#cài-đặt-và-chạy-lại). Từ thư mục gốc, chạy:

```sh
python main-detection/A2/train.py
python main-detection/A2/scenarios.py
```

`train.py` lưu checkpoint và chỉ số theo hướng, seed trong `main-detection/outputs/A2/`. `scenarios.py` dùng checkpoint đã lưu để tạo dự đoán theo từng đoạn cắt. Có thể thử một lượt huấn luyện bằng `python main-detection/A2/train.py --directions L --seeds 0`. Xem [cấu hình](config.yaml), [chi tiết mô hình](../docs.md#4-a2) và [kết quả so sánh](../README.md#kết-quả).
