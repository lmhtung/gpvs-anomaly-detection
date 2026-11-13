# E1

CNN 1D nhị phân dùng đặc trưng MMD/Energy nhân quả và MMD với các cửa sổ tham chiếu trễ. Mỗi đầu vào gồm 140 kênh qua 32 bước. E1 cần đủ 32 bước liên tiếp trong cùng phân đoạn trước khi dự đoán. Với cấu hình mặc định, đầu ra đầu tiên có ở khoảng 0,34 s sau warm-up. Các tham số hiệu chỉnh được tính từ chế độ nguồn.

Chuẩn bị dữ liệu, bản clone Hoque và kịch bản theo [hướng dẫn chung](../README.md#cài-đặt-và-chạy-lại). Từ thư mục gốc, chạy:

```sh
python main-detection/E1/train.py
python main-detection/E1/scenarios.py
```

`train.py` lưu checkpoint và chỉ số theo hướng, seed trong `main-detection/outputs/E1/`. `scenarios.py` dùng checkpoint đã lưu để tạo dự đoán theo từng đoạn cắt. Có thể thử một lượt huấn luyện bằng `python main-detection/E1/train.py --directions L --seeds 0`. Xem [cấu hình](config.yaml), [chi tiết mô hình](../docs.md#3-e1) và [kết quả so sánh](../README.md#kết-quả).
