# GPVS Anomaly Detection

Mã nguồn định vị thời điểm bắt đầu lỗi và đánh giá mô hình phát hiện lỗi nhị phân khi chuyển giữa hai chế độ IPPT và MPPT của hệ thống điện mặt trời.

## Cài đặt

```sh
git clone https://github.com/lmhtung/gpvs-anomaly-detection.git
cd gpvs-anomaly-detection
python -m venv .venv
```

Kích hoạt môi trường bằng `.\.venv\Scripts\Activate.ps1` (PowerShell) hoặc `source .venv/bin/activate` (bash), rồi chạy `python -m pip install -r requirements.txt`. Các lệnh đã được kiểm tra với Python 3.12.

Tải `CSV_Files.zip` từ [GPVS-Faults v1](https://doi.org/10.17632/n76t439f65.1) và giải nén vào `data/`. Sau khi giải nén, các tệp `F0L.csv`, `F0M.csv`, … nằm trong `data/CSV_Files/`. Bộ dữ liệu của Bakdi, Guichi, Mekhilef và Bounoua (2020) dùng [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Chạy thí nghiệm

- [Định vị onset bằng MMD](mmd_onset/README.md) · chạy `python -m mmd_onset.run` từ thư mục gốc.
- [Huấn luyện và đánh giá E1, A2](main-detection/README.md) · hướng dẫn clone repo Hoque, chạy baseline và xem kết quả.

Mã nguồn dùng [MIT License](LICENSE). Mã CNN chuyển thể từ Hoque đi kèm [thông báo bản quyền gốc](main-detection/Hoque/LICENSE).
