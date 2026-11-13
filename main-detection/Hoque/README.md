# Baseline CNN của Hoque

Hai script dùng CNN 1D từ [mã E3b của Khandoker Samiul Hoque](https://github.com/KhandokerSamiulHoque/gpvs-fault-detection/blob/fefea5a9caee329a17832448c32818b03782b09d/code/11_cnn.py). Bản chuyển thể trong [`repo_cnn.py`](repo_cnn.py) trả về mô hình và tham số chuẩn hóa để có thể xuất xác suất. Xem [giấy phép MIT và thông báo bản quyền gốc](LICENSE).

- `reproduce_e3b.py` tái lập phân loại đa lớp trên CPU, với đầu vào raw hoặc delta, và so với `results/cnn_results_seeds.csv` của Hoque. Bản delta lấy tham chiếu trên ba giây đầu của mỗi bản ghi.
- `binary_delta.py` huấn luyện Repo-Delta với hai lớp normal/fault trên cùng kịch bản với E1 và A2. Cửa sổ có tâm từ ba giây sau S trở đi mới được chấm bởi mô hình. Bộ so sánh tính các cửa sổ chưa có quyết định là normal.

Thực hiện [bước cài đặt, clone và tạo kịch bản](../README.md#cài-đặt-và-chạy-lại), rồi chạy từ thư mục gốc:

```sh
python main-detection/Hoque/reproduce_e3b.py
python main-detection/Hoque/binary_delta.py
```

Kết quả E3b ghi vào `main-detection/outputs/Hoque/reproduction_summary.csv`. Kết quả nhị phân nằm trong `main-detection/outputs/Hoque/binary_delta/`. [Xem các bảng đối chiếu](../README.md#kết-quả).
