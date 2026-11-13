# Phát hiện lỗi nhị phân giữa IPPT và MPPT

E1 và A2 dùng các đặc trưng nhân quả và hiệu chỉnh trên chế độ nguồn. Phép thử huấn luyện trên L (IPPT) và kiểm tra trên M (MPPT), rồi đổi chiều. Baseline CNN delta của Hoque dùng ba giây đầu sau khi hệ thống bắt đầu chạy để chuẩn hóa từng bản ghi đích.

Quy ước thời gian: 0–0,5 s của mọi bản ghi là warm-up, nằm ngoài phạm vi; **hệ thống bắt đầu chạy ở giây 0,5** (S).

## Thư mục

| Thư mục | Nội dung | Quyết định đầu tiên sau S |
|---|---|---|
| [`E1/`](E1/README.md) | MMD/Energy nhân quả + kênh lag, chuỗi 32 bước phải đủ lịch sử | 0,34 s |
| [`A2/`](A2/README.md) | E1 + đệm chuỗi + 16 đặc trưng không thứ nguyên + tăng cường bằng đoạn cắt từ bản ghi nguồn | 0,04 s |
| [`Hoque/`](Hoque/README.md) | Tái lập E3b và huấn luyện baseline nhị phân Repo-Delta | 3 s với Repo-Delta |

Xem [chi tiết đặc trưng, kiến trúc và huấn luyện](docs.md). E1 và A2 tính thời điểm có đầu ra theo mép phải của cửa sổ. Repo-Delta áp ngưỡng 3 s lên tâm cửa sổ.

## Cài đặt và chạy lại

Cài thư viện và đặt dữ liệu theo [README chính](../README.md). Từ thư mục gốc của repo, clone bản tham chiếu của Khandoker Samiul Hoque và checkout commit dùng cho các bảng dưới đây. Các script đọc `results/onsets.csv` và `results/cnn_results_seeds.csv` từ bản clone.

```sh
git clone https://github.com/KhandokerSamiulHoque/gpvs-fault-detection.git main-detection/gpvs-fault-detection
git -C main-detection/gpvs-fault-detection checkout --detach fefea5a9caee329a17832448c32818b03782b09d
```

Các kết quả trong mục này dùng cột `onset_t` của Hoque để gán nhãn và tạo kịch bản. Quy trình [định vị onset bằng MMD](../mmd_onset/README.md) chạy độc lập và dùng cột `Time` cho tọa độ trong CSV gốc.

```sh
python main-detection/make_scenarios.py
python main-detection/E1/train.py
python main-detection/E1/scenarios.py
python main-detection/A2/train.py
python main-detection/A2/scenarios.py
python main-detection/Hoque/binary_delta.py
python main-detection/compare.py
```

Chạy thêm `python main-detection/Hoque/reproduce_e3b.py` để đối chiếu kết quả E3b với số liệu của repo gốc. Lệnh này chạy trên CPU. E1, A2 và Repo-Delta tự dùng CUDA khi PyTorch nhận GPU, hoặc chạy CPU. CUDA được khuyến nghị cho lượt đầy đủ, gồm hai hướng và năm seed. [Cài PyTorch với CUDA](https://pytorch.org/get-started/locally/)

| Đường dẫn dưới `main-detection/outputs/` | Nội dung |
|---|---|
| `scenarios.csv` | Danh sách đoạn cắt dùng chung |
| `E1/`, `A2/` | Checkpoint, dự đoán, lịch sử huấn luyện, chỉ số và kết quả theo kịch bản |
| `Hoque/` | Đối chiếu E3b và đầu ra nhị phân trong `binary_delta/` |
| `comparison/` | `event_summary.csv`, `window_summary.csv`, `hypotheses.csv` và kết quả chi tiết |

## Kế hoạch (ghi trước khi chạy)

**Tái lập repo.** `Hoque/reproduce_e3b.py` chạy giao thức E3b đa lớp của repo (seed 0–4) và so với
`gpvs-fault-detection/results/cnn_results_seeds.csv`. Tái lập được chấp nhận nếu |chênh lệch macro-F1-six trung bình| <= 0,05
ở cả hai hướng.

**Bản nhị phân của repo (Repo-Delta).** Cùng CNN, cùng huấn luyện (15 epoch, Adam 1e-3, batch 128, trọng số lớp, chuẩn hóa
kênh theo tập huấn luyện), đầu ra 2 lớp. Baseline = trung bình từng kênh trên cửa sổ có tâm trong [S, S+3) của chính bản ghi
(theo quy ước S = 0,5 s; repo gốc dùng [0, 3)). Chỉ quyết định cửa sổ có tâm >= S+3; trước đó tính là normal.

**Kịch bản (mode đích, `outputs/scenarios.csv`).** Cắt bản ghi sao cho:
- `original`: bản ghi nguyên vẹn (S = 0,5 s);
- `onset_after_X`, X ∈ {0,5; 1; 1,5; 2; 3} s: lỗi xuất hiện X giây sau S;
- `faulty_start`: S nằm 1 s sau onset (lỗi có sẵn khi hệ thống chạy);
- `healthy_start`: F0 với S ở giây 2, 5, 8.
Bỏ đoạn có < 1 s dữ liệu sau S. Cohort: mọi cửa sổ 20 ms có tâm >= S + 0,02 s.

**Chỉ số.**
- Mức sự kiện (chính) dùng ba cửa sổ 20 ms liên tiếp có xác suất lỗi ≥ 0,5. Độ trễ được đo từ onset, hoặc từ S nếu lỗi đã có sẵn, đến tâm cửa sổ đầu tiên trong chuỗi xác nhận. Hệ thống chỉ có đủ bằng chứng sau khi nhận cả ba cửa sổ. Báo động giả là chuỗi ba cửa sổ nằm hoàn toàn trước onset hoặc xuất hiện trong đoạn khỏe.
- Mức cửa sổ (phụ) báo cáo F1, precision và recall của lớp lỗi, cùng tỷ lệ báo động giả trên cửa sổ khỏe. Loại các cửa sổ có tâm cách onset dưới 0,5 s.
- Seed 0–4 cho mọi mô hình.

**Giả thuyết và tiêu chí.**
- H1 (lỗi sớm, X <= 2 s): A2 có độ trễ trung vị thấp hơn Repo-Delta và tỉ lệ bỏ sót không cao hơn, ở cả hai hướng.
- H2 (`faulty_start`): recall cửa sổ của Repo-Delta <= 0,5 (tham chiếu đã chứa lỗi) và A2 cao hơn Repo-Delta ở cả hai hướng.
- E1 được báo cáo cùng tiêu chí, không có giả thuyết riêng.
- Báo cáo cả báo động giả: một mô hình "phát hiện sớm" nhưng báo nhầm nhiều không được coi là thắng.

**Giới hạn biết trước.** Dữ liệu gốc không có lỗi nào trong 6 s đầu; kịch bản lỗi sớm là bản ghi cắt, không có transient
khởi động thật. Mỗi lớp lỗi một bản ghi mỗi mode. Onset là ước lượng (`gpvs-fault-detection/results/onsets.csv`).

## Kết quả

Các bảng dưới đây dùng kết quả đã lưu cho seed 0–4.

**Đối chiếu E3b trên CPU.** Giao thức có tối đa 7 lớp sau khi loại F4. Bảng dùng trung bình macro-F1-six trên các nhãn F0, F1, F2, F5, F6, F7 theo quy ước của Hoque. Cả bốn cấu hình có sai khác tuyệt đối ≤ 0,05.

| Hướng | Đầu vào | Tái lập | Repo công bố | Chênh lệch |
|---|---|---|---|---|
| IPPT→MPPT | delta | 0,7423 | 0,7398 | +0,0025 |
| MPPT→IPPT | delta | 0,4644 | 0,4895 | −0,0251 |
| IPPT→MPPT | raw | 0,2060 | 0,2155 | −0,0095 |
| MPPT→IPPT | raw | 0,2620 | 0,3020 | −0,0399 |

**Thí nghiệm nhị phân trên GPU, bản ghi gốc.** Mỗi chỉ số được tính theo từng seed trên cùng tập cửa sổ, sau đó lấy trung bình qua 5 seed. Số **in đậm** đánh dấu kết quả tốt nhất cho từng chỉ số trong mỗi hướng. Tỷ lệ báo động giả (FPR) càng thấp càng tốt.

| Hướng | Mô hình | F1 | Precision | Recall | FPR |
|---|---|---:|---:|---:|---:|
| L→M | E1 | **0,938** | 0,935 | 0,941 | 2,3% |
| L→M | A2 | 0,748 | 0,598 | **1,000** | 23,8% |
| L→M | Repo-Delta | 0,833 | **0,966** | 0,739 | **1,0%** |
| M→L | E1 | **0,924** | 0,941 | **0,911** | 2,3% |
| M→L | A2 | 0,903 | **0,973** | 0,846 | **1,1%** |
| M→L | Repo-Delta | 0,549 | 0,539 | 0,647 | 29,9% |

**Lỗi sớm (onset 0,5–2 s sau S).** Tỷ lệ bỏ sót và độ trễ được gộp qua các đoạn và seed. Độ trễ chỉ tính trên các lần phát hiện. Cột cuối là khoảng tỷ lệ báo động giả giữa bốn thời điểm onset. Cả ba chỉ số càng thấp càng tốt.

| Hướng | Mô hình | Bỏ sót | Độ trễ trung vị | Có báo động giả trước onset |
|---|---|---|---|---|
| L→M | A2 | **0%** | **0,01 s** | 33–73% số đoạn |
| L→M | E1 | 28% | 0,74 s | **0%** |
| L→M | Repo-Delta | 51% | 1,83 s | **0%** |
| M→L | A2 | **0%** | **0,22 s** | 0–13% |
| M→L | E1 | 10% | 0,28 s | 0–10% |
| M→L | Repo-Delta | 41% | 2,01 s | **0%** |

**Lỗi có sẵn khi hệ thống chạy (`faulty_start`).** Các cửa sổ được chấm đều thuộc lớp lỗi, nên precision bằng 1,000 cho cả ba mô hình. F1 và recall thể hiện khả năng phát hiện trong kịch bản này.

| Hướng | Mô hình | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| L→M | E1 | 0,490 | **1,000** | 0,325 |
| L→M | A2 | **0,937** | **1,000** | **0,882** |
| L→M | Repo-Delta | 0,378 | **1,000** | 0,234 |
| M→L | E1 | 0,390 | **1,000** | 0,246 |
| M→L | A2 | **0,610** | **1,000** | **0,444** |
| M→L | Repo-Delta | 0,318 | **1,000** | 0,196 |

Ở L→M, cần đọc độ trễ 0,01 s của A2 cùng tỷ lệ báo động giả 33–73%. Ở M→L, A2 phát hiện mọi sự kiện trong các đoạn lỗi sớm, với độ trễ trung vị 0,22 s và tỷ lệ báo động giả 0–13%. Cột `supported` trong `hypotheses.csv` kiểm tra H1 theo độ trễ và tỷ lệ bỏ sót. Tỷ lệ báo động giả được báo cáo riêng trong `event_summary.csv`.

Khoảng chờ 3 s của Repo-Delta làm giảm thời gian còn lại để phát hiện lỗi trong các đoạn ngắn. Ở M→L, Repo-Delta báo nhầm 30% cửa sổ khỏe trên bản ghi gốc và 38% trên các đoạn `healthy_start`. F1 của E1 ở L→M dao động 0,91–0,97 giữa các seed trên GPU.
