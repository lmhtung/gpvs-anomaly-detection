# Tài liệu mô hình E1 và A2

Hai mô hình phát hiện lỗi nhị phân (normal / fault) trên GPVS-Faults, đánh giá cross-mode: huấn luyện trên một mode
(L = IPPT hoặc M = MPPT), kiểm tra trên mode còn lại. Cả hai không cần baseline hay hiệu chỉnh trên hệ thống đích.
Code nằm trong `E1/` và `A2/`, mỗi thư mục tự chạy độc lập; kết quả trong `outputs/E1`, `outputs/A2`.

| | E1 | A2 |
|---|---|---|
| Ý tưởng | Phát hiện sự thay đổi so với quá khứ của chính bản ghi | E1 + quyết định ngay từ đầu + kênh không cần lịch sử |
| Kênh đầu vào | 140 × 32 | 157 × 32 |
| Số tham số | 230.018 | 235.458 |
| Quyết định đầu tiên (sau khi hệ thống chạy) | 0,34 s | 0,04 s |
| Dữ liệu huấn luyện | bản ghi nguồn gốc | bản ghi nguồn gốc + các đoạn cắt 4 s |

## 1. Dữ liệu và nhãn

- 16 file CSV, 13 tín hiệu ở 10 kHz: Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc, Vf.
- Dùng F0, F1, F2, F3, F5, F6, F7 của mỗi mode. F4 bị loại vì không có onset.
- **Warm-up:** 0–0,5 s của mọi bản ghi nằm ngoài phạm vi nghiên cứu và không bao giờ được dùng. Hệ thống được coi là
  bắt đầu chạy ở giây 0,5 (gọi là S).
- **Onset** lấy từ cột `onset_t` trong `gpvs-fault-detection/results/onsets.csv`, theo ước lượng của repo Hoque.
- **Cửa sổ đánh giá:** 200 mẫu (20 ms, một chu kỳ lưới 50 Hz), bước 200 mẫu. Nhãn theo tâm cửa sổ: 0 nếu trước onset
  hoặc thuộc F0, 1 nếu từ onset trở đi.
- **Vùng bảo vệ:** cửa sổ có tâm cách onset dưới 0,5 s bị loại khỏi tập huấn luyện và khỏi chỉ số mức cửa sổ, giống repo.
- Nhãn chỉ được gắn sau khi đã tính xong đặc trưng; quá trình tính đặc trưng không đọc onset.

## 2. Tiền xử lý dùng chung (E1 và A2)

Đặc trưng tại mép phải e của cửa sổ chỉ dùng mẫu trước e. Các script hiện tại đánh giá đặc trưng nhân quả trên bản ghi đã lưu.

### 2.1 Descriptor 20 ms

Cửa sổ 200 mẫu, hop 100 mẫu (một descriptor mỗi 10 ms), bắt đầu từ 0,5 s. Mỗi descriptor có 13 × 2 giá trị:

| Nhóm | Tín hiệu | Hai giá trị |
|---|---|---|
| Biến mức | Ipv, Vpv, Vdc, Iabc, If, Vabc, Vf | mean, std |
| Pha | ia, ib, ic, va, vb, vc | RMS, mean có dấu |

### 2.2 Calibration chỉ trên F0 của mode nguồn

Dùng descriptor của F0 nguồn từ giây 3 trở đi (tránh transient khởi động). Mode đích không tham gia.

- **Chuẩn hóa từng biến:** tâm = median, thang = max(1,4826 × MAD, sàn). Sàn = max(0,001 × median|tín hiệu|, 0,01),
  riêng Vpv, Vdc, Vabc dùng sàn 0,1.
- **Băng thông kernel Gaussian từng biến:** khoảng cách trung vị giữa tối đa 512 descriptor F0, sàn 0,1.
- **Giá trị lớn nhất của từng điểm MMD trên F0:** dùng làm mẫu số ở bước 2.5.

### 2.3 MMD nhân quả giữa hai đoạn kề nhau

Tại điểm kết thúc e với ngữ cảnh h, so sánh tập descriptor trong [e − 2h, e − h) với [e − h, e), từng biến riêng:

| Loại | Kernel | Ngữ cảnh h | Số điểm |
|---|---|---|---|
| Gaussian MMD | trung bình 3 RBF (0,5×, 1×, 2× băng thông) | 0,2 / 0,5 / 1 / 2 s | 13 × 4 = 52 |
| Energy distance | −‖x − y‖ | 0,04 / 0,08 / 0,16 s | 13 × 3 = 39 |

MMD tính chính xác bằng V-statistic. Một ngữ cảnh chỉ hợp lệ khi mỗi phía có ít nhất 2 descriptor. Có 7 mask hợp lệ,
dùng chung cho cả 13 biến. Kernel chép từ `mmd_onset/detector.py`.

### 2.4 Kênh lag

Gaussian MMD giữa đoạn hiện tại [e − 1, e) và đoạn cũ [e − lag − 1, e − lag), với lag = 2 / 4 / 6 s. Kết quả là
13 × 3 = 39 điểm cộng 3 mask. Đoạn cũ không được nằm trước `start_seconds`. Kênh lag giúp vẫn thấy được lỗi đã kéo dài,
lúc mà hai đoạn kề nhau đều đã là lỗi. Giá trị lớn nhất cũng lấy từ F0 nguồn, từ giây 3.

### 2.5 Chuẩn hóa điểm

1. log1p(MMD / giá trị lớn nhất trên F0 nguồn), sàn 1e-6.
2. Trừ mean, chia std của từng kênh, tính trên tập huấn luyện nguồn và chỉ trên giá trị hợp lệ. Sàn std 0,1, cắt ±10.
3. Giá trị không hợp lệ đặt bằng 0. Mask giữ 0/1.

### 2.6 Chuỗi đầu vào

Lấy 32 descriptor liên tiếp (cách nhau 10 ms, phủ 0,33 s) kết thúc đúng ở mép phải của cửa sổ đánh giá.

## 3. E1

- **Đầu vào [140, 32]:** 91 điểm MMD/Energy, 39 điểm lag, 7 mask, 3 mask lag.
- **Điều kiện có quyết định:** phải đủ cả 32 descriptor. Vì vậy cửa sổ đầu tiên ở 0,84 s, tức 0,34 s sau khi hệ thống chạy.
- **`start_seconds` = 0,5:** đoạn tham chiếu của kênh lag được dùng dữ liệu ngay từ khi hệ thống chạy.
- **Dữ liệu huấn luyện:** chỉ các bản ghi nguồn gốc, đã bỏ vùng bảo vệ.

| Hướng | Tập huấn luyện | normal / fault | Tập kiểm tra |
|---|---|---|---|
| L→M | 4.152 chuỗi | 2.989 / 1.163 | 4.037 chuỗi |
| M→L | 4.037 chuỗi | 2.958 / 1.079 | 4.152 chuỗi |

## 4. A2

A2 = E1 cộng ba thay đổi.

### 4.1 Đệm chuỗi

Các bước chưa có descriptor (trước 0,5 s hoặc qua khoảng trống thời gian) được đặt bằng 0. Kênh "bước hợp lệ" (1 kênh)
cho mô hình biết bước nào là thật. Không cửa sổ nào bị bỏ vì thiếu lịch sử.

Cửa sổ đầu tiên là [0,52; 0,54) s, tức quyết định sau 0,04 s. Cửa sổ [0,50; 0,52) bị mất vì mẫu ở đúng 0,5 s có thời gian
0,49999… và bị mốc cắt loại (`start_seconds` = 0,52).

### 4.2 Mười sáu kênh tuyệt đối không thứ nguyên

Các kênh này tính trên từng descriptor, không cần lịch sử, nên có giá trị ngay từ quyết định đầu tiên. Hằng số ε lấy từ
F0 nguồn.

| Kênh | Công thức | Số kênh |
|---|---|---|
| Tỉ lệ gợn | log1p(std / (\|mean\| + ε)) cho 7 biến mức | 7 |
| Độ lệch DC của pha | mean / (RMS + ε) cho 6 pha | 6 |
| Mất cân bằng 3 pha | (max − min) / (trung bình + ε) của RMS ia,ib,ic và của va,vb,vc | 2 |
| Tỉ lệ công suất | log((\|mean(Iabc) · mean(Vabc)\| + e) / (\|mean(Ipv) · mean(Vpv)\| + e)) | 1 |

Cố ý không dùng mức DC tuyệt đối như mean(Vpv) hay mean(Ipv). Thí nghiệm E cho thấy chúng là lối tắt nhận diện bản ghi
và lệch giữa hai mode. Các kênh được chuẩn hóa z-score trên tập huấn luyện nguồn (chỉ bước hợp lệ), sàn std 0,1, cắt ±10.

### 4.3 Augmentation cắt bản ghi (chỉ trên mode nguồn)

Mỗi bản ghi nguồn được cắt thêm thành các đoạn [s, s + 4) s với s = 1, 2, 3, …, thời gian đặt lại về 0, onset dịch theo.
Mục đích là để lúc huấn luyện mô hình thấy được ba tình huống:
- lỗi xuất hiện khi lịch sử còn ngắn;
- đoạn bắt đầu khi lỗi đã có sẵn;
- F0 bắt đầu giữa chừng, không có transient khởi động.

Mode đích không bao giờ được cắt để huấn luyện.

- **Đầu vào [157, 32]:** 91 MMD/Energy, 39 lag, 16 tuyệt đối, 7 mask, 3 mask lag, 1 bước hợp lệ.

| Hướng | Tập huấn luyện (gốc + đoạn cắt) | Tập kiểm tra |
|---|---|---|
| L→M | 14.453 chuỗi (10.802 normal / 3.651 fault) | 4.142 chuỗi |
| M→L | 13.919 chuỗi | 4.257 chuỗi |

## 5. Kiến trúc (giống nhau, chỉ khác số kênh vào)

CNN 1D có residual, chuẩn hóa GroupNorm(8). Đầu vào [batch, C, 32], đầu ra 2 logit (normal, fault).

| Lớp | Cấu hình | Tham số E1 (C = 140) | Tham số A2 (C = 157) |
|---|---|---|---|
| Stem | Conv1d C→64, kernel 5, không bias + GroupNorm + ReLU | 44.928 | 50.368 |
| Residual 1 | 64→64, stride 1 | 24.832 | 24.832 |
| Residual 2 | 64→96, stride 2 (có nhánh tắt 1×1) | 52.800 | 52.800 |
| Residual 3 | 96→128, stride 2 (có nhánh tắt 1×1) | 99.072 | 99.072 |
| Pooling | AdaptiveAvgPool1d(1) → 128 chiều | 0 | 0 |
| Head | Linear 128→64 + ReLU + Dropout 0,1 + Linear 64→2 | 8.386 | 8.386 |
| **Tổng** | | **230.018** | **235.458** |

Mỗi residual block gồm Conv1d kernel 3 + GroupNorm + ReLU, rồi Conv1d kernel 3 + GroupNorm. Kết quả được cộng với nhánh
tắt (identity, hoặc Conv 1×1 + GroupNorm khi đổi kích thước) rồi qua ReLU.

## 6. Huấn luyện

| Mục | Giá trị |
|---|---|
| Loss | Cross-entropy, trọng số lớp = N / (2 × số mẫu lớp) trên nhãn nguồn |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4 |
| Lịch lr | Cosine 60 epoch, giảm tới 5% lr |
| Batch | 128 |
| Cắt gradient | norm 1,0 |
| Regularization | Feature dropout 0,1: bỏ cùng lúc 7 kênh MMD/Energy của một biến, giống nhau trên cả 32 bước; không áp dụng cho kênh lag, tuyệt đối và mask |
| Chọn checkpoint | Epoch cuối cố định, không validation, không chọn theo mode đích |
| Seed | 0–4 |
| Phần cứng | GPU NVIDIA L40S, môi trường `anomaly-gpu` (torch 2.10.0+cu128) |
| Thời gian | Khoảng 25 s mỗi seed với E1, khoảng 90 s với A2; trích xuất đặc trưng 1–2 phút mỗi hướng |

Mode đích không tham gia bất kỳ bước fit nào: bank, scaler, ngưỡng hay chọn checkpoint.

## 7. Suy luận

- softmax rồi ngưỡng 0,5 trên p(fault), không làm mượt.
- Đặc trưng được tính bằng bank và scaler đã đóng băng từ mode nguồn, không đọc onset.
- Code hiện xử lý theo lô trên file CSV. Vì mọi tính toán đều nhân quả nên chuyển sang chạy trên luồng dữ liệu được mà
  không đổi kết quả, nhưng phần đó chưa được viết.

**Thời gian chờ sau khi hệ thống chạy:**

| | E1 | A2 |
|---|---|---|
| Quyết định đầu tiên | 0,34 s | 0,04 s |
| Kênh tuyệt đối đầy đủ | – | 0,04 s |
| Kênh MMD 2 s đầy đủ | 4 s | 4 s |
| Kênh lag 2 / 4 / 6 s có giá trị | 3 / 5 / 7 s | 3 / 5 / 7 s |

Trước khi đủ lịch sử, kênh tương ứng bị mask = 0; mô hình vẫn quyết định bằng các kênh đã có.

## 8. Kết quả tóm tắt (5 seed, GPU)

**Bản ghi đích gốc (mức cửa sổ):**

| Hướng | Mô hình | F1 (fault) | Recall | Báo động giả | AUROC |
|---|---|---|---|---|---|
| L→M | E1 | 0,938 ± 0,025 | 0,941 | 2,4% | 0,987 |
| L→M | A2 | 0,748 ± 0,024 | 1,000 | 23,8% | 0,999 |
| M→L | E1 | 0,924 ± 0,036 | 0,911 | 2,4% | 0,985 |
| M→L | A2 | 0,903 ± 0,018 | 0,846 | 1,1% | 0,976 |

**So với Repo-Delta khi lỗi xuất hiện 0,5–2 s sau khi hệ thống chạy (mức sự kiện):**

| Hướng | Mô hình | Bỏ sót | Độ trễ trung vị | Đoạn có báo động giả trước onset |
|---|---|---|---|---|
| L→M | E1 | 28% | 0,74 s | 0% |
| L→M | A2 | 0% | 0,01 s | 33–73% |
| L→M | Repo-Delta | 51% | 1,83 s | 0% |
| M→L | E1 | 10% | 0,28 s | 0–10% |
| M→L | A2 | 0% | 0,22 s | 0–13% |
| M→L | Repo-Delta | 41% | 2,01 s | 0% |

**Đánh giá:**
- E1 ổn định và ít báo nhầm, nhưng chậm hơn và yếu khi lỗi có sẵn từ đầu (recall 0,33 / 0,25).
- A2 phát hiện sớm và tốt nhất khi lỗi có sẵn (recall 0,88 / 0,44). Tuy nhiên ở L→M nó báo nhầm nhiều, chủ yếu ở vùng
  trước onset của F6M và F7M, dù AUROC là 0,999. Tức mô hình xếp hạng đúng nhưng ngưỡng 0,5 bị lệch khi chuyển mode.

Xem [bảng so sánh và hướng dẫn chạy lại](README.md).

## 9. Giới hạn

- Mỗi lớp lỗi chỉ có một bản ghi mỗi mode, nên danh tính bản ghi gần như trùng với danh tính lớp.
- Onset là ước lượng; một phần "báo động giả" trước onset có thể là lỗi nhãn.
- Dữ liệu gốc không có lỗi nào trong 6 s đầu. Các kịch bản lỗi sớm là bản ghi cắt, không có transient khởi động thật.
- Chỉ có 5 seed. E1 nhạy với seed: F1 L→M dao động 0,91–0,97 trên GPU. Với cùng code, kết quả trên CPU và GPU khác nhau
  do thứ tự trộn batch.

## 10. Chạy lại

Xem [các lệnh chạy từ thư mục gốc](README.md#cài-đặt-và-chạy-lại).

| File | Vai trò |
|---|---|
| `features.py` | descriptor, kernel, MMD, lag, calibration F0, chuỗi, scaler (A2: thêm đệm, kênh tuyệt đối, cắt bản ghi) |
| `model.py` | CNN 1D, feature dropout, hàm dự đoán |
| `train.py` | chuẩn bị dữ liệu theo hướng, huấn luyện từng seed, lưu checkpoint, dự đoán, metric |
| `scenarios.py` | chấm các đoạn trong `main-detection/outputs/scenarios.csv` cho từng seed đã huấn luyện |
| `config.yaml` | toàn bộ tham số |
