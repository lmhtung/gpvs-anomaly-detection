# Định vị onset bằng MMD trên 13 kênh GPVS-Faults

Phương pháp tìm hồi cứu một mốc chuyển tiếp cho mỗi bản ghi để hỗ trợ gán nhãn theo thời gian. Đầu vào là CSV gốc và bản ghi khỏe cùng chế độ, F0L cho IPPT và F0M cho MPPT. Hai phía của mỗi điểm cắt phải có đủ dữ liệu. Mốc trả về là ước lượng từ tín hiệu đã quan sát và độ chính xác vật lý phụ thuộc vào khả năng nhận biết chuyển tiếp trong tín hiệu.

## Dữ liệu và giới hạn định vị

- GPVS-Faults gồm 16 bản ghi thực nghiệm, F0–F7 trong hai chế độ L/M. Lỗi được đưa vào bằng tay. Tên tệp xác định kịch bản, còn bộ CSV cung cấp thời gian đo và 13 tín hiệu điện. Mốc tiêm lỗi chính xác theo mẫu chưa được công bố. [GPVS-Faults v1](https://data.mendeley.com/datasets/n76t439f65/1) · [Bakdi và cộng sự, 2021, mục 2.2 và 4](https://doi.org/10.1016/j.ijepes.2020.106457)
- Nhiễu cảm biến, thay đổi bức xạ và nhiệt độ, cùng tác động của MPPT/IPPT có thể làm tín hiệu nền biến động. F4 là che bóng cục bộ 10–20%, còn F6/F7 là thay đổi tham số PI. Những chuyển tiếp yếu hoặc chậm có thể khó tách khỏi nền. Khi bằng chứng chưa đủ, quy trình để trống onset. F4L/F4M được giữ cho phân tích chẩn đoán và nằm ngoài tập huấn luyện dựa trên mốc xác nhận. [Bakdi và cộng sự, 2021, Bảng 1 và mục 4.2](https://doi.org/10.1016/j.ijepes.2020.106457)
- Phân tích bỏ 0,5 s đầu để loại đoạn quá độ khởi động và giữ toàn bộ phần đuôi. Bài gốc mô tả dừng sớm ở các ca nghiêm trọng. Trong CSV, F3L dài khoảng 10,35 s và F3M khoảng 7,00 s. Nhánh ngắn có thể chấm điểm gần cuối với ít dữ liệu hơn nhánh dài. F3 là kịch bản sụt áp gián đoạn, vì vậy mốc dùng để phân đoạn theo kịch bản. [Bakdi và cộng sự, 2021, mục 2.2 và Bảng 1](https://doi.org/10.1016/j.ijepes.2020.106457)

## Thuật toán

1. **Đọc CSV.** Giữ toàn bộ 13 kênh. Ước lượng bước lấy mẫu bằng trung vị các hiệu thời gian. Tách phân đoạn khi khe thời gian vượt 5 lần bước lấy mẫu và chỉ lấy cửa sổ trong từng phân đoạn. Chỉ số mẫu là chỉ số dòng CSV gốc, bắt đầu từ 0.
2. **Mô tả mỗi cửa sổ.** Dùng cửa sổ 20 ms, bước 10 ms. Các kênh mức và tần số dùng trung bình và độ lệch chuẩn. Sáu kênh pha dùng RMS và trung bình có dấu để biểu diễn cả mức biên độ lẫn độ lệch trong chu kỳ. Tổng cộng có 26 thành phần. Chuẩn hóa từng thành phần bằng median và $1{,}4826 \times \mathrm{MAD}$ trên F0 cùng chế độ. Sàn tuyệt đối và sàn tỷ lệ với biên độ điển hình của kênh F0 giới hạn khuếch đại ở kênh ít biến động.
3. **Đo thay đổi trước và sau điểm cắt.** Với mỗi độ dài ngữ cảnh $h$, dùng các cửa sổ nằm trọn một phía trong đoạn dài $h$ tính từ điểm cắt. MMD dạng V-statistic lấy trung bình kernel trong hai nhóm rồi trừ hai lần trung bình kernel chéo. Nhánh ngắn dùng âm khoảng cách Euclid trong phép tính này, thu được energy distance thực nghiệm. Hai nhánh Gaussian dùng trung bình ba RBF với băng thông bằng $0{,}5$, $1$ và $2$ lần băng thông cơ sở ước lượng từ tối đa 512 cửa sổ F0. [Gretton và cộng sự, 2012](https://jmlr.org/papers/v13/gretton12a.html) · [Sejdinovic và cộng sự, 2013](https://doi.org/10.1214/13-AOS1140)
4. **Gộp điểm và hiệu chỉnh ngưỡng.** Chia MMD của từng cặp kênh–ngữ cảnh cho cực đại tương ứng trên F0, có sàn số học ở mẫu số. Lấy trung bình nhân qua các ngữ cảnh hợp lệ rồi lấy max qua các kênh ở nhánh đơn kênh. Với nhánh joint, gộp ngữ cảnh trên điểm của một vector 26 chiều. Mỗi ngưỡng bằng $1{,}2$ nhân cực đại của 1, đỉnh trên F0 dùng để fit và hai đỉnh thu được khi lần lượt fit trên nửa này rồi chấm nửa kia của F0. `ratio` trong CSV kết quả là điểm tổng hợp chia cho ngưỡng nhánh. Tỷ số xác nhận theo từng ngữ cảnh dùng mẫu số là cực đại F0 của chính ngữ cảnh đó.
5. **Chọn một điểm cắt trên lưới.**

   | Thứ tự | Bằng chứng | Điều kiện chọn |
   |---|---|---|
   | 1 | Energy từng kênh, 40/80/160 ms mỗi phía | Chọn sự kiện mạnh xác nhận sớm nhất. Đỉnh có `ratio ≥ 6`, độ nổi ít nhất 1 và ít nhất 20% chiều cao đỉnh, với khoảng cách tối thiểu 50 ms sau lọc độ nổi. Mốc xác nhận phải có cùng một kênh đạt `ratio ≥ 6` và hai tỷ số ngữ cảnh 40/80 ms cùng đạt 1,2. |
   | 2 | Gaussian từng kênh, 0,2/0,5/1/2 s mỗi phía | Nếu chưa có sự kiện Energy, đỉnh Gaussian phải đạt `ratio ≥ 2`. Chọn mép sớm liên tục còn vượt ngưỡng nhánh và đạt ít nhất 95% điểm đỉnh, lùi tối đa 50 ms. |
   | 3 | Gaussian joint, 0,5/1/2 s mỗi phía | Khi bằng chứng đơn kênh yếu, dùng vector 26 chiều để kiểm tra thay đổi đồng thời giữa các kênh. Cần cả ba ngữ cảnh hợp lệ, tối thiểu hai tỷ số ngữ cảnh lớn hơn 1 và `ratio > 1`. Dùng cùng quy tắc mép sớm. |

   Nếu cả ba nhánh đều thiếu điều kiện, bảng onset để trống và điểm ứng viên nằm trong chẩn đoán. F0 đi qua cùng quy tắc chọn. Joint chỉ được tính cho bản ghi đích khi các nhánh đơn kênh chưa đủ mạnh.
6. **Tinh chỉnh sự kiện nhanh về mẫu gốc.** Với Energy, lấy phần dư $d[k]=x[k]-x[k-P]$, trong đó $P$ là số mẫu của một chu kỳ 50 Hz. Tính log của một cộng trung bình năng lượng phần dư đã chuẩn hóa theo F0 và chặn theo kênh. Tìm điểm chia hai đoạn có mức tăng năng lượng và giảm tổng bình phương phần dư lớn nhất. Khoảng tìm kiếm chạy từ một chu kỳ trước điểm xác nhận đến một chu kỳ sau đỉnh đã lọc độ nổi và được giới hạn trong cùng phân đoạn. Giữ mốc lưới khi khoảng tìm kiếm thiếu hỗ trợ hoặc nghiệm quá sát biên phải. Hai nhánh Gaussian giữ mốc lưới.

## Chạy và đọc đầu ra

Cài thư viện và đặt dữ liệu theo [README chính](../README.md). Các lệnh sau chạy từ thư mục gốc của repo. Bản ghi F0 tương ứng được đọc tự động khi chạy một tập con.

```sh
python -m mmd_onset.run
python -m mmd_onset.run --recordings F1L F3M --run-name subset
python -m mmd_onset.plot --recordings F1L F3M --run-name subset
```

[config.yaml](config.yaml) mặc định đọc `data/CSV_Files/` và ghi vào `outputs/mmd_onset/gpvs_13ch/`. Có thể dùng `--config` để chọn một YAML khác. Đường dẫn tương đối trong YAML tính từ thư mục gốc của repo. Khi trùng tên, lượt mới nhận hậu tố `_v2`, `_v3`… Dùng tên đã in ở cuối lượt cho lệnh vẽ lại.

Để chạy kiểm thử, cài `python -m pip install pytest` rồi chạy `python -m pytest mmd_onset/tests -q`.

| Đầu ra | Nội dung |
|---|---|
| `onsets/<file>.csv` | Tối đa một dòng. `sample_index` và `Time` định vị mốc cuối trong CSV gốc. `elapsed` là thời gian kể từ mẫu đầu tiên trước khi cắt 0,5 s. Các cột `coarse_*` ghi mốc lưới. |
| `selection_diagnostics.csv`, `scores/` | Trạng thái từng bản ghi, điểm và ngưỡng, chỉ báo đủ ngữ cảnh và bằng chứng từng nhánh. |
| `refinement/`, `figures/` | Đường giảm SSE khi tinh chỉnh và đồ thị tín hiệu, điểm số hoặc mốc đã chọn. |
| `reference_models.yaml`, `config_used.yaml`, `source_snapshot/` | Tham số hiệu chỉnh theo F0, cấu hình và mã chạy của lượt xuất. |

`run.py` ước lượng onset. `plot.py` vẽ lại từ đầu ra đã lưu. Cả hai kiểm tra tọa độ mốc theo CSV gốc. Thí nghiệm nhị phân [E1/A2](../main-detection/README.md) dùng bảng onset tham chiếu của Hoque để tái lập kết quả đã báo cáo.

## Đối chiếu với công trình trước

### Repo Hoque

Repo `gpvs-fault-detection` ước lượng một điểm cắt hồi cứu cho từng bản ghi để gán nhãn cửa sổ trong E3. README của repo ghi F4 nằm ngoài E3 vì onset không định vị được. Phép thử E2 đạt macro-F1 1,000 ngay trên các cửa sổ trước lỗi khi chia ngẫu nhiên, minh họa rủi ro mô hình học đặc trưng nhận diện bản ghi. [Repo Hoque, quy trình và giới hạn](https://github.com/KhandokerSamiulHoque/gpvs-fault-detection)

| Kịch bản | MMD IPPT (L) | Hoque IPPT (L) | MMD MPPT (M) | Hoque MPPT (M) |
|---|---:|---:|---:|---:|
| F1 | 8.658944 | 8.669143 | 8.464054 | 8.469153 |
| F2 | 8.886397 | 8.901983 | 8.709326 | 8.729124 |
| F3 | 8.915716 | 8.929115 | 6.736324 | 6.469350 |
| F4 | — | — | — | — |
| F5 | 7.146661 | 7.169259 | 7.389519 | 7.409317 |
| F6 | 11.168975 | 11.208871 | 11.568910 | 11.588808 |
| F7 | 9.309117 | 9.329015 | 11.158890 | 11.168789 |

Tất cả các mốc trong bảng tính bằng giây kể từ mẫu đầu tiên của từng CSV. MMD xác nhận 12 onset, để trống hai bản ghi F4 và hai bản ghi khỏe F0. Bảng dùng `elapsed` cho MMD và lấy `onset_t` trừ timestamp mẫu đầu tiên cho Hoque. Cả hai cột là ước lượng. [Bảng onset gốc của Hoque](https://github.com/KhandokerSamiulHoque/gpvs-fault-detection/blob/fefea5a9caee329a17832448c32818b03782b09d/results/onsets.csv)

### Bài báo trên GPVS-Faults

| Nghiên cứu | Kết quả có liên quan | Phạm vi so sánh |
|---|---|---|
| Bakdi và cộng sự (2021) | PCA–KDE–KLD báo phát hiện cả 7 loại F1–F7, với mức báo động giả dưới 1% theo Bảng 2. Hình 9–12 đánh dấu thời điểm cảnh báo của chỉ số online. | Bài tổng hợp theo loại lỗi. Cột onset cho từng CSV L/M để trống vì các hình biểu diễn cảnh báo và Bảng 2 chỉ tổng hợp theo loại. [Bài gốc, Bảng 2 và Hình 9–12](https://doi.org/10.1016/j.ijepes.2020.106457) |
| Chokr và cộng sự (2023) | Ước lượng thời điểm tiêm lỗi dựa trên kết quả của Bakdi để gán nhãn đoạn sau mốc trên MPPT. Bảng 1 liệt kê số mẫu đã gán nhãn theo lớp, tổng cộng 438.005 mẫu. | Giao nhau ở mục tiêu chuẩn bị nhãn theo ranh giới thời gian. Mốc dùng trong bài dựa vào kết quả Bakdi và bài báo công bố số lượng mẫu thay cho bảng timestamp. [Bài báo](https://doi.org/10.1016/j.solener.2023.111918) · [Bản tác giả, mục 2.2](https://univ-angers.hal.science/hal-04436119v1/document) |
| He và cộng sự (DPAL) | Tự động gán nhãn bằng sàng lọc ngưỡng kép, phân cụm và đối chiếu chỉ báo vật lý. Bài báo cáo ARI 0,7356 và NMI 0,7892 cho bước gán nhãn. | Giao nhau ở mục tiêu tạo nhãn. Các chỉ số đánh giá sự đồng thuận của nhóm nhãn, còn đầu ra MMD là mốc chuyển tiếp theo từng bản ghi. [Bài DPAL](https://doi.org/10.1016/j.eswa.2026.133804) |
