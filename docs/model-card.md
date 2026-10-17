# Model Card — Phân loại tự động mô tả sự cố

Thành phần 6 của Bài tập 3 cho track AI. Model card viết cho **người sẽ DÙNG**
mô hình, không phải cho người xây nó. Mục tiêu: người đọc quyết định được mô hình
này có phù hợp với việc của họ hay không, mà không cần đọc mã nguồn.

---

## 1. Thông tin cơ bản

| | |
|---|---|
| **Tên mô hình** | `phan-loai-su-co` |
| **Phiên bản** | v1.0 |
| **Loại mô hình đã chọn** | `ComplementNB` (Naive Bayes bổ sung) trên đặc trưng TF-IDF 1–2 gram |
| **Ngày tạo** | 09/2026 |
| **Người tạo** | Repo mẫu của học phần Chuyên đề tốt nghiệp 1, Khoa CNTT, Trường Đại học Văn Lang |
| **Cách liên hệ** | Qua giảng viên phụ trách học phần |
| **Sinh lại mô hình** | `python -m src.models.train --config configs/train.yaml` |

---

## 2. Mục đích sử dụng dự kiến

**GỢI Ý** nhóm lỗi cho nhân viên tiếp nhận tại trung tâm bảo hành, ngay khi họ
nhập mô tả sự cố, nhằm giảm thời gian nhập liệu và giảm việc chọn sai nhóm lỗi.

- **Ai dùng:** nhân viên tiếp nhận tại quầy.
- **Dùng ở đâu trong quy trình:** bước nhập mô tả sự cố khi tiếp nhận thiết bị
  (luồng nghiệp vụ L2 và L10 của case study).
- **Vai trò của mô hình:** đưa ra **ba** nhãn gần nhất kèm độ tin cậy.
  **Nhân viên là người quyết định cuối cùng** — hệ thống không tự gán nhãn.

---

## 3. Trường hợp KHÔNG nên dùng

| Không dùng để | Vì sao |
|---|---|
| **Tự động gán nhãn không có người xem** | Mô hình sai khoảng 17% số ca (F1 macro 0,832). Gán tự động sẽ đưa dữ liệu sai vào hệ thống mà không ai phát hiện. |
| **Tự động phân công kỹ thuật viên** | Phân công dựa trên nhóm lỗi sai sẽ gửi thiết bị tới người không có kỹ năng phù hợp, làm chậm cả quy trình. |
| **Quyết định thiết bị có được bảo hành hay không** | Đây là quyết định có hệ quả tài chính với khách hàng, dựa trên hợp đồng và ngày bán — không liên quan tới nhóm lỗi. |
| **Đánh giá năng lực hay hiệu suất nhân viên** | Mô hình được huấn luyện trên nhãn **do nhân viên gán**, nên nó học lại chính thói quen của họ. Dùng nó để đánh giá họ là lập luận vòng tròn. |
| **Ngôn ngữ khác tiếng Việt** | Hàm làm sạch và từ vựng TF-IDF đều dựa trên tiếng Việt. Mô tả bằng ngôn ngữ khác sẽ cho kết quả vô nghĩa nhưng vẫn có độ tin cậy — nguy hiểm hơn là báo lỗi. |
| **Mô tả dài hơn 2.000 ký tự** | Dịch vụ từ chối với mã 422. Mô tả rất dài chưa được kiểm thử và có thể làm dịch vụ chậm. |
| **Loại thiết bị ngoài điện thoại và phụ kiện** | Dữ liệu huấn luyện chỉ có sự cố của điện thoại, tai nghe, sạc, máy tính bảng, đồng hồ, loa, pin dự phòng. |

---

## 4. Dữ liệu huấn luyện

| | |
|---|---|
| **Nguồn** | `data/mau/issue_descriptions.csv` |
| **Bản chất** | **Dữ liệu SINH RA**, không phải dữ liệu thật. Sinh bởi `data/mau/gen_mau.py`, hạt giống `20260901`. |
| **Số mẫu** | 2.400 |
| **Chia** | 70 / 15 / 15 **phân tầng theo nhãn**, hạt giống `42` → 1.680 / 360 / 360 |
| **Số dòng bị bỏ** | 0 (không dòng nào thiếu nhãn hoặc thiếu văn bản) |

**Phân bố nhãn — mất cân bằng có ý:**

| Nhãn | Số mẫu | Tỉ lệ |
|---|---:|---:|
| MAN_HINH | 518 | 21,6% |
| PIN | 434 | 18,1% |
| KHAC | 428 | 17,8% |
| PHAN_MEM | 380 | 15,8% |
| SAC | 375 | 15,6% |
| NUOC_VAO | 265 | 11,0% |

**Các thiên lệch đã biết TRONG dữ liệu — cố ý đưa vào để mô phỏng dữ liệu thật:**

1. **Câu mô tả nhập nhằng** dùng chung giữa các cặp nhãn gần nghĩa. Ví dụ
   `máy không lên nguồn` và `cắm sạc không lên gì` dùng cho **cả PIN và SAC**.
   Điều này tạo ra một **trần trên** cho độ chính xác đạt được: không mô hình nào
   phân biệt được những câu này, vì bản thân câu không chứa đủ thông tin.
2. **Nhiễu nhãn ~7%** — một tỉ lệ dòng bị gán sai nhãn, mô phỏng việc người nhập
   chọn nhanh cho xong. Nhiễu nhãn đặt thêm một trần nữa lên chỉ số đạt được.
3. **Nhãn `KHAC` là thùng rác**, nhận nhiều chủ đề không liên quan nhau.
4. **~30% mô tả dưới 10 từ**, cố ý ít thông tin hơn.
5. **~12% viết không dấu, ~4% viết CHỮ HOA.**

> **CẢNH BÁO QUAN TRỌNG:** vì dữ liệu là dữ liệu sinh ra, con số F1 0,832 **chỉ
> có ý nghĩa tương đối** — dùng để so sánh các mô hình với nhau trên cùng bộ này.
> Nó **không** dự báo được năng lực của mô hình trên mô tả sự cố thật của khách
> hàng. Trước khi triển khai thật, **bắt buộc** huấn luyện lại và đo lại trên dữ
> liệu thật đã được gán nhãn.

---

## 5. Kiến trúc và cách xử lý

**Tiền xử lý** (`src/features/text.py`):

1. Hạ chữ thường.
2. **GIỮ DẤU tiếng Việt.** Đánh đổi có ý: bỏ dấu sẽ gộp `sạc` và `sách` thành
   `sach`, làm mất thông tin của ~84% dữ liệu có dấu để đổi lấy việc gộp ~12%
   dữ liệu không dấu. Kết luận: giữ dấu, và chấp nhận rằng mô tả không dấu được
   xử lý kém hơn.
3. Thay thông tin định danh bằng thẻ: số điện thoại → `<sdt>`, số serial →
   `<serial>`, địa chỉ thư điện tử → `<email>`. Ba loại này không giúp phân loại
   nhóm lỗi, và giữ lại chúng vừa gây quá khớp vừa có nguy cơ lọt vào nhật ký.
4. Bỏ ký tự đặc biệt (giữ chữ Việt và hai dấu `<`, `>` của thẻ), gộp khoảng trắng.

**Vector hoá:** TF-IDF, `ngram_range=(1,2)`, `min_df=2`, `max_features=20000`.

**Mô hình:** `ComplementNB`. Được chọn **trên tập validation** trong ba ứng viên.

**Đã kiểm rằng phép kiểm rò rỉ biết phát hiện lỗi:** hai lần cố ý chèn rò rỉ
(một vào chính test, một vào `train.py`) cho thấy bộ test ban đầu vẫn xanh — xem
BUG-05 và BUG-06 trong `docs/bug-log.md`. Nay `RR-03` kiểm hai chiều và `RR-06`
kiểm trên **chính tệp `artifacts/model_v1.joblib` sẽ được triển khai**: từ vựng
trong đó trùng khít 1.308 từ với bản fit lại chỉ trên tập train, và khác 1.391
từ của bản fit trên toàn bộ dữ liệu.

**Bảo đảm bằng cấu trúc rằng không rò rỉ dữ liệu:** vectorizer nằm **trong**
`sklearn.pipeline.Pipeline`, nên nó chỉ được `fit` trên tập truyền vào `.fit()`
và không thể vô tình nhìn thấy tập validation hay tập test. Có test
`tests/test_no_leakage.py` kiểm điều này.

**Hạn chế về xác suất:** `ComplementNB` cho xác suất **chưa được hiệu chỉnh**
(uncalibrated), và `LinearSVC` thậm chí không có `predict_proba` nên dịch vụ dùng
softmax trên điểm quyết định. Nghĩa là **`do_tin_cay` là một thang đo tin cậy
tương đối, không phải xác suất thật.** Không được đọc `do_tin_cay = 0.7` là "đúng
70% số lần". Cách xử lý đúng nếu cần xác suất thật: dùng `CalibratedClassifierCV`
— chưa làm ở phạm vi học phần này.

---

## 6. Chỉ số đánh giá

**Chỉ số chính: F1 macro.** Lý do chọn — có số liệu cụ thể: nhãn mất cân bằng từ
11,0% tới 21,6%, nên mô hình luôn đoán nhãn đa số đã đạt **21,6% độ chính xác**
mà không học được gì. F1 macro tính trung bình **đều** cho cả sáu nhãn, nên nếu
mô hình bỏ rơi nhãn `NUOC_VAO` (ít mẫu nhất) thì chỉ số tụt rõ.

### Điều kiện đo — bắt buộc ghi kèm

| | |
|---|---|
| Hạt giống | 42 |
| Nguồn dữ liệu | `data/mau/issue_descriptions.csv` (2.400 dòng, sinh với hạt giống 20260901) |
| Số mẫu train / val / test | 1.680 / 360 / 360 |
| Tập đo | **Tập test**, đo **đúng một lần** (có bộ đếm ném lỗi nếu đo lần hai) |

### Kết quả

| Mô hình | F1 macro (validation) | F1 macro (test) |
|---|---:|---:|
| Cơ sở — đoán nhãn đa số | 0,0587 | — |
| Cơ sở — TF-IDF + hồi quy logistic | 0,8008 | — |
| Ứng viên — `logistic` | 0,8008 | — |
| Ứng viên — `linear_svc` | 0,8055 | — |
| **Đã chọn — `complement_nb`** | **0,8236** | **0,8318** |

### F1 và recall theo TỪNG NHÃN — chỉ số tổng thể che giấu điều này

| Nhãn | F1 | Recall | Nhận xét |
|---|---:|---:|---|
| MAN_HINH | 0,9006 | 0,9872 | Tốt nhất — từ vựng đặc trưng rõ |
| PIN | 0,8525 | 0,8000 | Bị nhầm sang SAC |
| SAC | 0,8361 | 0,9107 | Bị nhầm sang PIN |
| NUOC_VAO | 0,8276 | 0,9000 | Tốt dù ít mẫu nhất |
| KHAC | 0,8067 | 0,7500 | Nhãn thùng rác |
| **PHAN_MEM** | **0,7677** | **0,6667** | **Kém nhất — xem mục 9** |

### Confusion matrix (tập test, 360 mẫu)

Hàng = nhãn thật, cột = nhãn dự đoán.

| thật ↓ / dự đoán → | KHAC | MAN_HINH | NUOC_VAO | PHAN_MEM | PIN | SAC |
|---|---:|---:|---:|---:|---:|---:|
| **KHAC** | **48** | 0 | 11 | 4 | 1 | 0 |
| **MAN_HINH** | 0 | **77** | 0 | 0 | 0 | 1 |
| **NUOC_VAO** | 4 | 0 | **36** | 0 | 0 | 0 |
| **PHAN_MEM** | 3 | 15 | 0 | **38** | 0 | 1 |
| **PIN** | 0 | 0 | 0 | 0 | **52** | 13 |
| **SAC** | 0 | 1 | 0 | 0 | 4 | **51** |

---

## 7. Mô hình cơ sở để so sánh

| Mô hình cơ sở | F1 macro | Ý nghĩa |
|---|---:|---|
| Đoán nhãn đa số (`MAN_HINH`) | **0,0587** | **Mức sàn.** Mô hình phải cao hơn con số này, nếu không thì nó không học được gì. |
| TF-IDF + hồi quy logistic | **0,8008** | **Mức chuẩn.** Với vài nghìn dòng văn bản, mô hình này rất khó vượt qua. |

Mô hình đã chọn (0,8318) cao hơn mức sàn **14 lần** và cao hơn mức chuẩn
**0,031**. Cải thiện so với mức chuẩn là **nhỏ** — cần nói thẳng điều này: nếu
tiêu chí lựa chọn là khả năng giải thích được, thì hồi quy logistic là lựa chọn
hợp lý hơn dù kém 0,031, vì trọng số của nó đọc được trực tiếp.

---

## 8. Thiên lệch đã đo

Đo trên tập test, **hai** cách nhóm, ngưỡng cảnh báo 0,15.

### Cách nhóm 1 — theo độ dài mô tả

| Nhóm | Số mẫu | F1 macro |
|---|---:|---:|
| Ngắn (< 10 từ) | 106 | 0,8187 |
| Dài (≥ 10 từ) | 254 | 0,8329 |
| **Chênh lệch** | | **0,0142** |

**Kết luận:** dưới ngưỡng 0,15. Chênh lệch nhỏ và theo chiều dự đoán được (mô tả
ngắn kém hơn). Vẫn áp dụng biện pháp giảm thiểu ở mục 9 vì chiều của chênh lệch
là chiều gây hại.

### Cách nhóm 2 — theo trung tâm bảo hành (`center_id`)

| Trung tâm | Số mẫu | F1 macro |
|---|---:|---:|
| 1 | 72 | 0,7658 |
| 2 | 51 | 0,9260 |
| 3 | 60 | 0,8479 |
| 4 | 57 | 0,8864 |
| 5 | 53 | 0,7847 |
| 6 | 67 | 0,8052 |
| **Chênh lệch (2 so với 1)** | | **0,1602** |

**Chênh lệch 0,1602 VƯỢT ngưỡng cảnh báo 0,15.** Nhưng đây là chỗ phải trung thực:

> **Chênh lệch này rất có thể là NHIỄU, không phải thiên lệch thật.** Trong
> `data/mau/gen_mau.py`, `center_id` được gán **hoàn toàn ngẫu nhiên** và **không
> có quan hệ nào** với nhãn hay với nội dung mô tả — theo cách xây dựng dữ liệu.
> Chênh lệch xuất hiện vì mỗi nhóm chỉ có **51–72 mẫu**, và F1 macro trên nhóm
> nhỏ như vậy dao động rất mạnh.
>
> **Không tuyên bố rằng mô hình có thiên lệch theo trung tâm.** Điều đúng đắn là
> nêu rõ: phép đo này **chưa đủ mẫu để kết luận**. Để kết luận được cần tối thiểu
> khoảng 200 mẫu mỗi nhóm, hoặc phải dùng khoảng tin cậy bằng bootstrap thay vì
> một con số điểm.

**Bài học chung:** một chênh lệch đo trên nhóm nhỏ có thể là nhiễu. Báo cáo một
con số như bằng chứng thiên lệch khi nó chỉ là dao động lấy mẫu là **cùng một
loại sai lệch số liệu** với việc báo cáo độ chính xác 89% trên dữ liệu mất cân
bằng — chỉ là theo chiều ngược lại.

---

## 9. Rủi ro và giới hạn

### Rủi ro 1 — Nhãn PHAN_MEM có recall thấp nhất (0,6667)

**Vấn đề:** một phần ba ca thật thuộc `PHAN_MEM` bị bỏ sót; **15 trong số đó bị
nhầm thành `MAN_HINH`**. Nguyên nhân đọc được từ dữ liệu: các mô tả nhập nhằng
như `màn hình đứng không bấm được` và `máy treo không tắt được` dùng chung cho hai
nhãn — máy treo do phần mềm nhưng biểu hiện trên màn hình.

**Hệ quả nghiệp vụ:** thiết bị lỗi phần mềm bị gửi tới kỹ thuật viên phần cứng,
mất thời gian chuyển tiếp và làm chậm thời gian xử lý.

**Biện pháp giảm thiểu — đã hiện thực:** dịch vụ trả về **top 3**, nên `PHAN_MEM`
gần như luôn nằm trong ba lựa chọn hiển thị cho nhân viên dù không phải lựa chọn
đầu. Nhân viên thấy cả ba và tự quyết.

**Biện pháp giảm thiểu — đề xuất, chưa làm:** thêm một câu hỏi phụ khi tiếp nhận
("máy có hiển thị hình bình thường không?") để tách hai nhãn này ngay ở nguồn.
Đây là thay đổi **quy trình nghiệp vụ**, cần thoả thuận với trưởng bộ phận tiếp nhận.

### Rủi ro 2 — Mô tả ngắn cho kết quả kém hơn

**Vấn đề:** mô tả dưới 10 từ có F1 macro 0,8187 so với 0,8329 của mô tả dài
(chênh 0,0142). Chênh lệch nhỏ nhưng theo chiều gây hại, và ~30% mô tả thuộc
nhóm ngắn.

**Biện pháp giảm thiểu — ĐÃ HIỆN THỰC:** dịch vụ áp **ngưỡng tin cậy riêng, cao
hơn** cho mô tả ngắn: **0,60** thay vì 0,45. Dưới ngưỡng thì không đưa ra nhãn
gợi ý và gắn cờ `can_nguoi_xem_lai`. Phản hồi trả về cả `nguong_da_ap_dung` và
`so_tu_mo_ta` để phía gọi kiểm chứng được. Có test `SV-04` chứng minh biện pháp
này tồn tại trong sản phẩm — xem `BUG-03` về việc biện pháp này ban đầu **chỉ có
trong tài liệu mà chưa có trong mã**.

### Rủi ro 3 — Nhãn KHAC là thùng rác, hút về dự đoán sai

**Vấn đề:** `KHAC` có F1 0,8067 và recall 0,75. Nó nhận các mô tả thuộc nhiều chủ
đề không liên quan (không nhận sim, loa rè, nút nguồn kẹt, wifi tự ngắt).

**Hệ quả nghiệp vụ:** phiếu gán nhãn `KHAC` không giúp phân công kỹ thuật viên,
nên `KHAC` càng nhiều thì mô hình càng ít giá trị.

**Biện pháp giảm thiểu — đề xuất, chưa làm:** xem lại **định nghĩa nhãn** cùng
người dùng nghiệp vụ và tách `KHAC` thành các nhóm nhỏ hơn (ví dụ `SIM_MANG`,
`LOA_MIC`, `NUT_PHIM`). Đây là thay đổi **danh mục nghiệp vụ**, không phải thay
đổi mô hình — mô hình không thể tự sửa một định nghĩa nhãn kém.

### Rủi ro 4 — Độ tin cậy chưa được hiệu chỉnh

**Vấn đề:** `do_tin_cay` không phải xác suất thật (xem mục 5). Người dùng có thể
đọc `0.7` là "đúng 70% số lần" — sai.

**Biện pháp giảm thiểu — đã hiện thực:** tài liệu và tên trường nói rõ đây là
**độ tin cậy**, không phải xác suất; dịch vụ trả về cả `nguong_da_ap_dung` để
người dùng thấy con số này chỉ được dùng để so với một ngưỡng, không phải để đọc
như một tỉ lệ.

### Rủi ro 5 — Dữ liệu huấn luyện là dữ liệu SINH RA

**Vấn đề:** mọi con số trong Model card này đo trên dữ liệu sinh ra, không phải
dữ liệu thật của khách hàng.

**Biện pháp giảm thiểu — đã hiện thực:** ghi cảnh báo rõ ràng ở mục 4 và ở
`docs/test-plan.md`. **Điều kiện bắt buộc trước khi triển khai thật:** huấn luyện
lại và đo lại trên dữ liệu thật đã gán nhãn, rồi cập nhật toàn bộ mục 6, 8 và 9
của tài liệu này.

---

## 10. Bảo trì

**Khi nào cần huấn luyện lại:**

| Dấu hiệu | Ngưỡng | Hành động |
|---|---|---|
| Tỉ lệ nhân viên **sửa** nhãn gợi ý | > 35% trong một tháng | Huấn luyện lại với dữ liệu mới |
| Tỉ lệ ca bị gắn cờ `can_nguoi_xem_lai` | > 50% trong một tuần | Kiểm phân bố độ tin cậy; có thể ngôn ngữ mô tả đã thay đổi |
| Thêm nhóm lỗi mới vào danh mục | Bất kỳ | **Bắt buộc** huấn luyện lại — mô hình không biết nhãn chưa từng thấy |
| Thêm loại thiết bị mới | Bất kỳ | Kiểm chỉ số tách theo loại thiết bị trước khi tin vào mô hình |
| Đổi phiên bản `scikit-learn` | Bất kỳ | **Bắt buộc** sinh lại tệp mô hình — tệp lưu bởi phiên bản khác có thể nạp lỗi |

**Cách theo dõi:** ghi nhật ký cả nhãn mô hình gợi ý và **lựa chọn cuối của nhân
viên**. Hiệu số giữa hai giá trị này là chỉ số theo dõi quan trọng nhất, và nó
đo được **mà không cần ai gán nhãn thêm**.

**Ai chịu trách nhiệm:** người bảo trì hệ thống tiếp nhận. Mọi lần huấn luyện lại
phải cập nhật `reports/metrics.json` và **cập nhật mục 6, 8, 9 của tài liệu này**
— một Model card mô tả phiên bản mô hình không còn chạy là tài liệu sai.

**Hiệu năng đã đo:** một lần dự đoán mất khoảng **2,5 ms** (đo bằng tay trên máy
phát triển, không phải phép đo dưới tải). Mô hình được nạp **một lần** khi khởi
động dịch vụ, không nạp lại mỗi yêu cầu.

---

## Phụ lục — Bảy câu hỏi AI có trách nhiệm

| # | Câu hỏi | Trả lời |
|---|---|---|
| 1 | Người bị ảnh hưởng bởi một dự đoán SAI là ai, và họ mất gì? | **Khách hàng** mang thiết bị đi bảo hành: nhãn sai → phân công kỹ thuật viên sai → thời gian xử lý dài hơn, có thể phải mang máy đi lại. **Nhân viên tiếp nhận**: gợi ý sai làm họ mất thời gian sửa, và nếu sai nhiều thì họ mất niềm tin và bỏ qua gợi ý — khi đó tính năng vô dụng. |
| 2 | Người đó có biết có hệ thống AI tham gia không? | **Nhân viên**: có, giao diện hiển thị rõ đây là gợi ý tự động kèm độ tin cậy. **Khách hàng**: không, và ở phạm vi hiện tại là hợp lý vì mô hình chỉ gợi ý cho nhân viên, không ra quyết định nào về khách hàng. **Nếu sau này** kết quả được dùng để ra quyết định ảnh hưởng khách hàng (ví dụ từ chối bảo hành) thì **bắt buộc** phải cho khách biết. |
| 3 | Người đó có cách nào phản đối hoặc xem lại kết quả không? | **Có, ở mức thiết kế:** nhân viên luôn là người quyết định cuối; mô hình chỉ gợi ý top 3. Không có quyết định tự động nào để phản đối. Đây là biện pháp bảo vệ mạnh nhất và là lý do mục 3 cấm dùng mô hình để gán nhãn tự động. |
| 4 | Mô hình có hoạt động kém hơn với nhóm nào không? Đã ĐO chưa? | **Đã đo, hai cách nhóm** — xem mục 8. Theo độ dài mô tả: chênh 0,0142, dưới ngưỡng, đã có biện pháp giảm thiểu. Theo trung tâm: chênh 0,1602 vượt ngưỡng **nhưng rất có thể là nhiễu** vì mỗi nhóm chỉ 51–72 mẫu — đã nêu rõ giới hạn của phép đo thay vì tuyên bố có thiên lệch. **Chưa đo** theo loại thiết bị và theo nhân viên tiếp nhận, vì bộ dữ liệu mẫu không có hai cột đó. |
| 5 | Có đặc trưng nào việc dùng nó là không phù hợp về đạo đức? | **Không.** Đặc trưng duy nhất là **nội dung mô tả sự cố** — thông tin kỹ thuật về thiết bị, không phải về con người. `center_id` chỉ dùng để **đo thiên lệch**, không phải đặc trưng đầu vào của mô hình. Số điện thoại, serial và địa chỉ thư điện tử bị **thay bằng thẻ** trước khi vector hoá, nên mô hình không thể học theo danh tính khách hàng. |
| 6 | Dữ liệu huấn luyện có được thu thập hợp lệ không? | Ở repo mẫu: **dữ liệu sinh ra**, không có dữ liệu của ai. Ở hệ thống thật: mô tả sự cố do nhân viên nhập trong quá trình tiếp nhận, thuộc dữ liệu vận hành nội bộ. **Yêu cầu bắt buộc trước khi dùng dữ liệu thật:** loại bỏ thông tin định danh (hàm `lam_sach` đã làm một phần), và thông báo cho nhân viên rằng dữ liệu họ nhập được dùng để huấn luyện mô hình. |
| 7 | Nếu mô hình ngừng hoạt động, hệ thống còn chạy không, và ai biết? | **Còn chạy.** Hệ thống gọi dịch vụ này **phải** có thời gian chờ tối đa và suy giảm có kiểm soát: mô hình chết thì nhân viên chọn nhóm lỗi bằng tay như trước khi có AI. Mã mẫu và phép thử ở `docs/deployment-guide.md` mục 6. **Ai biết:** `HEALTHCHECK` của Docker phát hiện trong vòng 10 giây; hệ thống gọi ghi nhật ký mức `warn` mỗi lần không gọi được. |
