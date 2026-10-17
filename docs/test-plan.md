# Test Plan rút gọn — Phân loại tự động mô tả sự cố

## 1. Phạm vi kiểm thử và phần chủ ý KHÔNG kiểm

**Được kiểm:**

- Hàm làm sạch văn bản: giữ dấu tiếng Việt, thay thông tin định danh bằng thẻ,
  xử lý giá trị rỗng và `NaN`.
- **Chống rò rỉ dữ liệu:** ba tập không chồng nhau, chia phân tầng, từ vựng
  vectorizer không chứa từ riêng của tập test.
- **Tính tái lập:** hai lần huấn luyện cùng hạt giống cho `metrics.json` giống hệt.
- **Tập test dùng đúng một lần** — có bộ đếm ném lỗi khi đo lần thứ hai.
- Báo cáo có đủ: điều kiện đo, hai mô hình cơ sở, chỉ số theo từng nhãn,
  confusion matrix, ba kết luận, đo thiên lệch theo hai cách nhóm.
- Dịch vụ dự đoán: luồng đúng, ngưỡng riêng cho mô tả ngắn, bốn loại dữ liệu sai,
  thông báo lỗi không lộ thông tin nội bộ.

**Chủ ý KHÔNG kiểm, kèm lý do:**

- **Chất lượng dự đoán trên dữ liệu thật của người dùng:** bộ mẫu là dữ liệu
  **sinh ra** (`data/mau/gen_mau.py`), không phải dữ liệu thật. Con số F1 0,83
  chỉ có ý nghĩa **tương đối** giữa các mô hình trên cùng bộ này. Đã ghi rõ ở
  `docs/model-card.md` mục 4.
- **Kiểm định chéo:** đã có phép chia phân tầng cố định và hạt giống ghi rõ. Kiểm
  định chéo 5 phần cho con số ổn định hơn nhưng chậm hơn năm lần; ở phạm vi
  prototype thì phép chia cố định là đủ. Đã ghi ở `docs/thay-doi-thiet-ke.md` mục 5.
- **Hiệu năng dịch vụ dưới tải:** thuộc Chuyên đề tốt nghiệp 2. Đã đo bằng tay
  thời gian một lần dự đoán (~2,5 ms) và ghi vào `docs/model-card.md` mục 10.
- **Hiệu chỉnh xác suất (calibration):** `ComplementNB` và `LinearSVC` không cho
  xác suất đã hiệu chỉnh. Đã ghi là hạn chế ở `docs/model-card.md` mục 5.

## 2. Loại test và số lượng mỗi tầng

| Tầng | Số lượng | Tệp | Ghi chú |
|---|---|---|---|
| Unit test hàm làm sạch văn bản | 12 | `tests/test_text.py` | Hàm thuần |
| Test chống rò rỉ dữ liệu | 8 | `tests/test_no_leakage.py` | Test quan trọng nhất của track. RR-03 dùng **thí nghiệm có đối chứng hai chiều**; RR-06 kiểm trên **chính tệp mô hình sẽ triển khai** |
| Test tính tái lập và nội dung báo cáo | 9 | `tests/test_tai_lap.py` | Có chạy huấn luyện thật |
| Test dịch vụ dự đoán | 12 | `tests/test_serve.py` | Dùng `TestClient` của FastAPI |
| **Tổng tự động** | **41** (40 chạy + 1 tự bỏ qua có lý do) | `pytest -q` | RR-03b tự bỏ qua vì bộ dữ liệu mẫu có từ vựng đóng |
| Test bằng tay | 4 | `docs/test-report.md` TC-31…TC-34 | Trên hệ thống dựng bằng Docker |

Tổng **41 test tự động** (40 chạy + 1 tự bỏ qua có lý do) và 4 test bằng tay.

**Bộ test này đã được kiểm bằng kiểm thử đột biến hai lần.** Lần đầu chèn rò rỉ
vào chính test chống rò rỉ → bộ test vẫn xanh (**BUG-05**). Lần sau chèn rò rỉ
vào `src/models/train.py` rồi huấn luyện lại → bộ test **vẫn xanh** (**BUG-06**),
vì không test nào chạm vào `artifacts/model_v1.joblib` — tệp duy nhất thật sự
được triển khai. RR-03 (thí nghiệm hai chiều) và RR-06 (kiểm trên tệp mô hình
đã lưu) lấp hai khe đó; chạy lại đúng hai phép đột biến ấy nay làm đỏ đúng hai
test tương ứng.

## 3. Tiêu chí thành công đo được

- Toàn bộ **40** test tự động PASS bằng một lệnh `pytest -q` (RR-03b tự bỏ qua kèm lý do — dùng `pytest -rs` để đọc lý do).
- Mỗi test quan trọng phải **đỏ được** khi điều nó kiểm bị làm sai (kiểm bằng kiểm thử đột biến).
- Test chống rò rỉ `RR-03` và `RR-04` PASS — điều kiện bắt buộc, không thương lượng.
- Hai lần chạy `python -m src.models.train` cho `reports/metrics.json` **giống hệt**
  (kiểm bằng `diff`).
- F1 macro trên tập test **cao hơn** mô hình cơ sở đoán nhãn đa số. Ngưỡng cụ thể:
  cao hơn ít nhất 0,50 (thực tế đạt 0,8318 so với 0,0587).
- `reports/metrics.json` có đủ tám trường điều kiện đo và ba kết luận từ
  confusion matrix.
- Mọi chênh lệch thiên lệch vượt ngưỡng 0,15 đều được nêu trong Model card kèm
  biện pháp giảm thiểu.
- Dịch vụ trả 422 cho cả bốn loại dữ liệu vào không hợp lệ, và **không** lộ vết
  ngăn xếp hay đường dẫn tệp.

## 4. Môi trường và dữ liệu thử

- **Test tự động:** `pytest -q` từ thư mục gốc repo. Ba tệp đầu không cần
  `artifacts/`; `tests/test_serve.py` tự **bỏ qua** (skip) nếu chưa có tệp mô hình,
  kèm thông báo nêu rõ lệnh cần chạy trước.
- **Dữ liệu thử:** `data/mau/issue_descriptions.csv` — 2.400 dòng, **sinh ra bởi**
  `data/mau/gen_mau.py` với hạt giống cố định `20260901`, nên tái lập được. Bộ này
  **cố ý** có ba đặc điểm của dữ liệu thật: câu mô tả nhập nhằng dùng chung giữa
  PIN và SAC, nhiễu nhãn ~7%, và nhãn `KHAC` là thùng rác. Lý do đầy đủ nằm trong
  docstring của `gen_mau.py`.
- **Hạt giống:** `42`, đặt trong `configs/train.yaml`. Mọi chỗ có ngẫu nhiên đều
  nhận hạt giống này.
- **Test tích hợp:** `docker compose up -d --build` rồi gọi API bằng `curl`.
