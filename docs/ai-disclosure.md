# Bảng khai báo sử dụng công cụ AI

Bắt buộc có trong mọi bài nộp, kể cả khi không sử dụng công cụ AI. Bốn cột: công
cụ · dùng vào việc gì · áp dụng ở tệp hoặc hàm nào · em đã kiểm chứng thế nào.

| Công cụ | Dùng vào việc gì | Áp dụng ở tệp / hàm nào | Em đã kiểm chứng thế nào |
|---|---|---|---|
| *(ví dụ)* GitHub Copilot | Gợi ý các biểu thức chính quy thay thông tin định danh bằng thẻ | `src/features/text.py`, các hằng `_RE_SDT`, `_RE_SERIAL`, `_RE_EMAIL` | Viết ba test `TX-03`, `TX-04`, `TX-05`, mỗi test kiểm **cả hai chiều**: thẻ có xuất hiện, và giá trị gốc **không** còn trong kết quả. Bản gợi ý ban đầu của `_RE_SERIAL` chỉ khớp `SN-10293` mà bỏ sót `SN10293` không có dấu gạch ngang; em phát hiện khi thêm ba biến thể vào test và tự sửa thành `\bSN-?\d+\b` kèm cờ `IGNORECASE`. |
| *(ví dụ)* ChatGPT | Hỏi cách bảo đảm vectorizer không bị fit trên tập test | `src/models/train.py`, hàm `cac_ung_vien`; `tests/test_no_leakage.py` | Bản gợi ý đề xuất dùng `sklearn.pipeline.Pipeline` — em đọc tài liệu chính thức của scikit-learn để xác nhận rằng bộ biến đổi trong Pipeline chỉ được `fit` trên dữ liệu truyền vào `.fit()`. Sau đó em tự viết `RR-04` **chứng minh** điều đó bằng cách so từ vựng của vectorizer trong Pipeline với vectorizer fit riêng trên train, và với vectorizer fit trên train+test. |
| *(ví dụ)* ChatGPT | Giải thích vì sao `LinearSVC` không có `predict_proba` và cách lấy thang đo tin cậy | `src/serve/app.py`, hàm `_xac_suat` | Đọc tài liệu scikit-learn về `decision_function` và về `CalibratedClassifierCV`. Em nhận ra một điều bản gợi ý **không** nói: softmax trên điểm quyết định **không** cho xác suất đã hiệu chỉnh. Em tự ghi hạn chế này vào `docs/model-card.md` mục 5 và mục 9 (rủi ro 4), vì nếu không ghi thì người dùng sẽ đọc `do_tin_cay = 0.7` là "đúng 70% số lần" — sai. |
| — không dùng — | *(các phần tự làm hoàn toàn)* | `src/evaluate.py` (toàn bộ, gồm `doc_confusion_matrix` và `do_thien_lech`) · `src/models/train.py` (lớp `BoDemTapTest` và toàn bộ trình tự chín bước) · `data/mau/gen_mau.py` (toàn bộ, gồm việc chọn các cặp nhãn nhập nhằng và tỉ lệ nhiễu nhãn) · toàn bộ `tests/` · toàn bộ `docs/` | Không áp dụng |

**Cam kết:** Tôi xác nhận đã đọc, hiểu và chịu trách nhiệm về toàn bộ nội dung nộp.

Họ tên: ………………………………  ·  MSSV: ………………  ·  Ngày: ……/……/2026

---

*Ghi chú của giảng viên: bảng trên là **ví dụ về mức độ chi tiết cần đạt**. Chú ý
rằng cả ba dòng đầu đều khai báo một chỗ mà bản gợi ý của AI **thiếu hoặc sai**,
và sinh viên đã tự phát hiện bằng test hoặc bằng cách đọc tài liệu chính thức.
Khai báo trung thực như vậy không bị trừ điểm mà được ghi nhận — nó chứng minh
sinh viên đã kiểm chứng thật, không chỉ chép và chạy.*
