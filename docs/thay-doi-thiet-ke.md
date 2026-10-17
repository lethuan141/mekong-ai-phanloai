# Nhật ký thay đổi thiết kế

Mọi sai lệch so với thiết kế ban đầu (Bài tập 1) đều ghi ở đây, mỗi thay đổi ba
dòng: **thay đổi gì · vì sao · ảnh hưởng tới thành phần nào**.

## 1. Thay bộ dữ liệu mẫu bằng bộ sinh ra có nhiễu nhãn và câu nhập nhằng

- **Thay đổi gì:** thiết kế ban đầu dùng trực tiếp `issue_descriptions.csv` của
  case study. Nay dùng bộ sinh bởi `data/mau/gen_mau.py`.
- **Vì sao:** bộ gốc cho F1 macro = **1,000** vì các nhãn được sinh từ những mẫu
  câu riêng biệt nên từ vựng gần như không chồng nhau. Điểm tuyệt đối làm bộ dữ
  liệu vô dụng với mục đích của repo này: confusion matrix không có ô nào ngoài
  đường chéo, không có nhãn nào recall thấp, không có thiên lệch nào để đo — ba
  trong mười mục của Model card không có gì thật để báo cáo. Xem `BUG-01`.
- **Ảnh hưởng:** `data/mau/gen_mau.py` (tệp mới, có docstring giải thích đầy đủ),
  `configs/train.yaml` (đường dẫn dữ liệu), toàn bộ mục 4, 6, 8, 9 của
  `docs/model-card.md`, `docs/test-plan.md` mục 4, `docs/bug-log.md` BUG-01.
  **F1 giảm từ 1,000 xuống 0,832 — đó là một cải thiện**, vì 0,832 nói được điều
  gì đó thật còn 1,000 thì không.

## 2. Bỏ dấu thời gian khỏi gói lưu mô hình

- **Thay đổi gì:** thiết kế ban đầu lưu `tao_luc` (thời điểm huấn luyện) trong
  tệp `model_v1.joblib`. Nay bỏ trường đó.
- **Vì sao:** dấu thời gian làm nội dung tệp khác nhau ở hai lần chạy, phá vỡ
  phép thử "hai lần chạy cho kết quả giống hệt" và làm không thể dùng `md5sum`
  để xác minh một tệp mô hình được sinh từ đúng mã nguồn và đúng dữ liệu. Xem
  `BUG-02`.
- **Ảnh hưởng:** `src/models/train.py`. Thời điểm chạy vẫn được ghi — nhưng vào
  **nhật ký** và **tên tệp báo cáo**, là những nơi dấu thời gian đúng chỗ.
  **Nguyên tắc chung:** mọi giá trị thay đổi theo mỗi lần chạy không được đưa vào
  sản phẩm cần tái lập.

## 3. Thêm ngưỡng tin cậy riêng cho mô tả ngắn

- **Thay đổi gì:** thiết kế ban đầu có một ngưỡng tin cậy duy nhất. Nay có hai:
  `NGUONG_TIN_CAY` (0,45) cho mô tả từ 10 từ trở lên và
  `NGUONG_TIN_CAY_MO_TA_NGAN` (0,60) cho mô tả ngắn hơn.
- **Vì sao:** đo được rằng mô hình hoạt động kém hơn với mô tả ngắn (F1 0,8187 so
  với 0,8329), và ~30% mô tả thuộc nhóm này. Biện pháp giảm thiểu này **đã được
  ghi trong Model card trước khi được hiện thực** — đó chính là `BUG-03`, một lỗi
  nhất quán giữa tài liệu và sản phẩm.
- **Ảnh hưởng:** `src/serve/app.py`, `.env.example`, `docker-compose.yml`,
  `docs/model-card.md` mục 9, `docs/deployment-guide.md` mục 3, test `SV-04`.
  **Bài học:** mỗi biện pháp giảm thiểu ghi trong Model card phải có một test
  chứng minh nó tồn tại trong sản phẩm.

## 4. Không commit tệp mô hình; ảnh Docker tự huấn luyện khi dựng

- **Thay đổi gì:** thiết kế ban đầu dự kiến commit `model_v1.joblib` vào repo.
  Nay `.gitignore` chặn `artifacts/*`, và `Dockerfile` có bước
  `RUN python -m src.models.train` khi dựng ảnh.
- **Vì sao:** tệp mô hình là tệp nhị phân, không xem được khác biệt giữa hai
  phiên bản, nên nó không thuộc về hệ thống quản lý phiên bản mã nguồn. Nhưng
  Bài tập 3 yêu cầu **gói đóng gói tự đủ**. Giải pháp dung hoà: repo chỉ chứa
  script sinh ra mô hình cộng hạt giống cộng phiên bản thư viện chính xác; ảnh
  Docker tự huấn luyện khi dựng.
- **Ảnh hưởng:** `.gitignore`, `Dockerfile`, `docs/deployment-guide.md` mục 2.
  **Đánh đổi:** dựng ảnh lâu hơn khoảng 40 giây. Chấp nhận được, và nó bảo đảm
  mô hình trong ảnh luôn khớp với mã nguồn trong ảnh.

## 5. Dùng phép chia cố định thay vì kiểm định chéo

- **Thay đổi gì:** thiết kế ban đầu dự kiến kiểm định chéo phân tầng 5 phần. Nay
  dùng một phép chia 70/15/15 phân tầng với hạt giống cố định.
- **Vì sao:** kiểm định chéo cho con số ổn định hơn kèm độ lệch chuẩn, nhưng chậm
  hơn năm lần và làm bước huấn luyện trong `Dockerfile` mất quá lâu. Ở phạm vi
  prototype, một phép chia cố định với hạt giống ghi rõ là đủ để **so sánh các
  mô hình với nhau** — là mục đích của bước chọn mô hình.
- **Ảnh hưởng:** `src/data/split.py`, `docs/test-plan.md` mục 1 (đã ghi vào phần
  chủ ý không kiểm). **Hạn chế đã biết:** không có độ lệch chuẩn nên không kết
  luận được liệu chênh lệch 0,031 giữa `complement_nb` và `tfidf_logistic` có ý
  nghĩa hay chỉ là dao động. Đã nêu điều này ở `docs/model-card.md` mục 7.

## 6. Thêm bộ đếm chặn việc đo tập test lần thứ hai

- **Thay đổi gì:** thêm lớp `BoDemTapTest` ném `TapTestDaDung` nếu tập test được
  đo lần thứ hai trong một lần chạy.
- **Vì sao:** biến một **nguyên tắc** ("tập test dùng đúng một lần") thành một
  **phép kiểm tự động**. Nguyên tắc phụ thuộc vào việc người ta có nhớ hay không;
  phép kiểm hoạt động dù người ta có nhớ hay không.
- **Ảnh hưởng:** `src/models/train.py`, test `TL-07` và `TL-08`.

## 7. Thêm cột `center_id` vào bộ dữ liệu mẫu

- **Thay đổi gì:** bộ dữ liệu mẫu có thêm cột `center_id`, **không** dùng làm
  đặc trưng đầu vào của mô hình.
- **Vì sao:** để có **hai** cách nhóm khi đo thiên lệch (Bài tập 3 yêu cầu ít
  nhất hai). Cột này chỉ dùng để **chia nhóm khi đánh giá**.
- **Ảnh hưởng:** `data/mau/gen_mau.py`, `src/evaluate.py` (hàm `do_thien_lech`),
  `docs/model-card.md` mục 8. **Lưu ý quan trọng:** vì `center_id` được gán ngẫu
  nhiên, chênh lệch đo theo nó **là nhiễu**. Đã nêu rõ điều này trong Model card
  thay vì tuyên bố có thiên lệch — xem `docs/test-report.md` ghi chú số 2.
