# Bug Log — Phân loại tự động mô tả sự cố

## BUG-01 · Mức độ: **CRITICAL**

1. **Phát hiện bởi:** đánh giá kết quả lần huấn luyện đầu tiên (trước khi viết
   test `RR-03`).
2. **Triệu chứng:** F1 macro trên tập test = **1,0000** — điểm tuyệt đối trên cả
   sáu nhãn. Confusion matrix không có ô nào ngoài đường chéo. Cả ba mô hình ứng
   viên và cả mô hình cơ sở `tfidf_logistic` đều đạt 1,0000.
3. **Cách tái lập:** đặt `duong_dan_du_lieu` trỏ vào bản sao trực tiếp của
   `issue_descriptions.csv` của case study rồi chạy
   `python -m src.models.train --config configs/train.yaml`.
4. **Nguyên nhân:** **không phải rò rỉ dữ liệu** — đó là giả thuyết đầu tiên và
   đã bị loại bỏ sau khi kiểm bằng `RR-03` (từ vựng vectorizer không chứa từ riêng
   của tập test). Nguyên nhân thật: bộ dữ liệu nguồn được **sinh từ những mẫu câu
   riêng biệt cho từng nhãn**, nên tập từ vựng của sáu nhãn gần như không chồng
   nhau. Bài toán tách được hoàn hảo bằng bag-of-words, và một mô hình tuyến tính
   đơn giản đủ để đạt điểm tuyệt đối. **Một điểm tuyệt đối là dấu hiệu cảnh báo,
   không phải một thành công.**
5. **Cách sửa:** viết `data/mau/gen_mau.py` sinh bộ dữ liệu mẫu có ba đặc điểm
   của dữ liệu thật: (a) tập câu mô tả **nhập nhằng** dùng chung giữa các cặp nhãn
   gần nghĩa — `máy không lên nguồn`, `cắm sạc không lên gì` dùng chung cho PIN và
   SAC; (b) **nhiễu nhãn ~7%**, mô phỏng việc người nhập gán nhãn sai; (c) nhãn
   `KHAC` là **thùng rác** nhận nhiều chủ đề khác nhau. Toàn bộ lý do được ghi
   trong docstring của tệp đó, và hạt giống cố định `20260901` bảo đảm tái lập được.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/2f8b41c`
   *(repo mẫu này không có lịch sử Git; trong bài của em phải là link mở được)*
7. **Xác nhận:** sau khi sửa, F1 macro trên tập test = **0,8318**; mô hình cơ sở
   đoán nhãn đa số = 0,0587; mô hình cơ sở TF-IDF + logistic = 0,8008. Confusion
   matrix nay có nội dung đọc được: **PIN ↔ SAC nhầm 17 lần**, `PHAN_MEM` có
   recall thấp nhất 0,667, `MAN_HINH` hút về 16 dự đoán sai. Ba kết luận này đã
   vào `docs/model-card.md` mục 8 và 9 → **PASS**.
   **Điểm F1 giảm từ 1,000 xuống 0,832 là một cải thiện**, vì con số 0,832 nói
   được điều gì đó thật còn con số 1,000 thì không.

## BUG-02 · Mức độ: **MAJOR**

1. **Phát hiện bởi:** test `TL-01` (hai lần huấn luyện phải cho `metrics.json`
   giống hệt).
2. **Triệu chứng:** hai lần chạy `python -m src.models.train` liên tiếp với cùng
   cấu hình và cùng hạt giống cho ra hai tệp `artifacts/model_v1.joblib` **khác
   nhau về nội dung nhị phân**, dù `metrics.json` giống nhau. Kiểm bằng
   `md5sum artifacts/model_v1.joblib` sau mỗi lần chạy.
3. **Cách tái lập:** chạy huấn luyện hai lần, mỗi lần tính `md5sum` của tệp mô
   hình, rồi so sánh.
4. **Nguyên nhân:** gói lưu vào tệp mô hình có trường
   `tao_luc: datetime.now(timezone.utc).isoformat()`. Dấu thời gian khác nhau ở
   hai lần chạy làm nội dung tệp khác nhau, phá vỡ phép thử "hai lần chạy cho kết
   quả giống hệt". Hậu quả thực tế: không thể dùng `md5sum` hay `diff` để xác
   minh rằng một tệp mô hình được sinh từ đúng mã nguồn và đúng dữ liệu đó.
5. **Cách sửa:** bỏ trường `tao_luc` khỏi gói lưu mô hình, và ghi rõ lý do bằng
   một dòng chú thích ngay tại chỗ. Thời điểm chạy vẫn được ghi — nhưng ghi vào
   **nhật ký** và vào **tên tệp báo cáo**, là những nơi mà dấu thời gian đúng chỗ.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/a71d5e9`
7. **Xác nhận:** hai lần chạy nay cho `md5sum` **giống hệt** cho cả
   `artifacts/model_v1.joblib` và `reports/metrics.json` → **PASS**. Test `TL-01`
   PASS. Bài học chung: **dấu thời gian và mọi giá trị thay đổi theo mỗi lần chạy
   không được đưa vào sản phẩm cần tái lập** — chúng thuộc về nhật ký.

## BUG-03 · Mức độ: **MAJOR**

1. **Phát hiện bởi:** test `SV-04` (mô tả ngắn phải áp ngưỡng cao hơn).
2. **Triệu chứng:** dịch vụ dự đoán áp **cùng một ngưỡng tin cậy 0,45** cho mọi
   mô tả, kể cả mô tả bốn từ. Với mô tả ngắn, mô hình đưa ra nhãn gợi ý với độ
   tin cậy thấp mà **không** gắn cờ `can_nguoi_xem_lai`, nên nhân viên tiếp nhận
   nhận được một gợi ý trông chắc chắn trong khi mô hình thực tế không chắc.
3. **Cách tái lập:** gọi `POST /du-doan` với `{"mo_ta": "màn hình sọc dọc"}`
   (4 từ) và với một mô tả 17 từ; so sánh trường `nguong_da_ap_dung` của hai phản
   hồi — cả hai đều 0,45.
4. **Nguyên nhân:** biện pháp giảm thiểu thiên lệch theo độ dài mô tả đã được
   **ghi trong Model card** mục 9 nhưng **chưa được hiện thực** trong
   `src/serve/app.py`. Tài liệu mô tả một biện pháp không tồn tại — đây là lỗi
   **nhất quán** giữa tài liệu và sản phẩm, và là loại lỗi nghiêm trọng vì nó
   khiến người đọc tin rằng rủi ro đã được xử lý.
5. **Cách sửa:** thêm biến môi trường `NGUONG_TIN_CAY_MO_TA_NGAN` (mặc định 0,60)
   và hằng `SO_TU_COI_LA_NGAN = 10`; chọn ngưỡng theo số từ của mô tả; **trả về
   ngưỡng đã áp dụng** trong phản hồi (trường `nguong_da_ap_dung`) và số từ
   (trường `so_tu_mo_ta`) để phía gọi kiểm chứng được. Bổ sung cả hai biến vào
   `.env.example` và `docker-compose.yml`.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/6c4a2f1`
7. **Xác nhận:** `SV-04` PASS — mô tả 4 từ áp ngưỡng 0,60, mô tả 17 từ áp ngưỡng
   0,45. `SV-05` PASS — mô tả nhập nhằng `máy không lên nguồn cắm sạc không lên gì`
   cho độ tin cậy 0,326 < 0,45 nên trả `nhan_goi_y: null` và
   `can_nguoi_xem_lai: true` → đúng hành vi mong muốn.
   **Bài học:** mỗi biện pháp giảm thiểu ghi trong Model card phải có một test
   chứng minh nó tồn tại trong sản phẩm.

## BUG-04 · Mức độ: **MINOR** — đã sửa

1. **Phát hiện bởi:** đọc nhật ký dịch vụ khi làm `SV-02`.
2. **Triệu chứng:** mỗi lần dự đoán ghi **hai** dòng nhật ký, trong đó một dòng
   là `du doan xong` không có thông tin gì.
3. **Cách tái lập:** gọi `POST /du-doan` một lần rồi đọc `docker compose logs model`.
4. **Nguyên nhân:** còn sót một lệnh `log.info('du doan xong', extra={})` từ bản
   nháp trước, bên cạnh dòng nhật ký thật có đủ số liệu.
5. **Cách sửa:** xoá dòng đó.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/b09e337`
7. **Xác nhận:** mỗi lần dự đoán nay ghi đúng một dòng nhật ký, gồm số từ, nhãn,
   độ tin cậy, ngưỡng và thời gian xử lý — **không** ghi nội dung mô tả → **PASS**.

## BUG-05 · Mức độ: **MAJOR** — đã sửa

1. **Phát hiện bởi:** kiểm thử đột biến (*mutation testing*) — cố ý làm sai rồi
   xem test có đỏ không. Cụ thể: sửa `test_RR03` cho fit vectorizer trên
   `train + test`, tức là tạo ra **đúng loại rò rỉ mà test này tồn tại để bắt**.
2. **Triệu chứng:** `pytest` vẫn **xanh 5/5**. Test chống rò rỉ — test được ghi
   trong `test-plan.md` là *quan trọng nhất của track* — không phát hiện được
   một rò rỉ tạo ra ngay trong chính nó.
3. **Cách tái lập:** trong `tests/test_no_leakage.py`, đổi
   `vec.fit(train['_sach'])` thành `vec.fit(pd.concat([train['_sach'], test['_sach']]))`
   rồi chạy `pytest tests/test_no_leakage.py`.
4. **Nguyên nhân:** phép kiểm cũ so `từ vựng vectorizer` với `từ chỉ có ở tập
   test` rồi khẳng định giao phải rỗng. Nhưng đếm thực tế cho thấy tập test
   **không có từ nào riêng**: bộ dữ liệu mẫu sinh ra từ một kho cụm từ đóng
   (từ vựng train 353 từ, test 319 từ, và test là tập con của train). Giao của
   một tập với **tập rỗng** luôn rỗng, nên `assert not ro_ri` đúng vô điều kiện.
   Test **không có tiền đề nào để kiểm** — nó là một khẳng định trống.
5. **Cách sửa:** viết lại `RR-03` thành **thí nghiệm có đối chứng hai chiều**:
   chèn một từ mốc chỉ vào tập test rồi kiểm cả *chiều âm* (fit trên train →
   không thấy từ mốc) và *chiều dương* (fit trên train+test → **phải** thấy từ
   mốc). Chiều dương là phần chứng minh phép kiểm biết phát hiện lỗi. Từ mốc
   được chèn vào **4** dòng vì vectorizer đặt `min_df = 2`. Phép kiểm tự nhiên
   cũ giữ lại thành `RR-03b`, nhưng nay **tự bỏ qua kèm lý do** khi tập test
   không có từ riêng, thay vì xanh một cách giả.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/4d81ba2`
7. **Xác nhận:** chạy lại đúng phép đột biến ở bước 3 → `RR-03` **đỏ**. `RR-03b`
   báo SKIP kèm lý do, thấy được trong `pytest -rs`.
   **Bài học:** một test xanh chỉ có giá trị khi nó **có thể đỏ**. Cách kiểm
   điều đó không phải đọc lại code test, mà là cố ý làm sai một lần rồi xem test
   có bắt được không.

## BUG-06 · Mức độ: **CRITICAL** — đã sửa

1. **Phát hiện bởi:** kiểm thử đột biến lần thứ hai, sau khi sửa BUG-05. Lần này
   rò rỉ được chèn vào **sản phẩm** chứ không vào test: trong
   `src/models/train.py`, buộc từ vựng của vectorizer bằng từ vựng fit trên
   **toàn bộ** dữ liệu, rồi huấn luyện lại và chạy toàn bộ `pytest`.
2. **Triệu chứng:** **40/40 test vẫn xanh** dù mô hình vừa lưu ra đã nhìn thấy
   từ vựng của tập validation và tập test.
3. **Cách tái lập:** thêm ba dòng vào vòng huấn luyện trong `train.py` để gán
   `mo_hinh.named_steps['tfidf'].vocabulary` từ một vectorizer fit trên cả
   `df`, chạy `python -m src.models.train`, rồi `pytest -q`.
4. **Nguyên nhân:** `RR-03` và `RR-04` **tự dựng pipeline riêng ngay trong
   test**. Chúng chứng minh rằng *cách làm* đúng, nhưng không nói gì về
   `artifacts/model_v1.joblib` — thứ duy nhất thật sự được đem đi phục vụ. Bộ
   test kiểm một bản sao của quy trình chứ không kiểm sản phẩm.
5. **Cách sửa:** thêm `RR-06`: nạp chính tệp mô hình đã lưu, so từ vựng của
   vectorizer bên trong nó với vectorizer fit lại chỉ trên tập train (cùng hạt
   giống, cùng cấu hình) — phải **trùng khít**; và so với vectorizer fit trên cả
   ba tập — phải **khác**. Thêm `RR-07`: `so_mau_train` ghi trong tệp mô hình
   phải đúng bằng kích thước tập train.
6. **Link commit:** `https://github.com/<tài-khoản>/<repo>/commit/e57c093`
7. **Xác nhận:** chạy lại đúng phép đột biến ở bước 3 → `RR-06` **đỏ**, báo
   `luu 1391 tu, fit lai 1308 tu`. Hoàn nguyên `train.py`, huấn luyện lại →
   F1 macro (test) trở lại **0,8318**, mức sàn **0,0587** / **0,8008** không đổi,
   và `RR-06` xanh với 1.308 = 1.308 từ.
   **Bài học:** test phải chạm vào **thứ sẽ được triển khai**. Một bộ test dựng
   lại quy trình bên trong chính nó có thể xanh trọn vẹn trong khi tệp mô hình
   đem đi phục vụ đã sai — và đó là tệp người dùng gặp.


---

**Ghi chú sư phạm:** BUG-01, BUG-05 và BUG-06 là ba bug đáng chú ý nhất của repo này.
BUG-01 cho thấy Nó cho thấy một
điều mà sinh viên rất hay hiểu ngược: **một chỉ số quá tốt là lý do để điều tra,
không phải để ăn mừng.** Giả thuyết đầu tiên (rò rỉ dữ liệu) đã bị **loại bỏ bằng
một test** trước khi tìm ra nguyên nhân thật — đó là cách gỡ lỗi đúng: nêu giả
thuyết, kiểm chứng nó, rồi mới chuyển sang giả thuyết khác.
