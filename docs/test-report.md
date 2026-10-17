# Test Report — Phân loại tự động mô tả sự cố

Lần chạy: `pytest -q` — **40 PASS, 1 SKIP có lý do** (41 test được thu thập). Bốn test tích hợp TC-31…TC-34 chạy trên
hệ thống dựng bằng Docker.

| Mã | Tiền điều kiện | Các bước | Dữ liệu thử | Kết quả mong đợi | Ưu tiên | Kết quả thực tế | P/F |
|---|---|---|---|---|---|---|---|
| TX-01 | Không | Gọi `lam_sach` với văn bản có dấu | `Màn hình bị nứt góc dưới` | Giữ nguyên dấu tiếng Việt, hạ chữ thường | Cao | `màn hình bị nứt góc dưới` | PASS |
| TX-03 | Không | Gọi `lam_sach` với mô tả chứa số điện thoại | `khách 0912345678 báo máy không lên nguồn` | Có `<sdt>`; **không còn** chuỗi `0912345678` | **Cao** | Có `<sdt>`, không còn số | PASS |
| TX-04 | Không | Gọi `lam_sach` với ba cách viết số serial | `SN-10293` · `sn10293` · `SN10293` | Cả ba thành `<serial>`; không còn `10293` | Cao | Đúng cả ba | PASS |
| TX-06 | Không | **Giá trị biên rỗng** | `None` · `''` · `'   '` · `float('nan')` | Cả bốn trả về chuỗi rỗng | Cao | Đúng cả bốn | PASS |
| TX-09 | Không | Gọi `dem_tu` trên văn bản có số điện thoại | `khách 0912345678 báo vỡ` | 4 từ (số điện thoại thành một thẻ) | Trung bình | 4 | PASS |
| RR-01 | Đã chia dữ liệu với hạt giống 42 | Kiểm giao của tập `desc_id` giữa ba tập | 2.400 dòng → 1.680 / 360 / 360 | Ba tập **không chồng nhau**; tổng bằng hợp | **Cao** | Không chồng nhau | PASS |
| RR-02 | Như trên | So sánh tỉ lệ nhãn giữa train và val/test | Sáu nhãn | Cả hai tập có **đủ sáu nhãn**; sai lệch tỉ lệ < 6 điểm phần trăm | **Cao** | Đủ sáu nhãn, sai lệch tối đa 1,4 điểm | PASS |
| RR-03 | Đã chia dữ liệu với hạt giống 42 | **Thí nghiệm có đối chứng hai chiều.** Chèn từ mốc `zzmatukhongcothat` vào 4 dòng của tập test, rồi (a) fit vectorizer chỉ trên train, (b) fit trên train+test | Từ mốc xuất hiện 4 lần, chỉ ở tập test | (a) từ mốc **không** trong từ vựng; (b) từ mốc **có** trong từ vựng | **Cao** | Đúng cả hai chiều — phép kiểm chứng minh được là nó biết phát hiện lỗi | PASS |
| RR-03b | Như trên | Phép kiểm **tự nhiên**: từ nào chỉ có ở tập test thì không được nằm trong từ vựng fit trên train | 2.400 dòng | Giao rỗng, **với điều kiện** tập test có từ riêng | Cao | Tập test có **0** từ riêng (từ vựng train 353 từ, test 319 từ, test ⊂ train) → test **tự bỏ qua kèm lý do** thay vì xanh giả | SKIP |
| RR-04 | Như trên | So từ vựng của vectorizer trong Pipeline với vectorizer fit riêng trên train, và với vectorizer fit trên train+test | — | Bằng cái fit trên train; **khác** cái fit trên train+test | **Cao** | Đúng như mong đợi | PASS |
| RR-06 | Đã có `artifacts/model_v1.joblib` | Nạp **chính tệp mô hình sẽ triển khai**, so từ vựng của vectorizer trong đó với vectorizer fit lại chỉ trên train, và với vectorizer fit trên cả ba tập | Tệp mô hình v1.0 | Bằng cái fit trên train; **khác** cái fit trên toàn bộ dữ liệu | **Cao** | 1.308 = 1.308 từ (trùng khít), và khác 1.391 từ của bản fit toàn bộ | PASS |
| RR-07 | Như trên | So `so_mau_train` ghi trong tệp mô hình với kích thước tập train dựng lại | — | Bằng nhau | Cao | 1.680 = 1.680 | PASS |
| TL-01 | Cấu hình có `hat_giong: 42` | Chạy `train.chay()` hai lần, so sánh `metrics.json` | Cùng cấu hình, cùng dữ liệu | Hai tệp **giống hệt** | **Cao** | Giống hệt | PASS |
| TL-03 | Như trên | Kiểm báo cáo có hai mô hình cơ sở, và mô hình đã chọn tốt hơn mức sàn | — | Có `nhan_da_so` và `tfidf_logistic`; F1 test > F1 mức sàn | **Cao** | 0,8318 > 0,0587 | PASS |
| TL-05 | Như trên | Kiểm ba kết luận từ confusion matrix | — | Cả ba trường đều có giá trị, không `None` | Cao | Cặp nhầm nhiều nhất: PIN ↔ SAC (17 lần); nhãn recall thấp nhất: PHAN_MEM (0,667); nhãn hút dự đoán sai nhiều nhất: MAN_HINH (16) | PASS |
| TL-06 | Như trên | Kiểm đo thiên lệch theo ít nhất hai cách nhóm | — | ≥ 2 cách nhóm, mỗi cách ≥ 2 nhóm | **Cao** | Hai cách: độ dài mô tả (chênh 0,014) và `center_id` (chênh 0,160) | PASS |
| TL-07 | Như trên | Kiểm số lần đo tập test trong một lần chạy | — | Đúng **1** | **Cao** | 1 | PASS |
| TL-08 | Không | Gọi `BoDemTapTest.dung()` hai lần | — | Lần thứ hai ném `TapTestDaDung` | **Cao** | Ném đúng ngoại lệ | PASS |
| TL-09 | Không | Nạp cấu hình **thiếu** `hat_giong` | `cot_van_ban: issue_desc` | Ném `ValueError` nêu rõ thiếu `hat_giong` | Cao | Ném đúng | PASS |
| SV-01 | Đã có `artifacts/model_v1.joblib` | Gọi `GET /health` | — | 200; có `phien_ban_mo_hinh` khác `khong_ro`; `so_nhan` = 6 | Cao | `{"status":"ok","phien_ban_mo_hinh":"v1.0","ten_mo_hinh":"complement_nb","so_nhan":6}` | PASS |
| SV-02 | Như trên | `POST /du-doan` với mô tả hợp lệ | `khách báo cổng sạc bị lỏng phải giữ mới vào điện` | 200; đúng 3 phần tử trong `top_3`; sắp xếp **giảm dần** theo độ tin cậy | Cao | 3 phần tử, giảm dần | PASS |
| SV-04 | Như trên | So ngưỡng áp dụng cho mô tả **ngắn** và **dài** | `màn hình sọc dọc` (4 từ) vs mô tả 17 từ | Mô tả ngắn dùng ngưỡng **cao hơn** (0,60 vs 0,45) | **Cao** | 0,60 vs 0,45 | PASS |
| SV-05 | Như trên | Dự đoán với mô tả **nhập nhằng** giữa PIN và SAC | `máy không lên nguồn cắm sạc không lên gì` | Nếu dưới ngưỡng: `nhan_goi_y` là `null` và `can_nguoi_xem_lai` là `true` | **Cao** | Độ tin cậy 0,326 < 0,45 → `nhan_goi_y: null`, `can_nguoi_xem_lai: true` | PASS |
| SV-06 | Như trên | **Bốn loại dữ liệu sai** | `''` · `'abc'` · `'   '` · 5.000 ký tự | Cả bốn trả về **422** | Cao | 422 cả bốn | PASS |
| SV-08 | Như trên | Mô tả dài hơn 5 ký tự nhưng sau khi làm sạch không còn nội dung | `!!!!! ????? .....` | 422 | Trung bình | 422 | PASS |
| SV-09 | Như trên | Đọc thân phản hồi lỗi, tìm dấu hiệu lộ thông tin nội bộ | Mô tả `abc` | **Không** chứa `traceback`, `site-packages`, `/app/`, `sklearn`, `joblib` | **Cao** | Không chứa chuỗi nào | PASS |
| TC-31 | Hệ thống dựng bằng `docker compose up -d --build` | `curl -s localhost:8080/health` | — | `{"status":"ok",...}` | Cao | Đúng như mong đợi | PASS |
| TC-32 | Như trên | `docker compose ps` | — | STATUS của `model` chứa `healthy` | Cao | healthy | PASS |
| TC-33 | Như trên | Chạy `diff` giữa `metrics.json` của hai lần huấn luyện liên tiếp | — | Không có khác biệt | **Cao** | Không có khác biệt | PASS |
| TC-34 | Như trên | `curl` với `{"mo_ta":"abc"}` và đọc mã trạng thái | — | 422 | Cao | 422 | PASS |

## Ghi chú về tính trung thực — hai điểm quan trọng

**1. Bộ dữ liệu mẫu đã bị THAY thế có ý.** Bộ ban đầu (lấy từ
`issue_descriptions.csv` của case study) cho F1 macro = **1,000** — điểm tuyệt đối,
vì các nhãn được sinh từ những mẫu câu riêng biệt nên từ vựng gần như không chồng
nhau. Điểm tuyệt đối làm bộ dữ liệu đó vô dụng với mục đích giảng dạy: confusion
matrix không có ô nào ngoài đường chéo, không có nhãn nào recall thấp, không có
thiên lệch nào để đo. Vì vậy `data/mau/gen_mau.py` sinh một bộ mới có nhiễu nhãn
và câu nhập nhằng. **Lý do đầy đủ nằm trong docstring của tệp đó.** Điểm F1 giảm
từ 1,000 xuống 0,832 — đó là một cải thiện, không phải suy giảm.

**2. Chênh lệch thiên lệch theo `center_id` (0,160) rất có thể là NHIỄU, không
phải thiên lệch thật.** Trong `gen_mau.py`, `center_id` được gán **hoàn toàn ngẫu
nhiên**, nên về mặt xây dựng nó **không** có quan hệ nào với nhãn hay với mô tả.
Chênh lệch 0,160 xuất hiện vì mỗi nhóm chỉ có 51–72 mẫu, và F1 macro trên nhóm
nhỏ dao động mạnh. Đây là bài học quan trọng: **một chênh lệch đo trên nhóm nhỏ
có thể là nhiễu.** Cách xử lý đúng đã ghi ở `docs/model-card.md` mục 8 — không
tuyên bố có thiên lệch mà nêu rõ giới hạn của phép đo và điều kiện để kết luận.
