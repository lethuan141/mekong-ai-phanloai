# Checklist bảo mật — tự đánh giá

Bảy mục bắt buộc của Bài tập 3. Mỗi mục ghi **ĐÃ LÀM / CHƯA LÀM / KHÔNG ÁP DỤNG**
kèm một dòng minh chứng. Ghi "CHƯA LÀM" trung thực không bị trừ điểm.

| # | Mục | Kết quả | Minh chứng |
|---|---|---|---|
| 1 | Không có secret nào trong repo | **ĐÃ LÀM** | Repo không cần secret nào: dịch vụ không kết nối cơ sở dữ liệu và không gọi API bên ngoài. `.env` nằm trong `.gitignore`; `.env.example` chỉ có ngưỡng và cổng. Kiểm: `git log --all --full-history -- .env` → rỗng; `grep -rn "api_key\|password\|token" src/` → rỗng. |
| 2 | Dữ liệu vào được kiểm ở tầng ứng dụng | **ĐÃ LÀM** | Pydantic `Field(min_length=5, max_length=2000)` chặn mô tả rỗng, quá ngắn và quá dài → trả 422. Thêm một phép kiểm sau khi làm sạch: nếu không còn nội dung nào thì cũng trả 422 (test `SV-08`). Giới hạn 2.000 ký tự là biện pháp chống yêu cầu quá lớn làm cạn bộ nhớ. Bốn loại dữ liệu sai đều có test — `SV-06`. |
| 3 | Kiểm soát truy cập | **KHÔNG ÁP DỤNG** ở phạm vi prototype | Dịch vụ không có dữ liệu người dùng và không có tài nguyên thuộc sở hữu của ai — mọi yêu cầu đều tương đương. Không có gì để phân quyền. **Khi triển khai thật:** đặt dịch vụ trong mạng nội bộ, chỉ cho hệ thống tiếp nhận gọi được; không mở ra internet. Đã ghi ở `docs/deployment-guide.md` mục 3. |
| 4 | Dữ liệu nhạy cảm được xử lý trước khi dùng | **ĐÃ LÀM** | Hàm `lam_sach` **thay** số điện thoại, số serial và địa chỉ thư điện tử bằng thẻ `<sdt>`, `<serial>`, `<email>` **trước khi** vector hoá. Nhờ vậy mô hình không học theo danh tính khách hàng, và từ vựng TF-IDF không chứa thông tin định danh. Có ba test: `TX-03`, `TX-04`, `TX-05`. |
| 5 | Thông báo lỗi không để lộ cấu trúc bên trong | **ĐÃ LÀM** | FastAPI trả lỗi xác thực dữ liệu ở dạng có cấu trúc, không kèm vết ngăn xếp. Có test `SV-09` kiểm rằng thân phản hồi lỗi **không** chứa `traceback`, `site-packages`, `/app/`, `sklearn`, `joblib` — tức là không lộ đường dẫn tệp, phiên bản thư viện hay cấu trúc mô hình. |
| 6 | Nhật ký không ghi secret hay dữ liệu cá nhân | **ĐÃ LÀM** | Nhật ký mỗi lần dự đoán chỉ ghi: **số từ**, nhãn dự đoán, độ tin cậy, ngưỡng đã áp dụng, thời gian xử lý. **TUYỆT ĐỐI không ghi nội dung mô tả** — mô tả có thể chứa số điện thoại, tên khách hay số serial. Kiểm bằng cách đọc `src/serve/app.py` hàm `du_doan`: lệnh `log.info` chỉ nhận năm giá trị số và một tên nhãn. Xem `BUG-04` về việc đã xoá một dòng nhật ký dư. |
| 7 | Phụ thuộc được ghim phiên bản và đã kiểm cảnh báo lỗ hổng | **MỘT PHẦN** | **ĐÃ LÀM:** `requirements.txt` ghim phiên bản chính xác bằng `==` cho cả chín gói; ảnh nền ghim `python:3.12-slim`, **không** dùng thẻ `latest`. Ghim `scikit-learn==1.5.2` đặc biệt quan trọng: tệp mô hình lưu bởi phiên bản khác có thể nạp lỗi (đã ghi ở `docs/deployment-guide.md` mục 7). **CHƯA LÀM:** chưa chạy `pip-audit`. Việc cần làm: `pip install pip-audit && pip-audit -r requirements.txt`. |

## Năm mục bổ sung đã làm, ngoài bảy mục bắt buộc

| Mục | Minh chứng |
|---|---|
| Container chạy bằng người dùng không đặc quyền | `Dockerfile`: `RUN useradd -m -u 10001 app` rồi `USER app` |
| Giới hạn kích thước đầu vào | `max_length=2000` ở lược đồ Pydantic — chống mô tả rất dài làm dịch vụ treo hoặc hết bộ nhớ |
| Mô hình nạp **một lần** khi khởi động | `lifespan` của FastAPI. Nạp lại mỗi yêu cầu vừa chậm gấp trăm lần vừa là một đường tấn công gây cạn tài nguyên |
| Không commit tệp mô hình vào repo | `.gitignore` chặn `artifacts/*`. Ảnh Docker tự huấn luyện khi dựng, nên gói vẫn tự đủ mà repo không phình to |
| Có endpoint `/health` riêng | Người vận hành kiểm tra được dịch vụ mà không cần gửi một yêu cầu dự đoán thật |

## Ba hạn chế đã biết và chủ ý không xử lý ở phạm vi học phần

1. **Chưa có giới hạn số lần gọi (rate limiting).** Ai gọi liên tục có thể chiếm
   hết tài nguyên của dịch vụ. Ở phạm vi prototype chỉ có giới hạn độ dài đầu
   vào. Cách xử lý đúng: giới hạn ở tầng proxy, cộng với thời gian chờ tối đa và
   suy giảm có kiểm soát ở phía gọi (`docs/deployment-guide.md` mục 6).
2. **Chưa có HTTPS.** Dịch vụ chạy HTTP trong mạng nội bộ. Khi triển khai thật
   phải đặt sau proxy có chứng chỉ TLS.
3. **Chưa ghi nhật ký ai gọi dịch vụ.** Không có xác thực nên cũng không có danh
   tính để ghi. Khi thêm xác thực thì phải ghi nhật ký mã người gọi để truy vết
   được khi có khiếu nại về một gợi ý cụ thể.
