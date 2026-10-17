# Mekong AI – Phân loại tự động mô tả sự cố

Dự án thuộc học phần **Chuyên đề tốt nghiệp 1 – Khoa Công nghệ thông tin, Trường Đại học Văn Lang**.

- **Track:** AI
- **Case study:** Smart CRM – Mekong Mobile
- **Mục tiêu:** Hỗ trợ nhân viên tiếp nhận phân loại nhóm sự cố dựa trên mô tả lỗi của thiết bị.
- **Phạm vi:** Xây dựng quy trình xử lý dữ liệu, huấn luyện mô hình phân loại văn bản, đánh giá kết quả và hỗ trợ dự đoán.

## 1. Mục tiêu và chức năng

Dự án hướng đến các chức năng sau:

- Đọc dữ liệu mô tả sự cố từ tệp CSV.
- Kiểm tra dữ liệu, xử lý bản ghi thiếu thông tin và thống kê phân bố nhãn.
- Khám phá dữ liệu bằng thống kê mô tả và biểu đồ.
- Chia dữ liệu thành tập huấn luyện, xác thực và kiểm thử trước khi học tham số.
- Huấn luyện và đánh giá mô hình phân loại văn bản.
- Lưu kết quả đánh giá vào tệp báo cáo để phục vụ việc theo dõi và so sánh.
- Hỗ trợ xây dựng dịch vụ dự đoán qua FastAPI nếu cấu hình triển khai được bật.

Các kết quả đánh giá cần được xác nhận bằng cách chạy chương trình thực tế trước khi đưa vào báo cáo chính thức.

## 2. Yêu cầu môi trường

- Python 3.11 trở lên.
- Git để quản lý mã nguồn.
- Các thư viện được khai báo trong `requirements.txt`.
- Matplotlib phục vụ trực quan hóa dữ liệu.
- Nếu chạy dịch vụ API, cần cài đặt các thư viện và cấu hình tương ứng trong dự án.

### Cài đặt môi trường

Mở Terminal hoặc CMD tại thư mục dự án:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```cmd
.venv\Scripts\activate
```

Cài đặt thư viện:

```bash
python -m pip install -r requirements.txt
```

## 3. Cấu trúc dự án

```text
mekong-ai-phanloai/
├── configs/
│   └── train.yaml
├── data/
│   └── mau/
│       ├── issue_descriptions.csv
│       └── gen_mau.py
├── docs/
│   ├── ai-disclosure.md
│   ├── bug-log.md
│   ├── checklist-bao-mat.md
│   ├── deployment-guide.md
│   ├── model-card.md
│   ├── test-plan.md
│   ├── test-report.md
│   └── thay-doi-thiet-ke.md
├── notebooks/
│   └── 01_eda.md
├── reports/
│   └── metrics.json
├── src/
│   ├── data/
│   │   ├── load.py
│   │   ├── split.py
│   │   └── eda.py
│   ├── features/
│   │   └── text.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── train.py
│   ├── serve/
│   │   └── app.py
│   ├── config.py
│   └── evaluate.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Cấu trúc trên thể hiện các thành phần chính của dự án. Khi cập nhật README, cần đối chiếu với các tệp thực tế trong repository và bổ sung những tệp còn thiếu.

## 4. Phân tích dữ liệu EDA

EDA (Exploratory Data Analysis) được thực hiện nhằm hiểu đặc điểm của dữ liệu trước khi huấn luyện mô hình.

Chạy chương trình phân tích dữ liệu:

```bash
python -m src.data.eda
```

Chương trình thống kê:

- Số lượng dòng và cột.
- Số bản ghi trùng lặp.
- Số lượng và tỷ lệ của từng nhãn.
- Thống kê độ dài mô tả sự cố.
- Tỷ lệ mô tả ngắn hơn 10 từ.
- Biểu đồ phân bố nhãn được lưu tại `docs/eda-phan-bo-nhan.png`.

**Lưu ý về sản phẩm nộp:** Tệp `notebooks/01_eda.md` là tài liệu Markdown, không phải Jupyter Notebook. Nếu đề bài yêu cầu notebook EDA có thống kê mô tả và biểu đồ, cần tạo thêm `notebooks/01_eda.ipynb` và chạy các ô lệnh để thể hiện kết quả phân tích.

## 5. Huấn luyện và đánh giá mô hình

Tệp cấu hình huấn luyện:

```text
configs/train.yaml
```

Chạy huấn luyện theo cấu hình:

```bash
python -m src.models.train --config configs/train.yaml
```

Sau khi chạy thành công, kiểm tra các kết quả đầu ra được cấu hình trong dự án, bao gồm:

- Mô hình đã huấn luyện.
- Báo cáo chỉ số đánh giá.
- Các chỉ số phục vụ việc so sánh và lựa chọn mô hình.

Tệp báo cáo mặc định được khai báo trong cấu hình là:

```text
reports/metrics.json
```

Cần kiểm tra kết quả thực tế sau khi chạy; không sử dụng số liệu giả định để thay thế kết quả đánh giá.

## 6. Kiểm thử

Chạy bộ kiểm thử bằng lệnh:

```bash
pytest -q
```

Nếu các bài kiểm thử thất bại, cần xem thông báo lỗi, xác định nguyên nhân và sửa trước khi kết luận hệ thống hoạt động đúng.

Các nội dung cần kiểm tra gồm:

- Quy trình xử lý và chia dữ liệu.
- Tính nhất quán của quá trình huấn luyện.
- Các chức năng xử lý văn bản.
- Khả năng hoạt động của dịch vụ dự đoán nếu được triển khai.

Chỉ ghi số lượng test thành công sau khi có kết quả chạy thực tế.

## 7. Trạng thái hiện tại

Các phần đã được bổ sung vào repository:

- [x] Khởi tạo cấu trúc dự án và quản lý mã nguồn bằng Git.
- [x] Tạo tệp ghi nhận thay đổi thiết kế tại `docs/thay-doi-thiet-ke.md`.
- [x] Bổ sung chương trình EDA tại `src/data/eda.py`.
- [x] Tạo biểu đồ phân bố nhãn tại `docs/eda-phan-bo-nhan.png`.
- [x] Bổ sung thư viện Matplotlib vào `requirements.txt`.

Các phần cần kiểm tra hoặc hoàn thiện trước khi nộp:

- [ ] Chạy lại chương trình EDA và xác nhận kết quả.
- [ ] Tạo Jupyter Notebook `notebooks/01_eda.ipynb` có thống kê mô tả, biểu đồ và nhận xét.
- [ ] Chạy huấn luyện mô hình và xác nhận báo cáo được sinh thành công.
- [ ] Chạy `pytest -q` và ghi nhận kết quả.
- [ ] Kiểm tra `.gitignore` và `.env.example`, bảo đảm không đưa khóa bí mật hoặc tệp mô hình lớn lên GitHub.
- [ ] Kiểm tra các thay đổi đã được commit và push lên GitHub.
- [ ] Cập nhật trạng thái sau khi hoàn tất các bước kiểm thử.

## 8. Nhật ký thay đổi thiết kế

Tệp `docs/thay-doi-thiet-ke.md` ghi nhận những thay đổi so với thiết kế ban đầu, lý do kỹ thuật và các thành phần bị ảnh hưởng.

Nội dung này giúp theo dõi quá trình phát triển và giải thích những quyết định thiết kế trong dự án.

## 9. Tài liệu dự án

| Tệp | Nội dung |
|---|---|
| `docs/thay-doi-thiet-ke.md` | Nhật ký thay đổi thiết kế |
| `docs/ai-disclosure.md` | Khai báo sử dụng công cụ AI |
| `docs/test-plan.md` | Kế hoạch kiểm thử |
| `docs/test-report.md` | Báo cáo kiểm thử |
| `docs/bug-log.md` | Nhật ký lỗi và cách xử lý |
| `docs/model-card.md` | Thông tin và giới hạn của mô hình |
| `docs/deployment-guide.md` | Hướng dẫn triển khai |
| `docs/checklist-bao-mat.md` | Danh sách kiểm tra bảo mật |

## 10. Quản lý mã nguồn

Dự án được quản lý bằng Git và lưu trữ trên GitHub.

Các thay đổi nên được ghi nhận bằng commit có nội dung rõ ràng theo Conventional Commits, chẳng hạn:

```bash
git add README.md
git commit -m "docs: update environment and project status"
git push origin master
```

Repository: https://github.com/lethuan141/mekong-ai-phanloai
