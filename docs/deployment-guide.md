# Deployment Guide — Dịch vụ phân loại mô tả sự cố

## 1. Yêu cầu môi trường

| Thành phần | Phiên bản tối thiểu | Cách kiểm |
|---|---|---|
| Docker Engine | 24.0 | `docker --version` |
| Docker Compose | v2.20 | `docker compose version` |
| Dung lượng đĩa trống | 2,5 GB | `df -h .` |
| Cổng 8080 | phải trống | `ss -ltn \| grep 8080` (đổi bằng `MODEL_PORT`) |
| RAM khả dụng | 1 GB | Mô hình nạp vào bộ nhớ khi khởi động |

Không cần cài Python trên máy để chạy hệ thống. Để chạy **bộ kiểm thử** hoặc
huấn luyện lại ngoài Docker: Python 3.11 trở lên và `pip install -r requirements.txt`.

## 2. Các bước dựng hệ thống

```bash
# Bước 1 — lấy mã nguồn
git clone --branch v1.0 https://github.com/<tài-khoản>/<repo>.git
cd <repo>

# Bước 2 — tạo tệp cấu hình từ mẫu
cp .env.example .env

# Bước 3 — dựng và khởi động (đây là lệnh thứ hai, tổng cộng 2 lệnh)
docker compose up -d --build
```

**Quan trọng:** ảnh Docker **tự huấn luyện mô hình trong lúc dựng**
(`RUN python -m src.models.train` trong `Dockerfile`). Nhờ vậy gói đóng gói
**tự đủ**: người khác clone repo về là chạy được, không cần tệp mô hình đính kèm
và không cần tải mô hình từ đâu. Đánh đổi: dựng ảnh lâu hơn khoảng 40 giây.

Lần đầu mất khoảng 120–180 giây (tải ảnh nền, cài phụ thuộc, huấn luyện).

### Huấn luyện lại mô hình ngoài Docker

```bash
pip install -r requirements.txt
python -m src.models.train --config configs/train.yaml
#   Sinh ra artifacts/model_v1.joblib và reports/metrics.json
uvicorn src.serve.app:app --port 8080
```

## 3. Biến môi trường

| Biến | Ý nghĩa | Giá trị mẫu | Bắt buộc |
|---|---|---|---|
| `MODEL_PORT` | Cổng dịch vụ dự đoán mở ra máy chủ | `8080` | Không (mặc định 8080) |
| `MODEL_PATH` | Đường dẫn tệp mô hình bên trong container | `artifacts/model_v1.joblib` | Không (mặc định như bên) |
| `NGUONG_TIN_CAY` | Ngưỡng tin cậy cho mô tả **từ 10 từ trở lên**. Dưới ngưỡng thì trả `nhan_goi_y: null` và `can_nguoi_xem_lai: true` | `0.45` | Không (mặc định 0,45) |
| `NGUONG_TIN_CAY_MO_TA_NGAN` | Ngưỡng riêng cho mô tả **dưới 10 từ** — cao hơn, vì mô hình hoạt động kém hơn với mô tả ngắn. Đây là biện pháp giảm thiểu thiên lệch đã đo | `0.60` | Không (mặc định 0,60) |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` | Không (mặc định INFO) |

**Về việc điều chỉnh hai ngưỡng:** hai giá trị này quyết định **bao nhiêu phần
trăm ca được chuyển sang người xem lại**. Nâng ngưỡng → ít gợi ý sai hơn nhưng
nhiều ca phải xem tay hơn. Hạ ngưỡng → ngược lại. Đây là **quyết định nghiệp vụ**,
không phải quyết định kỹ thuật: phải thống nhất với trưởng bộ phận tiếp nhận, và
mỗi lần đổi phải ghi lý do vào `docs/model-card.md` mục 9.

**Khi triển khai thật, ba việc bắt buộc làm thêm:**

1. Đặt dịch vụ sau một proxy có chứng chỉ TLS; **không** mở cổng 8080 trực tiếp
   ra internet.
2. Thêm giới hạn số lần gọi (rate limiting) ở tầng proxy. Ở phạm vi prototype chỉ
   có giới hạn độ dài đầu vào (2.000 ký tự) — xem `docs/checklist-bao-mat.md` mục 2.
3. Nếu hệ thống gọi dịch vụ này, **phải** đặt thời gian chờ tối đa và xử lý suy
   giảm có kiểm soát: mô hình chết thì nhân viên chọn nhóm lỗi bằng tay, nghiệp vụ
   chính vẫn chạy. Xem mục 6 dưới đây.

## 4. Dữ liệu mẫu

Bộ mẫu nằm trong `data/mau/issue_descriptions.csv` — **2.400 dòng**, đủ nhỏ để
commit vào repo. Bộ này **được sinh ra** bởi `data/mau/gen_mau.py` với hạt giống
cố định `20260901`, nên tái lập được:

```bash
python data/mau/gen_mau.py
```

Bộ mẫu **cố ý** có ba đặc điểm của dữ liệu thật — lý do đầy đủ nằm trong docstring
của `gen_mau.py` và trong `BUG-01`:

| Đặc điểm | Chi tiết |
|---|---|
| Câu mô tả nhập nhằng | Dùng chung giữa PIN ↔ SAC, MAN_HINH ↔ PHAN_MEM, NUOC_VAO ↔ KHAC |
| Nhiễu nhãn | ~7% dòng bị gán sai nhãn, mô phỏng người nhập chọn nhanh |
| Nhãn thùng rác | `KHAC` nhận nhiều chủ đề khác nhau |
| Mô tả ngắn | ~30% dưới 10 từ — dùng để đo thiên lệch |
| Cách viết không nhất quán | ~12% không dấu, ~4% CHỮ HOA |
| Phân bố nhãn mất cân bằng | 11,0% (NUOC_VAO) tới 21,6% (MAN_HINH) |

Sau khi huấn luyện, `reports/metrics.json` có:

| Chỉ số | Giá trị |
|---|---|
| Số mẫu train / val / test | 1.680 / 360 / 360 |
| F1 macro — mô hình cơ sở đoán nhãn đa số | 0,0587 |
| F1 macro — mô hình cơ sở TF-IDF + logistic | 0,8008 |
| F1 macro — mô hình đã chọn (`complement_nb`) trên tập test | **0,8318** |

## 5. Cách kiểm tra hệ thống đã chạy đúng

Chạy lần lượt năm lệnh dưới đây. Cả năm phải cho đúng kết quả mong đợi.

```bash
# 5.1 — dịch vụ phải "healthy", không chỉ "running"
docker compose ps
#      Kết quả mong đợi: STATUS của model chứa "healthy".

# 5.2 — dịch vụ sống và đã nạp mô hình
curl -s http://localhost:8080/health
#      Kết quả mong đợi:
#      {"status":"ok","phien_ban_mo_hinh":"v1.0","ten_mo_hinh":"complement_nb","so_nhan":6}

# 5.3 — nghiệp vụ chính hoạt động: dự đoán với một mô tả rõ ràng
curl -s -X POST http://localhost:8080/du-doan \
  -H 'Content-Type: application/json' \
  -d '{"mo_ta":"khách báo cổng sạc bị lỏng phải giữ mới vào điện"}'
#      Kết quả mong đợi: JSON có đúng 3 phần tử trong top_3, có do_tin_cay,
#      can_nguoi_xem_lai, nguong_da_ap_dung, so_tu_mo_ta, phien_ban_mo_hinh.

# 5.4 — mô tả NHẬP NHẰNG phải được gắn cờ cần người xem lại
curl -s -X POST http://localhost:8080/du-doan \
  -H 'Content-Type: application/json' \
  -d '{"mo_ta":"máy không lên nguồn cắm sạc không lên gì"}' \
  | python -c "import json,sys; b=json.load(sys.stdin); \
    print('goi_y =', b['nhan_goi_y'], '| xem_lai =', b['can_nguoi_xem_lai'])"
#      Kết quả mong đợi: goi_y = None | xem_lai = True
#      (độ tin cậy khoảng 0,33 < ngưỡng 0,45)

# 5.5 — dữ liệu vào không hợp lệ bị TỪ CHỐI, dịch vụ không sập
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/du-doan \
  -H 'Content-Type: application/json' -d '{"mo_ta":"abc"}'
#      Kết quả mong đợi: 422
```

**Phép thử TÁI LẬP ĐƯỢC** (mục kiểm chứng quan trọng nhất của track AI):

```bash
python -m src.models.train --config configs/train.yaml
cp reports/metrics.json /tmp/lan1.json
md5sum artifacts/model_v1.joblib > /tmp/md5_lan1

python -m src.models.train --config configs/train.yaml
diff /tmp/lan1.json reports/metrics.json && echo "metrics.json: GIONG HET"
md5sum -c /tmp/md5_lan1 && echo "model.joblib: GIONG HET"
```

Cả hai phải giống hệt. Nếu `model.joblib` khác nhau thì có một giá trị thay đổi
theo mỗi lần chạy đã lọt vào gói lưu mô hình — xem `BUG-02`.

## 6. Suy giảm có kiểm soát — bắt buộc với hệ thống gọi dịch vụ này

Dịch vụ mô hình là một thành phần **có thể chết**: hết bộ nhớ, mô hình nạp lỗi,
mạng nội bộ chậm. Đây là điều bình thường, không phải trường hợp hiếm.

**Nguyên tắc:** mô hình chết thì **nghiệp vụ chính vẫn phải chạy** — nhân viên
chọn nhóm lỗi bằng tay như trước khi có AI. Một tính năng phụ trợ **không được**
đánh sập nghiệp vụ chính.

```javascript
// Trong hệ thống gọi dịch vụ mô hình
const ctrl = new AbortController();
const hetHan = setTimeout(() => ctrl.abort(),
                          Number(process.env.MODEL_TIMEOUT_MS || 1500));
try {
  const res = await fetch(`${process.env.MODEL_URL}/du-doan`, {
    method: 'POST', signal: ctrl.signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mo_ta: moTa }),
  });
  if (!res.ok) throw new Error(`model tra ve ${res.status}`);
  return { nguon: 'mo_hinh', ...(await res.json()) };
} catch (e) {
  // Mo hinh chet -> mat phan goi y, KHONG mat chuc nang tiep nhan phieu
  logger.warn('khong goi duoc dich vu mo hinh, chuyen sang chon tay',
              { ly_do: e.message });
  return { nguon: 'chon_tay', nhan_goi_y: null, can_nguoi_xem_lai: true };
} finally {
  clearTimeout(hetHan);
}
```

**Phép thử:** dừng dịch vụ mô hình (`docker compose stop model`) rồi gọi chức
năng tiếp nhận phiếu. Chức năng đó **phải** vẫn tạo được phiếu, chỉ không có nhãn
gợi ý. Nếu nó trả về 500 thì chưa xử lý suy giảm và phải sửa trước khi triển khai.

**Bắt buộc phải có thời gian chờ tối đa.** Không có timeout thì một dịch vụ mô
hình treo sẽ làm mọi yêu cầu tiếp nhận phiếu treo theo, và hệ thống sập vì cạn
nguồn kết nối.

## 7. Gỡ lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `FileNotFoundError: artifacts/model_v1.joblib` | Chưa huấn luyện mô hình (khi chạy ngoài Docker) | `python -m src.models.train --config configs/train.yaml` |
| Dịch vụ trả 503 | Mô hình chưa nạp xong | Chờ 20 giây; xem `docker compose logs model --tail 30` |
| Mô hình nạp lỗi sau khi đổi phiên bản scikit-learn | Tệp mô hình lưu bởi phiên bản khác | Huấn luyện lại. `requirements.txt` ghim `scikit-learn==1.5.2`; đổi phiên bản thì **phải** sinh lại mô hình |
| Hai lần huấn luyện cho `metrics.json` khác nhau | Thiếu hạt giống, hoặc có giá trị thay đổi theo lần chạy lọt vào báo cáo | Kiểm `hat_giong` trong `configs/train.yaml`; xem `BUG-02` |
| F1 macro bằng 1,0000 | **Dấu hiệu cảnh báo, không phải thành công.** Kiểm rò rỉ dữ liệu, rồi kiểm xem bài toán có tách được hoàn hảo không | Chạy `pytest tests/test_no_leakage.py`; xem `BUG-01` |
| Dịch vụ luôn trả `can_nguoi_xem_lai: true` | Ngưỡng đặt quá cao | Đọc `reports/metrics.json` để biết phân bố độ tin cậy thực tế, rồi bàn với bộ phận nghiệp vụ trước khi hạ ngưỡng |
