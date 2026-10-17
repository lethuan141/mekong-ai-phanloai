# Khám phá dữ liệu (EDA) — ghi chép

> **Đây KHÔNG phải sản phẩm nộp.** Tệp này là ghi chép khám phá dữ liệu, tương
> đương một notebook. Sản phẩm nộp là `python -m src.models.train` chạy được bằng
> một lệnh và sinh ra `reports/metrics.json`.
>
> Repo mẫu dùng Markdown thay cho `.ipynb` để bản ghi này đọc được trực tiếp trên
> GitHub và xem được khác biệt giữa hai phiên bản. Trong bài của em, `.ipynb` cũng
> hoàn toàn được chấp nhận.

## Việc đầu tiên phải làm: xem phân bố nhãn

Bước này mất 30 giây và quyết định toàn bộ những gì làm sau đó.

```python
import pandas as pd
df = pd.read_csv('data/mau/issue_descriptions.csv')
print(df.shape)
print(df['issue_category'].value_counts())
print(df['issue_category'].value_counts(normalize=True).round(3))
```

**Kết quả:**

| Nhãn | Số mẫu | Tỉ lệ |
|---|---:|---:|
| MAN_HINH | 518 | 21,6% |
| PIN | 434 | 18,1% |
| KHAC | 428 | 17,8% |
| PHAN_MEM | 380 | 15,8% |
| SAC | 375 | 15,6% |
| NUOC_VAO | 265 | 11,0% |
| **Tổng** | **2.400** | 100% |

### Ba câu hỏi phải trả lời được TRƯỚC KHI làm tiếp

**1. Nhãn nào nhiều nhất, chiếm bao nhiêu phần trăm?**
`MAN_HINH`, 21,6%. Đó chính là **độ chính xác của mô hình "luôn đoán nhãn đa số"**.
Nếu mô hình của em không hơn con số này, nó vô dụng. Đây là lý do phải có mô hình
cơ sở — và vì sao **không dùng độ chính xác** làm chỉ số chính.

**2. Nhãn nào ít nhất, có đủ mẫu để học không?**
`NUOC_VAO`, 265 mẫu. Sau khi chia 70/15/15 thì tập test chỉ còn khoảng 40 mẫu của
nhãn này — đủ để đo nhưng số liệu sẽ dao động. Phải báo cáo chỉ số **tách theo
từng nhãn**, không chỉ chỉ số tổng thể.

**3. Có nhãn nào là thùng rác không?**
`KHAC`, 17,8%. Nhãn này nhận mọi thứ không thuộc năm nhóm còn lại: không nhận sim,
loa rè, nút nguồn kẹt, khay sim móp, wifi tự ngắt. Dự đoán: nó sẽ **hút về nhiều
dự đoán sai** và có F1 thấp. Cần xem lại định nghĩa nhãn cùng người dùng nghiệp vụ.

## Xem vài mẫu thật của từng nhãn

```python
for nhan in df['issue_category'].unique():
    print('##', nhan)
    for v in df[df.issue_category == nhan]['issue_desc'].head(3):
        print('  ', v)
```

**Phát hiện quan trọng:** một số mô tả **về bản chất là nhập nhằng** giữa hai nhãn:

| Mô tả | Có thể là | Vì sao nhập nhằng |
|---|---|---|
| `máy không lên nguồn` | PIN hoặc SAC | Không lên nguồn có thể do pin chết hoặc do sạc không vào |
| `cắm sạc không lên gì` | PIN hoặc SAC | Cùng lý do trên |
| `sạc không vào điện` | PIN hoặc SAC | Cùng lý do trên |
| `màn hình đứng không bấm được` | MAN_HINH hoặc PHAN_MEM | Đứng máy có thể do phần mềm treo |
| `loa bị rè sau khi máy ẩm` | NUOC_VAO hoặc KHAC | Cả hai đều hợp lý |

Đây **không phải lỗi dữ liệu** — nhân viên tiếp nhận trong thực tế cũng không phân
biệt được. Nó có nghĩa là **tồn tại một trần trên** cho độ chính xác đạt được, và
mô hình sẽ nhầm lẫn đúng ở các cặp nhãn này. Dự đoán trước khi huấn luyện:
**PIN ↔ SAC sẽ là cặp bị nhầm nhiều nhất.**

## Độ dài mô tả

```python
df['so_tu'] = df['issue_desc'].str.split().str.len()
print(df['so_tu'].describe())
print('mô tả ngắn (< 10 từ):', (df['so_tu'] < 10).mean())
```

Khoảng **30%** mô tả có dưới 10 từ. Mô tả ngắn có ít thông tin hơn, nên dự đoán
mô hình sẽ hoạt động kém hơn với nhóm này → **đây là một chiều để đo thiên lệch**.

## Cách viết không nhất quán

Ba cách viết cùng tồn tại trong dữ liệu:

| Cách viết | Ví dụ | Tỉ lệ |
|---|---|---|
| Có dấu, chữ thường | `màn hình bị nứt góc dưới` | ~84% |
| Không dấu | `man hinh bi nut goc duoi` | ~12% |
| CHỮ HOA | `MÀN HÌNH BỊ NỨT GÓC DƯỚI` | ~4% |

**Quyết định:** hạ chữ thường (giải quyết trường hợp CHỮ HOA), nhưng **GIỮ DẤU**.
Bỏ dấu để gộp trường hợp không dấu sẽ làm mất thông tin của 84% dữ liệu còn lại:
`sạc` và `sách` đều thành `sach`. Đánh đổi này đã ghi vào `docs/model-card.md` mục 5.

## Kết luận của bước khám phá — bốn quyết định

1. **Chỉ số chính là F1 macro**, không phải độ chính xác: nhãn mất cân bằng
   (11%–21,6%) và mọi nhãn đều quan trọng như nhau về nghiệp vụ.
2. **Bắt buộc có hai mô hình cơ sở:** đoán nhãn đa số (mức sàn) và
   TF-IDF + hồi quy logistic (mức chuẩn).
3. **Phải báo cáo chỉ số tách theo từng nhãn** vì `NUOC_VAO` chỉ có 265 mẫu.
4. **Đo thiên lệch theo hai chiều:** độ dài mô tả (ngắn/dài) và trung tâm bảo hành.

## Điều KHÔNG làm ở bước này

- Không fit vectorizer nào. Bước khám phá chỉ **đếm và xem**; mọi bộ học tham số
  chỉ được fit **sau khi chia dữ liệu**, trong `src/models/train.py`.
- Không thử mô hình để "xem thử độ chính xác bao nhiêu" trên toàn bộ dữ liệu —
  con số đó sẽ neo kỳ vọng vào một giá trị bị rò rỉ.
