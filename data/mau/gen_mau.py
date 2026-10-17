#!/usr/bin/env python3
"""Sinh bo du lieu mau CO THAT TINH THUC TE cho repo mau track AI.

VI SAO CAN TEP NAY
------------------
Bo du lieu goc `issue_descriptions.csv` cua case study duoc sinh tu cac mau cau
RIENG BIET cho tung nhom loi, nen tu vung cua sau nhom gan nhu khong chong nhau.
Ket qua: mot mo hinh TF-IDF don gian dat F1 macro = 1,000 — diem tuyet doi.

Diem tuyet doi lam bo du lieu do VO DUNG voi muc dich giang day cua repo nay:
confusion matrix khong co o nao ngoai duong cheo, khong co nhan nao recall thap,
khong co thien lech nao de do. Ba trong muoi muc cua Model card khong co gi
that de bao cao.

Tep nay sinh mot bo mau co ba dac diem CO Y, mo phong du lieu that:

  1. CAU MO TA NHAP NHANG dung chung giua PIN va SAC.
     "khong len nguon", "het pin rat nhanh", "cam sac khong vao dien" — trong
     thuc te nhan vien tiep nhan cung khong phan biet duoc, va do la nguyen
     nhan that cua viec hai nhan nay bi nham lan.

  2. NHIEU NHAN (~7%). Mot ti le mo ta bi gan sai nhan, mo phong viec nguoi
     nhap chon nhanh cho xong. Day la dieu xay ra o moi bo du lieu that co nhan
     do nguoi gan.

  3. NHAN KHAC LA "THUNG RAC". Nhan KHAC nhan cac mo ta thuoc nhieu chu de khac
     nhau, nen no hut ve nhieu du doan sai — dung nhu du lieu that.

Thêm hai dac diem de DO THIEN LECH:

  4. MO TA NGAN (duoi 10 tu) chiem ~22%, va mo ta ngan CO Y it thong tin hon,
     nen mo hinh hoat dong kem han voi nhom nay. Day la thien lech do duoc.

  5. Cot `center_id` de do thien lech theo trung tam.

CHAY LAI DUOC: dung hat giong co dinh, nen chay lai cho ra dung tep nay.
    python data/mau/gen_mau.py
"""
import csv
import pathlib
import random
import unicodedata

HAT_GIONG = 20260901
SO_MAU = 2400
TI_LE_NHIEU_NHAN = 0.07
TI_LE_MO_TA_NGAN = 0.22

RA = pathlib.Path(__file__).resolve().parent / 'issue_descriptions.csv'

# ---------------------------------------------------------------- ngu lieu
# Cum tu DAC TRUNG cho tung nhan — giup mo hinh hoc duoc.
DAC_TRUNG = {
    'MAN_HINH': [
        'màn hình bị nứt góc dưới', 'màn hình sọc dọc nhiều đường',
        'màn hình đen không thấy gì', 'cảm ứng không nhận ở nửa trên',
        'màn hình bị chảy mực loang', 'kính màn hình vỡ nhưng vẫn hiển thị',
        'màn hình nhấp nháy liên tục', 'điểm chết giữa màn hình',
    ],
    'PIN': [
        'pin phồng làm hở viền máy', 'pin tụt từ tám mươi xuống hai mươi trong nửa giờ',
        'máy nóng bất thường khi dùng nhẹ', 'pin chai chỉ dùng được hai tiếng',
        'máy tự tắt khi còn ba mươi phần trăm', 'pin sụt nhanh cả khi để không',
    ],
    'SAC': [
        'cổng sạc bị lỏng phải giữ mới vào', 'chân sạc gỉ xanh',
        'sạc vào rất chậm mất cả đêm', 'củ sạc nóng và có mùi khét',
        'cáp sạc chỉ vào khi cắm một chiều', 'chân cắm bị vênh',
    ],
    'PHAN_MEM': [
        'ứng dụng tự thoát liên tục', 'máy treo logo khi khởi động',
        'không cài được ứng dụng báo lỗi', 'quên mật khẩu tài khoản',
        'máy chạy chậm sau khi cập nhật', 'camera mở lên bị đen',
        'bàn phím ảo không hiện lên',
    ],
    'NUOC_VAO': [
        'máy rơi vào bồn rửa', 'điện thoại dính nước mưa cả buổi',
        'đổ nước ngọt vào bàn phím', 'máy ngâm nước biển khi đi tắm',
        'rơi vào bồn cầu đã lau khô', 'nước vào loa nghe ọc ọc',
    ],
    'KHAC': [
        'không nhận sim', 'loa ngoài rè không nghe rõ',
        'nút nguồn bị kẹt', 'rung không hoạt động',
        'không kết nối được tai nghe không dây', 'khay sim bị móp',
        'mất tiếng khi gọi nhưng loa ngoài vẫn tốt', 'wifi tự ngắt liên tục',
    ],
}

# CUM TU NHAP NHANG — dung chung giua nhieu nhan.
# Day la nguyen nhan THAT cua viec cac nhan bi nham lan.
NHAP_NHANG = {
    ('PIN', 'SAC'): [
        'máy không lên nguồn', 'cắm sạc không lên gì',
        'sạc không vào điện', 'máy không nhận sạc',
        'hết pin rất nhanh và sạc lâu', 'cắm sạc vào không thấy báo gì',
        'máy chỉ lên khi cắm sạc', 'sạc đầy nhưng rút ra là tắt',
    ],
    ('MAN_HINH', 'PHAN_MEM'): [
        'màn hình đứng không bấm được', 'máy treo không tắt được',
        'hình bị nhoè khi mở ứng dụng',
    ],
    ('NUOC_VAO', 'KHAC'): [
        'loa bị rè sau khi máy ẩm', 'máy có hơi nước trong camera',
    ],
    ('PHAN_MEM', 'KHAC'): [
        'máy không nhận thẻ nhớ', 'không đăng nhập được vào máy',
    ],
}

MO_DAU = ['khách hàng phản ánh ', 'kh nói ', 'khách báo ', 'tình trạng: ',
          'khách mang máy tới, ', '']
KET = ['', ' xảy ra sau khi rơi', ' bị từ tuần trước', ' mới bị 2 ngày nay',
       ' đã thử khởi động lại không được', ' máy còn bảo hành',
       ' khách chưa từng sửa ở đâu', ' đã mang đi sửa một lần']


def bo_dau(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.replace('đ', 'd').replace('Đ', 'D')


def sinh():
    rnd = random.Random(HAT_GIONG)

    # Ti le nhan: co y KHONG can bang, mo phong du lieu that.
    ti_le = {'MAN_HINH': 0.22, 'PIN': 0.18, 'SAC': 0.16,
             'PHAN_MEM': 0.15, 'NUOC_VAO': 0.11, 'KHAC': 0.18}
    nhan_list, trong_so = list(ti_le), list(ti_le.values())

    dong = []
    for i in range(1, SO_MAU + 1):
        nhan_that = rnd.choices(nhan_list, weights=trong_so, k=1)[0]

        # 30% mo ta lay tu cum tu NHAP NHANG neu nhan nay co cap nhap nhang
        cac_cap = [c for c in NHAP_NHANG if nhan_that in c]
        if cac_cap and rnd.random() < 0.30:
            loi = rnd.choice(NHAP_NHANG[rnd.choice(cac_cap)])
        else:
            loi = rnd.choice(DAC_TRUNG[nhan_that])

        ngan = rnd.random() < TI_LE_MO_TA_NGAN
        if ngan:
            # Mo ta NGAN: it thong tin hon -> mo hinh kem hon voi nhom nay
            mo_ta = loi
        else:
            mo_ta = rnd.choice(MO_DAU) + loi + rnd.choice(KET)

        # ~12% viet khong dau, ~4% viet HOA — mo phong cach nhap thuc te
        r = rnd.random()
        if r < 0.12:
            mo_ta = bo_dau(mo_ta)
        elif r < 0.16:
            mo_ta = mo_ta.upper()

        # NHIEU NHAN: gan sai nhan cho mot ti le dong
        nhan_ghi = nhan_that
        if rnd.random() < TI_LE_NHIEU_NHAN:
            khac = [n for n in nhan_list if n != nhan_that]
            # Gan sai thuong xay ra giua cac nhan gan nghia hon
            uu_tien = [n for c in NHAP_NHANG if nhan_that in c
                       for n in c if n != nhan_that]
            nhan_ghi = rnd.choice(uu_tien) if uu_tien and rnd.random() < 0.7 \
                else rnd.choice(khac)

        dong.append({
            'desc_id': i,
            'issue_desc': mo_ta.strip(),
            'issue_category': nhan_ghi,
            'center_id': rnd.randint(1, 6),
        })

    with open(RA, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['desc_id', 'issue_desc',
                                           'issue_category', 'center_id'])
        wr.writeheader()
        wr.writerows(dong)

    from collections import Counter
    print(f'Da ghi {len(dong)} dong vao {RA}')
    print('Phan bo nhan:', dict(Counter(d['issue_category'] for d in dong)))
    ngan = sum(1 for d in dong if len(d['issue_desc'].split()) < 10)
    print(f'Mo ta ngan (< 10 tu): {ngan} ({ngan / len(dong):.1%})')


if __name__ == '__main__':
    sinh()
