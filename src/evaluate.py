"""Do va bao cao. Bao gom DO THIEN LECH giua cac nhom."""
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             recall_score)


def do_theo_nhan(y_that, y_du_doan, nhan_labels: list) -> dict:
    """Chi so TACH THEO TUNG NHAN.

    Chi so tong the che giau viec mo hinh bo roi hoan toan mot nhan it mau,
    nen bao cao phai co ca chi so theo tung nhan.
    """
    f1 = f1_score(y_that, y_du_doan, average=None, labels=nhan_labels,
                  zero_division=0)
    rc = recall_score(y_that, y_du_doan, average=None, labels=nhan_labels,
                      zero_division=0)
    return {
        'f1_macro': round(float(f1_score(y_that, y_du_doan, average='macro',
                                         zero_division=0)), 4),
        'f1_theo_nhan': {n: round(float(v), 4) for n, v in zip(nhan_labels, f1)},
        'recall_theo_nhan': {n: round(float(v), 4) for n, v in zip(nhan_labels, rc)},
        'confusion_matrix': confusion_matrix(y_that, y_du_doan,
                                             labels=nhan_labels).tolist(),
        'nhan_labels': list(nhan_labels),
        'bao_cao_chi_tiet': classification_report(
            y_that, y_du_doan, labels=nhan_labels, zero_division=0,
            output_dict=True),
    }


def doc_confusion_matrix(cm: list, nhan_labels: list) -> dict:
    """Bien bang so thanh BA KET LUAN doc duoc.

    Mot confusion matrix dan vao bao cao ma khong co ba ket luan nay thi gan nhu
    khong duoc diem — xem docs/model-card.md muc 8 va 9.
    """
    m = np.array(cm, dtype=float)
    n = len(nhan_labels)

    # 1. Cap nham nhieu nhat (tong hai chieu)
    cap, so_nham = None, -1
    for i in range(n):
        for j in range(i + 1, n):
            tong = m[i, j] + m[j, i]
            if tong > so_nham:
                so_nham, cap = tong, (nhan_labels[i], nhan_labels[j])

    # 2. Nhan co recall thap nhat (trong cac nhan co mau that)
    co_mau = [i for i in range(n) if m[i].sum() > 0]
    recall = {nhan_labels[i]: m[i, i] / m[i].sum() for i in co_mau}
    nhan_kem = min(recall, key=recall.get) if recall else None

    # 3. Nhan "thung rac": hut ve nhieu du doan sai nhat tu cac nhan khac
    hut_ve = {nhan_labels[j]: float(m[:, j].sum() - m[j, j]) for j in range(n)}
    nhan_thung_rac = max(hut_ve, key=hut_ve.get) if hut_ve else None

    return {
        'cap_nham_nhieu_nhat': {'cap': list(cap) if cap else None,
                                'so_lan': int(so_nham) if so_nham >= 0 else 0},
        'nhan_recall_thap_nhat': {'nhan': nhan_kem,
                                  'recall': round(recall[nhan_kem], 4)
                                  if nhan_kem else None},
        'nhan_hut_du_doan_sai_nhieu_nhat': {'nhan': nhan_thung_rac,
                                            'so_du_doan_sai': int(hut_ve[nhan_thung_rac])
                                            if nhan_thung_rac else 0},
    }


def do_thien_lech(df_test: pd.DataFrame, y_that, y_du_doan, cot_nhom: str,
                  nhan_labels: list, so_mau_toi_thieu: int = 30) -> dict:
    """Do F1 macro TACH THEO NHOM va tinh chenh lech giua nhom tot nhat va kem nhat.

    Mot chi so tong the tot van co the che giau viec mo hinh hoat dong kem han
    voi mot nhom du lieu nao do.
    """
    d = df_test.assign(_that=list(y_that), _du_doan=list(y_du_doan))
    theo_nhom = {}
    for ten_nhom, nhom in d.groupby(cot_nhom, sort=True):
        if len(nhom) < so_mau_toi_thieu:
            # So lieu tren nhom qua it mau khong dang tin — ghi nhan, khong do.
            theo_nhom[str(ten_nhom)] = {
                'so_mau': int(len(nhom)),
                'ghi_chu': f'khong do — duoi nguong {so_mau_toi_thieu} mau'}
            continue
        theo_nhom[str(ten_nhom)] = {
            'so_mau': int(len(nhom)),
            'f1_macro': round(float(f1_score(
                nhom['_that'], nhom['_du_doan'], average='macro',
                zero_division=0)), 4),
        }

    da_do = {k: v['f1_macro'] for k, v in theo_nhom.items() if 'f1_macro' in v}
    ra = {'cot_nhom': cot_nhom, 'theo_nhom': theo_nhom}
    if len(da_do) >= 2:
        tot, kem = max(da_do, key=da_do.get), min(da_do, key=da_do.get)
        ra.update({
            'nhom_tot_nhat': {'nhom': tot, 'f1_macro': da_do[tot]},
            'nhom_kem_nhat': {'nhom': kem, 'f1_macro': da_do[kem]},
            'chenh_lech': round(da_do[tot] - da_do[kem], 4),
        })
    return ra
