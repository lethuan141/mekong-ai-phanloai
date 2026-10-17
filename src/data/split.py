"""CHIA DU LIEU — buoc phai lam TRUOC moi bien doi hoc tham so.

Day la quy tac quan trong nhat cua track AI. Neu fit vectorizer (hoac bo chuan
hoa, bo dien gia tri thieu, bo chon dac trung) tren toan bo du lieu roi moi chia,
thi bo do da "nhin thay" tu vung cua tap test va ket qua do tren tap test cao
hon thuc te — con so bao cao khong con y nghia.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def chia(df: pd.DataFrame, cot_nhan: str, hat_giong: int,
         ti_le_test: float = 0.15, ti_le_val: float = 0.15
         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia thanh train / val / test, PHAN TANG theo nhan.

    Phan tang (stratify) bao dam moi tap deu co du ti le cac nhan. Chia thuong
    co the tao ra mot tap khong co mau nao cua nhan it gap, lam chi so cua tap
    do vo nghia.
    """
    tong_giu = ti_le_test + ti_le_val
    if not 0 < tong_giu < 1:
        raise ValueError('ti_le_test + ti_le_val phai nam trong (0, 1).')

    train, tam = train_test_split(
        df, test_size=tong_giu, random_state=hat_giong, stratify=df[cot_nhan])
    # Chia phan con lai thanh val va test theo dung ti le tuong doi
    ti_le_test_trong_tam = ti_le_test / tong_giu
    val, test = train_test_split(
        tam, test_size=ti_le_test_trong_tam, random_state=hat_giong,
        stratify=tam[cot_nhan])

    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
