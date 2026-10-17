"""Nap du lieu tho. Khong bien doi gi ngoai viec bo dong thieu nhan."""
import pathlib
import pandas as pd


def nap(duong_dan: pathlib.Path, cot_van_ban: str, cot_nhan: str) -> pd.DataFrame:
    if not duong_dan.exists():
        raise FileNotFoundError(
            f'Khong tim thay tep du lieu {duong_dan}. '
            f'Kiem duong_dan_du_lieu trong configs/train.yaml.')
    df = pd.read_csv(duong_dan, dtype=str, keep_default_na=False, na_values=[''])

    for cot in (cot_van_ban, cot_nhan):
        if cot not in df.columns:
            raise ValueError(f'Tep du lieu thieu cot {cot}. '
                             f'Cac cot co: {list(df.columns)}')

    # Dong thieu NHAN thi khong dung de huan luyen duoc — nhung phai DEM va
    # BAO CAO, khong am tham bo.
    truoc = len(df)
    df = df[df[cot_nhan].notna() & (df[cot_nhan].str.strip() != '')]
    df = df[df[cot_van_ban].notna() & (df[cot_van_ban].str.strip() != '')]
    da_bo = truoc - len(df)

    return df.reset_index(drop=True).assign(_so_dong_bi_bo=da_bo)
