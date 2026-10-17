"""Dich vu du doan FastAPI.

Nam quyet dinh thiet ke, moi cai la mot cau tra loi cho bo ba cau hoi chuan:
  1. Nap mo hinh MOT LAN khi khoi dong, khong nap moi lan co yeu cau.
  2. Tra ve TOP 3 chu khong chi mot nhan — nhan vien can thay cac lua chon gan
     nhat de tu quyet.
  3. Co NGUONG TIN CAY va co "can nguoi xem lai" — mo hinh khong chac thi noi ra
     la khong chac, khong gia vo chac chan.
  4. Nguong RIENG cho mo ta ngan — bien phap giam thieu thien lech da do.
  5. Tra ve PHIEN BAN MO HINH trong moi phan hoi — de ba thang sau con truy vet duoc.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.text import lam_sach, dem_tu

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
log = logging.getLogger('serve')

MODEL_PATH = os.environ.get('MODEL_PATH', 'artifacts/model_v1.joblib')
NGUONG = float(os.environ.get('NGUONG_TIN_CAY', '0.45'))
NGUONG_NGAN = float(os.environ.get('NGUONG_TIN_CAY_MO_TA_NGAN', '0.60'))
SO_TU_COI_LA_NGAN = 10

trang_thai: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nap mo hinh MOT LAN khi khoi dong. Nap mo hinh mat hang tram mili giay;
    # nap lai moi lan co yeu cau bien dich vu nhanh thanh cham gap tram lan.
    goi = joblib.load(MODEL_PATH)
    trang_thai.update(goi)
    log.info('da nap mo hinh %s (%s) tu %s',
             goi.get('phien_ban'), goi.get('ten_mo_hinh'), MODEL_PATH)
    yield
    trang_thai.clear()


app = FastAPI(title='Phan loai su co — Mekong Mobile', version='1.0',
              lifespan=lifespan)


class YeuCau(BaseModel):
    mo_ta: str = Field(min_length=5, max_length=2000,
                       description='Mo ta su co do nhan vien tiep nhan nhap')


class NhanDuDoan(BaseModel):
    nhan: str
    do_tin_cay: float


class PhanHoi(BaseModel):
    nhan_goi_y: str | None
    do_tin_cay: float
    top_3: list[NhanDuDoan]
    can_nguoi_xem_lai: bool
    nguong_da_ap_dung: float
    so_tu_mo_ta: int
    phien_ban_mo_hinh: str


def _xac_suat(pipeline, van_ban: str) -> np.ndarray:
    """Lay xac suat cho tung nhan.

    LinearSVC khong co predict_proba, nen dung diem quyet dinh roi chuan hoa
    bang softmax. Day KHONG phai xac suat da hieu chinh — chi la mot thang do
    tin cay tuong doi. Da ghi han che nay trong docs/model-card.md muc 5.
    """
    if hasattr(pipeline, 'predict_proba'):
        return pipeline.predict_proba([van_ban])[0]
    d = pipeline.decision_function([van_ban])
    d = np.atleast_2d(d)[0]
    e = np.exp(d - d.max())
    return e / e.sum()


@app.get('/health')
def health():
    """De docker-compose va nguoi van hanh biet dich vu co song."""
    if 'pipeline' not in trang_thai:
        raise HTTPException(503, 'Mo hinh chua nap xong')
    return {'status': 'ok',
            'phien_ban_mo_hinh': trang_thai.get('phien_ban', 'khong_ro'),
            'ten_mo_hinh': trang_thai.get('ten_mo_hinh', 'khong_ro'),
            'so_nhan': len(trang_thai.get('nhan_labels', []))}


@app.post('/du-doan', response_model=PhanHoi)
def du_doan(req: YeuCau):
    if 'pipeline' not in trang_thai:
        raise HTTPException(503, 'Mo hinh chua nap xong')

    t0 = time.perf_counter()
    sach = lam_sach(req.mo_ta)
    so_tu = dem_tu(req.mo_ta)
    if not sach:
        raise HTTPException(422, 'Mo ta sau khi chuan hoa khong con noi dung nao.')

    xac_suat = _xac_suat(trang_thai['pipeline'], sach)
    nhan_labels = list(trang_thai['pipeline'].classes_)
    thu_tu = np.argsort(xac_suat)[::-1]
    top3 = [NhanDuDoan(nhan=str(nhan_labels[i]),
                       do_tin_cay=round(float(xac_suat[i]), 4))
            for i in thu_tu[:3]]

    # Nguong RIENG cho mo ta ngan — bien phap giam thieu thien lech da do.
    nguong = NGUONG_NGAN if so_tu < SO_TU_COI_LA_NGAN else NGUONG
    du_tin_cay = top3[0].do_tin_cay >= nguong

    # NHAT KY: chi ghi do dai, nhan va do tin cay.
    # TUYET DOI KHONG ghi noi dung mo ta — mo ta co the chua so dien thoai,
    # ten khach hay so serial.
    log.info('so_tu=%d nhan=%s do_tin_cay=%.4f nguong=%.2f thoi_gian_ms=%.1f',
             so_tu, top3[0].nhan, top3[0].do_tin_cay, nguong,
             (time.perf_counter() - t0) * 1000)

    return PhanHoi(
        nhan_goi_y=top3[0].nhan if du_tin_cay else None,
        do_tin_cay=top3[0].do_tin_cay,
        top_3=top3,
        can_nguoi_xem_lai=not du_tin_cay,
        nguong_da_ap_dung=nguong,
        so_tu_mo_ta=so_tu,
        phien_ban_mo_hinh=str(trang_thai.get('phien_ban', 'khong_ro')),
    )
