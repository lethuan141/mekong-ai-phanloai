"""Huan luyen, chon mo hinh tren VALIDATION, do tap TEST dung MOT LAN.

Chay bang mot lenh:
    python -m src.models.train --config configs/train.yaml

Sinh ra:
    artifacts/model_v1.joblib   (vectorizer + mo hinh + sieu du lieu)
    reports/metrics.json        (moi con so, KEM DIEU KIEN DO)
"""
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from src import config
from src.data import load, split
from src.features.text import tao_vectorizer, lam_sach, dem_tu
from src.models import baseline
from src import evaluate

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
log = logging.getLogger('train')


class TapTestDaDung(RuntimeError):
    """Nem ra khi co gang do tap test lan thu hai."""


class BoDemTapTest:
    """Bao dam tap test chi duoc do DUNG MOT LAN trong mot lan chay.

    Moi lan xem ket qua tren tap test roi quay lai sua mo hinh, tap test mat gia
    tri vi no da anh huong toi lua chon cua ta. Bo dem nay bien nguyen tac do
    thanh mot phep kiem tu dong.
    """

    def __init__(self):
        self.so_lan = 0

    def dung(self):
        self.so_lan += 1
        if self.so_lan > 1:
            raise TapTestDaDung(
                'Tap test da duoc do roi. Moi lan do lai tren tap test roi sua '
                'mo hinh se lam con so bao cao cao hon nang luc that. Hay chon '
                'mo hinh tren tap VALIDATION.')


def dat_hat_giong(hat_giong: int) -> None:
    """Dat hat giong o MOI cho co ngau nhien."""
    random.seed(hat_giong)
    np.random.seed(hat_giong)


def cac_ung_vien(cfg: dict, hat_giong: int) -> dict:
    """Cac mo hinh THAT de so sanh. Moi mo hinh la mot Pipeline hoan chinh.

    Dung Pipeline la QUAN TRONG: vectorizer nam trong pipeline nen no chi duoc
    fit tren tap dua vao .fit(), khong bao gio nhin thay val hay test.
    """
    t = cfg.get('tfidf', {})
    return {
        'logistic': Pipeline([
            ('tfidf', tao_vectorizer(t)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                       random_state=hat_giong)),
        ]),
        'linear_svc': Pipeline([
            ('tfidf', tao_vectorizer(t)),
            ('clf', LinearSVC(class_weight='balanced', random_state=hat_giong)),
        ]),
        'complement_nb': Pipeline([
            ('tfidf', tao_vectorizer(t)),
            ('clf', ComplementNB()),
        ]),
    }


def chay(duong_dan_cfg: str | None = None) -> dict:
    cfg = config.nap(duong_dan_cfg)
    hat_giong = int(cfg['hat_giong'])
    dat_hat_giong(hat_giong)

    cot_vb, cot_nhan = cfg['cot_van_ban'], cfg['cot_nhan']

    # ---------- 1. NAP ----------
    df = load.nap(config.duong_dan(cfg['duong_dan_du_lieu']), cot_vb, cot_nhan)
    so_dong_bi_bo = int(df['_so_dong_bi_bo'].iloc[0]) if len(df) else 0
    log.info('nap %d dong (bo %d dong thieu nhan hoac thieu van ban)',
             len(df), so_dong_bi_bo)

    phan_bo = df[cot_nhan].value_counts()
    log.info('phan bo nhan: %s', phan_bo.to_dict())

    # ---------- 2. CHIA TRUOC ----------
    # Buoc nay phai dung TRUOC moi bien doi hoc tham so.
    train, val, test = split.chia(df, cot_nhan, hat_giong,
                                  float(cfg['ti_le_test']), float(cfg['ti_le_val']))
    log.info('chia du lieu: train=%d val=%d test=%d', len(train), len(val), len(test))

    # ---------- 3. LAM SACH (khong hoc tham so nen an toan o buoc nay) ----------
    for tap in (train, val, test):
        tap['_sach'] = tap[cot_vb].map(lam_sach)
        tap['_so_tu'] = tap[cot_vb].map(dem_tu)
        # Nhom dung de DO THIEN LECH theo do dai mo ta
        tap['_nhom_do_dai'] = np.where(tap['_so_tu'] < 10, 'ngan_duoi_10_tu',
                                       'dai_tu_10_tu')

    nhan_labels = sorted(df[cot_nhan].unique())

    # ---------- 4. HAI MO HINH CO SO ----------
    co_so = {}
    m0 = baseline.co_so_nhan_da_so(hat_giong)
    m0.fit(np.zeros((len(train), 1)), train[cot_nhan])
    co_so['nhan_da_so'] = {
        'f1_macro_val': round(float(f1_score(
            val[cot_nhan], m0.predict(np.zeros((len(val), 1))),
            average='macro', zero_division=0)), 4),
        'ghi_chu': 'MUC SAN — luon doan nhan xuat hien nhieu nhat',
    }
    m1 = baseline.co_so_tfidf_logistic(cfg, hat_giong)
    m1.fit(train['_sach'], train[cot_nhan])
    co_so['tfidf_logistic'] = {
        'f1_macro_val': round(float(f1_score(
            val[cot_nhan], m1.predict(val['_sach']),
            average='macro', zero_division=0)), 4),
        'ghi_chu': 'MUC CHUAN — TF-IDF + hoi quy logistic',
    }
    for ten, v in co_so.items():
        log.info('co so %-16s F1 macro (val) = %.4f', ten, v['f1_macro_val'])

    # ---------- 5. SO SANH CAC MO HINH THAT — CHON TREN VALIDATION ----------
    ket_qua_val = {}
    da_fit = {}
    for ten, mo_hinh in cac_ung_vien(cfg, hat_giong).items():
        mo_hinh.fit(train['_sach'], train[cot_nhan])
        f1 = f1_score(val[cot_nhan], mo_hinh.predict(val['_sach']),
                      average='macro', zero_division=0)
        ket_qua_val[ten] = round(float(f1), 4)
        da_fit[ten] = mo_hinh
        log.info('ung vien %-16s F1 macro (val) = %.4f', ten, f1)

    ten_tot_nhat = max(ket_qua_val, key=ket_qua_val.get)
    mo_hinh_tot_nhat = da_fit[ten_tot_nhat]
    log.info('chon mo hinh %s (tren tap VALIDATION)', ten_tot_nhat)

    # ---------- 6. DO TAP TEST — DUNG MOT LAN ----------
    bo_dem = BoDemTapTest()
    bo_dem.dung()
    y_test_pred = mo_hinh_tot_nhat.predict(test['_sach'])
    chi_so_test = evaluate.do_theo_nhan(test[cot_nhan], y_test_pred, nhan_labels)
    ket_luan_cm = evaluate.doc_confusion_matrix(
        chi_so_test['confusion_matrix'], nhan_labels)

    # ---------- 7. DO THIEN LECH THEO HAI CACH NHOM ----------
    thien_lech = [
        evaluate.do_thien_lech(test, test[cot_nhan], y_test_pred,
                               '_nhom_do_dai', nhan_labels, so_mau_toi_thieu=30),
    ]
    if 'center_id' in test.columns:
        thien_lech.append(evaluate.do_thien_lech(
            test, test[cot_nhan], y_test_pred, 'center_id', nhan_labels,
            so_mau_toi_thieu=30))
    nguong_tl = float(cfg.get('nguong_thien_lech', 0.15))
    vuot_nguong = [t for t in thien_lech
                   if t.get('chenh_lech', 0) > nguong_tl]

    # ---------- 8. GHI KET QUA — KEM DIEU KIEN DO ----------
    bao_cao = {
        # DIEU KIEN DO: mot con so khong co dieu kien do la con so khong
        # kiem chung duoc.
        'dieu_kien_do': {
            'hat_giong': hat_giong,
            'nguon_du_lieu': str(cfg['duong_dan_du_lieu']),
            'so_dong_du_lieu': int(len(df)),
            'so_dong_bi_bo_vi_thieu_nhan': so_dong_bi_bo,
            'so_mau_train': int(len(train)),
            'so_mau_val': int(len(val)),
            'so_mau_test': int(len(test)),
            'phan_bo_nhan': {k: int(v) for k, v in phan_bo.items()},
            'tfidf': cfg.get('tfidf', {}),
        },
        'mo_hinh_co_so': co_so,
        'so_sanh_tren_validation': ket_qua_val,
        'mo_hinh_da_chon': ten_tot_nhat,
        'chi_so_tren_tap_test': {
            k: v for k, v in chi_so_test.items() if k != 'bao_cao_chi_tiet'},
        'ba_ket_luan_tu_confusion_matrix': ket_luan_cm,
        'thien_lech': {
            'nguong_canh_bao': nguong_tl,
            'cac_cach_nhom': thien_lech,
            'so_cach_nhom_vuot_nguong': len(vuot_nguong),
        },
        'so_lan_do_tap_test': bo_dem.so_lan,
    }

    p_bc = config.duong_dan(cfg['duong_dan_bao_cao'])
    p_bc.parent.mkdir(parents=True, exist_ok=True)
    p_bc.write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding='utf-8')

    # ---------- 9. LUU MO HINH: vectorizer VA mo hinh CUNG MOT LUC ----------
    p_mh = config.duong_dan(cfg['duong_dan_mo_hinh'])
    p_mh.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'pipeline': mo_hinh_tot_nhat,      # da gom ca vectorizer
        'nhan_labels': nhan_labels,
        'ten_mo_hinh': ten_tot_nhat,
        'hat_giong': hat_giong,
        'phien_ban': 'v1.0',
        'nguon_du_lieu': str(cfg['duong_dan_du_lieu']),
        'so_mau_train': int(len(train)),
        # KHONG luu dau thoi gian: dau thoi gian se lam hai lan chay khac nhau
        # va pha vo phep thu tai lap duoc.
    }, p_mh)

    log.info('F1 macro (test) = %.4f | da ghi %s va %s',
             chi_so_test['f1_macro'], p_bc, p_mh)
    if vuot_nguong:
        for t in vuot_nguong:
            log.warning('THIEN LECH vuot nguong theo %s: chenh lech %.4f > %.2f '
                        '— PHAI neu trong Model card kem bien phap giam thieu',
                        t['cot_nhom'], t['chenh_lech'], nguong_tl)
    return bao_cao


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Huan luyen mo hinh phan loai su co')
    ap.add_argument('--config', default=None, help='Duong dan tep cau hinh YAML')
    a = ap.parse_args(argv)
    chay(a.config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
