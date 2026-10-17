"""Test CHONG RO RI DU LIEU — test quan trong nhat cua track AI.

Ro ri du lieu la khi mot bo bien doi hoc tham so (vectorizer, bo chuan hoa, bo
dien gia tri thieu) duoc fit tren du lieu bao gom ca tap test. Khi do ket qua do
tren tap test cao hon nang luc that, va con so bao cao khong con y nghia.

Day khong chi la loi ky thuat: bao cao mot con so cao hon nang luc that la sai
lech so lieu, va neu ai do tin con so do roi trien khai mo hinh thi hau qua
thuoc ve nguoi bao cao.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data import load, split
from src.features.text import lam_sach, tao_vectorizer


@pytest.fixture(scope='module')
def cfg():
    return config.nap()


@pytest.fixture(scope='module')
def cac_tap(cfg):
    df = load.nap(config.duong_dan(cfg['duong_dan_du_lieu']),
                  cfg['cot_van_ban'], cfg['cot_nhan'])
    train, val, test = split.chia(df, cfg['cot_nhan'], int(cfg['hat_giong']),
                                  float(cfg['ti_le_test']), float(cfg['ti_le_val']))
    for t in (train, val, test):
        t['_sach'] = t[cfg['cot_van_ban']].map(lam_sach)
    return train, val, test


def test_RR01_ba_tap_khong_chong_nhau(cac_tap, cfg):
    """Dieu kien tien quyet: mot mau khong duoc nam o hai tap."""
    train, val, test = cac_tap
    a = set(train['desc_id'])
    b = set(val['desc_id'])
    c = set(test['desc_id'])
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(a) + len(b) + len(c) == len(a | b | c)


def test_RR02_ti_le_nhan_duoc_giu_o_ca_ba_tap(cac_tap, cfg):
    """Chia PHAN TANG: moi tap phai co du sau nhan voi ti le tuong tu.

    Neu khong phan tang, mot tap co the khong co mau nao cua nhan it gap va
    chi so cua tap do tro nen vo nghia.
    """
    train, val, test = cac_tap
    cot = cfg['cot_nhan']
    p_train = train[cot].value_counts(normalize=True).sort_index()
    for tap, ten in ((val, 'val'), (test, 'test')):
        p = tap[cot].value_counts(normalize=True).sort_index()
        assert set(p.index) == set(p_train.index), f'{ten} thieu nhan'
        # Sai lech ti le khong duoc qua 6 diem phan tram
        assert (p - p_train).abs().max() < 0.06, f'{ten} lech ti le nhan qua nhieu'


SENTINEL = 'zzmatukhongcothat'


def test_RR03_phep_kiem_ro_ri_DUOC_KIEM_CHUNG_BANG_HAI_CHIEU(cac_tap, cfg):
    """Test CHONG RO RI chinh — dung THI NGHIEM CO DOI CHUNG.

    Mot phep kiem chi dang tin khi no BIET PHAT HIEN loi. Vi vay test nay lam
    hai chieu tren cung mot du lieu:

      chieu am  (dung cach)  fit vectorizer CHI tren train  -> tu sentinel chi
                             co o tap test PHAI KHONG nam trong tu vung;
      chieu duong (co y sai) fit vectorizer tren train+test -> tu sentinel PHAI
                             nam trong tu vung.

    Neu chieu duong khong that bai duoc thi phep kiem la VO NGHIA: no se bao
    "khong ro ri" ca khi da ro ri that. Do la loi BUG-05 cua repo nay: ban
    dau test chi co chieu am va no xanh ngay ca khi co tinh fit tren train+test.

    Sentinel duoc chen vao 4 dong cua tap test vi vectorizer dat min_df = 2:
    mot tu xuat hien duy nhat mot lan se bi loai khoi tu vung bat ke ro ri.
    """
    train, _val, test = cac_tap
    van_ban_train = list(train['_sach'])
    van_ban_test = list(test['_sach'])
    for i in range(4):
        van_ban_test[i] = van_ban_test[i] + ' ' + SENTINEL

    def tu_vung_don(du_lieu):
        vec = tao_vectorizer(cfg.get('tfidf', {}))
        vec.fit(du_lieu)
        return {t for t in vec.vocabulary_ if ' ' not in t}

    # Chieu am: lam dung -> khong duoc thay sentinel
    tv_train = tu_vung_don(van_ban_train)
    assert SENTINEL not in tv_train, (
        'Vectorizer fit tren train ma van biet tu chi co o tap test.')

    # Chieu duong: co y sai -> PHAI thay sentinel, chung minh phep kiem chay dung
    tv_ca_hai = tu_vung_don(van_ban_train + van_ban_test)
    assert SENTINEL in tv_ca_hai, (
        'Phep kiem khong phat hien duoc ro ri co y tao ra — phep kiem vo nghia.')


def test_RR03b_tu_vung_rieng_cua_tap_test_tren_du_lieu_that(cac_tap, cfg):
    """Phep kiem TU NHIEN: tu nao chi xuat hien o tap test thi khong duoc nam
    trong tu vung cua vectorizer fit tren train.

    Voi bo du lieu mau cua repo, phep kiem nay KHONG CHUNG MINH DUOC GI: bo mau
    duoc sinh tu mot kho cum tu dong (353 tu o train), nen tap test khong co tu
    nao rieng. Khi do test tu bo qua kem theo ly do, thay vi xanh mot cach gia.
    Doi sang du lieu that (co loi chinh ta, ma may, ten rieng) thi phep kiem nay
    moi co hieu luc va se chay.
    """
    train, _val, test = cac_tap
    tu_train = set()
    for s in train['_sach']:
        tu_train.update(s.split())
    tu_test = set()
    for s in test['_sach']:
        tu_test.update(s.split())
    chi_co_o_test = tu_test - tu_train

    if not chi_co_o_test:
        pytest.skip('Bo du lieu mau co tu vung DONG: tap test khong co tu rieng '
                    'nen phep kiem tu nhien nay khong chung minh duoc gi '
                    '(xem docs/bug-log.md, BUG-05). Test RR03 lo phan nay.')

    vec = tao_vectorizer(cfg.get('tfidf', {}))
    vec.fit(train['_sach'])
    tu_vung_don = {t for t in vec.vocabulary_ if ' ' not in t}
    ro_ri = tu_vung_don & chi_co_o_test
    assert not ro_ri, f'Ro ri tu vung tap test: {sorted(ro_ri)[:5]}'


def test_RR04_pipeline_bao_dam_vectorizer_chi_fit_tren_du_lieu_dua_vao(cac_tap, cfg):
    """Dung Pipeline la cach BAO DAM BANG CAU TRUC rang khong ro ri.

    Vectorizer nam trong pipeline nen no chi duoc fit tren tap dua vao .fit();
    khong the vo tinh fit tren toan bo du lieu.
    """
    train, _val, test = cac_tap
    pipe = Pipeline([
        ('tfidf', tao_vectorizer(cfg.get('tfidf', {}))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                   random_state=int(cfg['hat_giong']))),
    ])
    pipe.fit(train['_sach'], train[cfg['cot_nhan']])

    tu_vung_pipeline = set(pipe.named_steps['tfidf'].vocabulary_.keys())

    # Fit RIENG mot vectorizer chi tren train -> phai cho CUNG tu vung
    vec = tao_vectorizer(cfg.get('tfidf', {}))
    vec.fit(train['_sach'])
    assert tu_vung_pipeline == set(vec.vocabulary_.keys())

    # Va phai KHAC tu vung khi fit tren toan bo du lieu (bang chung la
    # pipeline khong nhin thay val/test)
    vec_all = tao_vectorizer(cfg.get('tfidf', {}))
    vec_all.fit(pd.concat([train['_sach'], test['_sach']]))
    assert tu_vung_pipeline != set(vec_all.vocabulary_.keys()), \
        'tu vung giong nhau — kiem lai xem tap test co du khac biet khong'


def test_RR05_lam_sach_khong_hoc_tham_so_nen_an_toan_o_moi_thu_tu(cac_tap, cfg):
    """lam_sach la ham THUAN: khong hoc gi tu du lieu, nen goi truoc hay sau
    khi chia deu cho cung ket qua. Day la ly do no duoc goi sau khi chia."""
    train, _val, test = cac_tap
    cot = cfg['cot_van_ban']
    mau = list(test['_sach'])[:20]
    lai = [lam_sach(s) for s in list(test[cot])[:20]]
    assert len(mau) == 20 and all(a == b for a, b in zip(mau, lai))

    # Va goi lam_sach TRUOC khi chia cho cung ket qua tren cung dong du lieu
    truoc = {int(r['desc_id']): lam_sach(r[cot]) for _, r in test.iterrows()}
    sau = {int(r['desc_id']): r['_sach'] for _, r in test.iterrows()}
    assert truoc == sau


def test_RR06_MO_HINH_DA_LUU_khong_chua_tu_vung_cua_val_va_test(cac_tap, cfg):
    """Kiem tren CHINH TEP MO HINH SE TRIEN KHAI, khong phai tren mot pipeline
    dung lai trong test.

    RR03 va RR04 dung pipeline do chinh test dung nen, nen chung chi chung minh
    rang CACH LAM la dung. Chung khong noi gi ve tep artifacts/model_v1.joblib
    — thu duy nhat that su duoc dem di phuc vu. Day la loi BUG-06: mot ro ri
    duoc chen thang vao src/models/train.py van khong lam do bat ky test nao.

    Phep kiem: tu vung cua vectorizer trong tep da luu phai TRUNG KHIT voi tu
    vung khi fit lai chi tren tap train (cung hat giong, cung cau hinh), va
    phai KHAC tu vung khi fit tren toan bo du lieu.
    """
    joblib = pytest.importorskip('joblib')
    duong = config.duong_dan('artifacts/model_v1.joblib')
    if not duong.exists():
        pytest.skip('Chua co tep mo hinh — chay `python -m src.models.train` truoc.')

    goi = joblib.load(duong)
    tu_vung_luu = set(goi['pipeline'].named_steps['tfidf'].vocabulary_)

    train, _val, test = cac_tap
    vec_train = tao_vectorizer(cfg.get('tfidf', {}))
    vec_train.fit(train['_sach'])
    assert tu_vung_luu == set(vec_train.vocabulary_), (
        'Tu vung trong tep mo hinh KHAC tu vung fit lai chi tren train: '
        f'luu {len(tu_vung_luu)} tu, fit lai {len(vec_train.vocabulary_)} tu. '
        'Rat co the mot buoc nao trong train.py da nhin thay val hoac test.')

    df_tat_ca = pd.concat([train['_sach'], _val['_sach'], test['_sach']])
    vec_tat_ca = tao_vectorizer(cfg.get('tfidf', {}))
    vec_tat_ca.fit(df_tat_ca)
    assert tu_vung_luu != set(vec_tat_ca.vocabulary_), (
        'Tu vung trong tep mo hinh TRUNG voi tu vung fit tren toan bo du lieu '
        '— dau hieu ro ri.')


def test_RR07_so_mau_train_ghi_trong_tep_mo_hinh_dung_bang_kich_thuoc_tap_train(cac_tap):
    """Tep mo hinh tu khai bao no duoc huan luyen tren bao nhieu mau. Neu con
    so do lon hon kich thuoc tap train thi mo hinh da an them du lieu."""
    joblib = pytest.importorskip('joblib')
    duong = config.duong_dan('artifacts/model_v1.joblib')
    if not duong.exists():
        pytest.skip('Chua co tep mo hinh.')
    goi = joblib.load(duong)
    train, _val, _test = cac_tap
    assert int(goi['so_mau_train']) == len(train)
