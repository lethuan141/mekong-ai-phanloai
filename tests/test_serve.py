"""Test dich vu du doan. Yeu cau da chay huan luyen truoc de co artifacts."""
import os
import pathlib

import pytest
from fastapi.testclient import TestClient

GOC = pathlib.Path(__file__).resolve().parent.parent
MODEL = GOC / 'artifacts' / 'model_v1.joblib'

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason='Chua co artifacts/model_v1.joblib. Chay: '
           'python -m src.models.train --config configs/train.yaml')


@pytest.fixture(scope='module')
def client():
    os.environ.setdefault('MODEL_PATH', str(MODEL))
    from src.serve.app import app
    with TestClient(app) as c:
        yield c


# ---------------- LOAI 1: LUONG DUNG ----------------
def test_SV01_health_tra_ve_phien_ban_mo_hinh(client):
    r = client.get('/health')
    assert r.status_code == 200
    b = r.json()
    assert b['status'] == 'ok'
    assert b['phien_ban_mo_hinh'] != 'khong_ro'
    assert b['so_nhan'] == 6


def test_SV02_du_doan_tra_ve_TOP_3_kem_do_tin_cay(client):
    r = client.post('/du-doan', json={
        'mo_ta': 'khách báo cổng sạc bị lỏng phải giữ mới vào điện'})
    assert r.status_code == 200
    b = r.json()
    assert len(b['top_3']) == 3
    # Top 3 phai sap xep giam dan theo do tin cay
    dt = [x['do_tin_cay'] for x in b['top_3']]
    assert dt == sorted(dt, reverse=True)
    assert all(0.0 <= x <= 1.0 for x in dt)


def test_SV03_moi_phan_hoi_deu_co_PHIEN_BAN_MO_HINH(client):
    """Ba thang sau, khi co nguoi bao 'he thong goi y sai', can biet luc do
    dang chay mo hinh nao. Day la nguyen tac TRUY VET DUOC."""
    b = client.post('/du-doan', json={'mo_ta': 'màn hình bị nứt góc dưới máy'}).json()
    assert b['phien_ban_mo_hinh'] != 'khong_ro'


# ---------------- LOAI 2: NGUONG TIN CAY VA CO XEM LAI ----------------
def test_SV04_mo_ta_NGAN_ap_nguong_CAO_HON(client):
    """Bien phap giam thieu thien lech: mo ta ngan dung nguong cao hon."""
    ngan = client.post('/du-doan', json={'mo_ta': 'màn hình sọc dọc'}).json()
    dai = client.post('/du-doan', json={
        'mo_ta': 'khách hàng phản ánh màn hình sọc dọc nhiều đường '
                 'xảy ra sau khi rơi đã thử khởi động lại không được'}).json()
    assert ngan['so_tu_mo_ta'] < 10
    assert dai['so_tu_mo_ta'] >= 10
    assert ngan['nguong_da_ap_dung'] > dai['nguong_da_ap_dung']


def test_SV05_duoi_nguong_thi_KHONG_dua_ra_nhan_goi_y(client):
    """Mo hinh khong chac thi noi ra la khong chac, khong gia vo chac chan."""
    b = client.post('/du-doan', json={
        'mo_ta': 'máy không lên nguồn cắm sạc không lên gì'}).json()
    if b['can_nguoi_xem_lai']:
        assert b['nhan_goi_y'] is None
        assert b['do_tin_cay'] < b['nguong_da_ap_dung']
    else:
        assert b['nhan_goi_y'] is not None
        assert b['do_tin_cay'] >= b['nguong_da_ap_dung']


# ---------------- LOAI 3: DU LIEU SAI ----------------
@pytest.mark.parametrize('mo_ta', ['', 'abc', '   ', 'x' * 5000])
def test_SV06_du_lieu_vao_khong_hop_le_tra_ve_422(client, mo_ta):
    """Tu choi dung cach, KHONG lam sap dich vu."""
    assert client.post('/du-doan', json={'mo_ta': mo_ta}).status_code == 422


def test_SV07_thieu_truong_bat_buoc_tra_ve_422(client):
    assert client.post('/du-doan', json={}).status_code == 422


def test_SV08_mo_ta_chi_gom_dau_cham_cau_tra_ve_422(client):
    """Du dai hon 5 ky tu nhung sau khi lam sach thi khong con noi dung nao."""
    assert client.post('/du-doan', json={'mo_ta': '!!!!! ????? .....'}).status_code == 422


def test_SV09_thong_bao_loi_khong_lo_duong_dan_tep_hay_phien_ban_thu_vien(client):
    r = client.post('/du-doan', json={'mo_ta': 'abc'})
    than = r.text.lower()
    for lo in ('traceback', 'site-packages', '/app/', 'sklearn', 'joblib'):
        assert lo not in than, f'thong bao loi lo thong tin noi bo: {lo}'
