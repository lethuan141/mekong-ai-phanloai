"""Test ham lam sach van ban — ham thuan nen test rat re va rat on dinh."""
import pytest
from src.features.text import lam_sach, dem_tu


# ---------------- LOAI 1: LUONG DUNG ----------------
def test_TX01_giu_dau_tieng_viet():
    """GIU dau la quyet dinh thiet ke: bo dau gop 'sac' va 'sach' thanh mot."""
    ra = lam_sach('Màn hình bị nứt góc dưới')
    assert 'màn hình' in ra
    assert 'nứt' in ra


def test_TX02_ha_chu_thuong():
    assert lam_sach('MÀN HÌNH SỌC DỌC') == lam_sach('màn hình sọc dọc')


# ---------------- LOAI 2: THAY THONG TIN DINH DANH BANG THE ----------------
def test_TX03_thay_so_dien_thoai_bang_the():
    ra = lam_sach('khách 0912345678 báo máy không lên nguồn')
    assert '<sdt>' in ra
    assert '0912345678' not in ra


def test_TX04_thay_so_serial_bang_the():
    for tho in ('SN-10293', 'sn10293', 'SN10293'):
        ra = lam_sach(f'thiết bị {tho} màn hình vỡ')
        assert '<serial>' in ra, f'khong thay the duoc {tho}'
        assert '10293' not in ra


def test_TX05_thay_dia_chi_thu_dien_tu_bang_the():
    ra = lam_sach('liên hệ lan@example.com để báo kết quả')
    assert '<email>' in ra
    assert 'example.com' not in ra


# ---------------- LOAI 3: GIA TRI BIEN VA DU LIEU SAI ----------------
@pytest.mark.parametrize('tho', [None, '', '   ', float('nan')])
def test_TX06_gia_tri_rong_tra_ve_chuoi_rong(tho):
    assert lam_sach(tho) == ''


def test_TX07_chuoi_chi_co_dau_cham_cau_tra_ve_chuoi_rong():
    assert lam_sach('...!!! ???') == ''


def test_TX08_gop_khoang_trang_va_bo_khoang_trang_hai_dau():
    assert lam_sach('   màn    hình   vỡ   ') == 'màn hình vỡ'


def test_TX09_dem_tu_dem_tren_van_ban_da_lam_sach():
    assert dem_tu('màn hình vỡ') == 3
    assert dem_tu('khách 0912345678 báo vỡ') == 4     # so dien thoai -> mot the
    assert dem_tu('') == 0
    assert dem_tu(None) == 0
