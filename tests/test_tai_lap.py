"""Test TINH TAI LAP — hai lan chay cung hat giong phai cho ket qua giong het."""
import json
import pathlib
import shutil
import tempfile

import yaml
import pytest

from src.models import train as train_mod
from src import config

GOC = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def cfg_tam():
    """Cau hinh tam, ghi dau ra vao thu muc tam de khong ghi de artifacts that."""
    goc_cfg = config.nap()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='tai_lap_'))
    cfg = dict(goc_cfg)
    cfg['duong_dan_mo_hinh'] = str(tmp / 'model.joblib')
    cfg['duong_dan_bao_cao'] = str(tmp / 'metrics.json')
    p = tmp / 'train.yaml'
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    yield p, tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_TL01_hai_lan_huan_luyen_cho_bao_cao_giong_het(cfg_tam):
    p, tmp = cfg_tam
    train_mod.chay(str(p))
    lan1 = json.loads((tmp / 'metrics.json').read_text(encoding='utf-8'))
    train_mod.chay(str(p))
    lan2 = json.loads((tmp / 'metrics.json').read_text(encoding='utf-8'))
    assert lan1 == lan2, 'hai lan chay cho ket qua khac nhau -> chua tai lap duoc'


def test_TL02_bao_cao_co_DU_DIEU_KIEN_DO(cfg_tam):
    """Mot con so khong co dieu kien do la con so khong kiem chung duoc."""
    p, tmp = cfg_tam
    bao_cao = train_mod.chay(str(p))
    dk = bao_cao['dieu_kien_do']
    for truong in ('hat_giong', 'nguon_du_lieu', 'so_dong_du_lieu',
                   'so_mau_train', 'so_mau_val', 'so_mau_test',
                   'phan_bo_nhan', 'tfidf'):
        assert truong in dk, f'bao cao thieu dieu kien do: {truong}'


def test_TL03_bao_cao_co_HAI_mo_hinh_co_so(cfg_tam):
    """Khong co mo hinh co so thi con so cua ta khong co y nghia."""
    p, _ = cfg_tam
    bao_cao = train_mod.chay(str(p))
    co_so = bao_cao['mo_hinh_co_so']
    assert 'nhan_da_so' in co_so and 'tfidf_logistic' in co_so
    # Mo hinh da chon PHAI tot hon muc san
    assert (bao_cao['chi_so_tren_tap_test']['f1_macro']
            > co_so['nhan_da_so']['f1_macro_val']), \
        'mo hinh khong tot hon muc san — no khong hoc duoc gi'


def test_TL04_bao_cao_co_chi_so_TACH_THEO_NHAN(cfg_tam):
    """Chi so tong the che giau viec mo hinh bo roi mot nhan it mau."""
    p, _ = cfg_tam
    t = train_mod.chay(str(p))['chi_so_tren_tap_test']
    assert 'f1_theo_nhan' in t and 'recall_theo_nhan' in t
    assert len(t['f1_theo_nhan']) == len(t['nhan_labels'])
    assert 'confusion_matrix' in t


def test_TL05_bao_cao_co_BA_KET_LUAN_tu_confusion_matrix(cfg_tam):
    """Mot bang so dan vao bao cao ma khong co ket luan thi gan nhu vo dung."""
    p, _ = cfg_tam
    k = train_mod.chay(str(p))['ba_ket_luan_tu_confusion_matrix']
    assert k['cap_nham_nhieu_nhat']['cap'] is not None
    assert k['nhan_recall_thap_nhat']['nhan'] is not None
    assert k['nhan_hut_du_doan_sai_nhieu_nhat']['nhan'] is not None


def test_TL06_bao_cao_co_DO_THIEN_LECH_theo_it_nhat_hai_cach_nhom(cfg_tam):
    p, _ = cfg_tam
    tl = train_mod.chay(str(p))['thien_lech']
    assert len(tl['cac_cach_nhom']) >= 2, \
        'phai do thien lech theo it nhat hai cach nhom'
    for t in tl['cac_cach_nhom']:
        assert 'theo_nhom' in t and len(t['theo_nhom']) >= 2


def test_TL07_tap_test_chi_duoc_do_MOT_LAN(cfg_tam):
    p, _ = cfg_tam
    assert train_mod.chay(str(p))['so_lan_do_tap_test'] == 1


def test_TL08_bo_dem_tap_test_nem_loi_khi_do_lan_thu_hai():
    """Bien nguyen tac 'tap test dung mot lan' thanh mot phep kiem tu dong."""
    bo_dem = train_mod.BoDemTapTest()
    bo_dem.dung()
    with pytest.raises(train_mod.TapTestDaDung):
        bo_dem.dung()


def test_TL09_cau_hinh_thieu_hat_giong_bi_TU_CHOI(tmp_path):
    """Khong co hat giong thi ket qua khong tai lap duoc -> phai tu choi ngay."""
    p = tmp_path / 'xau.yaml'
    p.write_text('cot_van_ban: issue_desc\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hat_giong'):
        config.nap(str(p))
