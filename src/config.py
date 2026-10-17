"""Nap cau hinh tu tep YAML. Khong ghi cung sieu tham so nao trong ma nguon."""
import pathlib
import yaml

GOC = pathlib.Path(__file__).resolve().parent.parent


def nap(duong_dan: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(duong_dan) if duong_dan else GOC / 'configs' / 'train.yaml'
    if not p.is_absolute():
        p = GOC / p
    with open(p, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'hat_giong' not in cfg:
        raise ValueError('Cau hinh phai co hat_giong — nếu không thì kết quả '
                         'không tái lập được.')
    return cfg


def duong_dan(cfg_val: str) -> pathlib.Path:
    p = pathlib.Path(cfg_val)
    return p if p.is_absolute() else GOC / p
