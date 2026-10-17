"""Lam sach van ban va vector hoa.

Hai quyet dinh thiet ke quan trong:
  1. GIU DAU tieng Viet. Bo dau lam mat nghia va gop cac tu khac nhau thanh mot:
     'sac' va 'sach' deu thanh 'sach'. Chi chuan hoa chu thuong.
  2. Thay thong tin dinh danh bang THE. So dien thoai, so serial va dia chi thu
     dien tu khong giup phan loai nhom loi, nhung neu de lai thi mo hinh co the
     hoc theo chung (qua khop) va nhat ky dich vu co the ghi lai chung.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Cac ky tu tieng Viet duoc giu lai
CHU_VIET = ('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ'
            'ìíỉĩịòóỏõọôồốổỗộơờớởỡợ'
            'ùúủũụưừứửữựỳýỷỹỵđ')

_RE_SDT    = re.compile(r'\b0\d{9}\b')
_RE_SERIAL = re.compile(r'\bSN-?\d+\b', re.IGNORECASE)
_RE_EMAIL  = re.compile(r'\S+@\S+')
_RE_KHAC   = re.compile(rf'[^\w\s{CHU_VIET}<>]')
_RE_TRANG  = re.compile(r'\s+')


def lam_sach(s) -> str:
    """Chuan hoa mot mo ta su co. GIU dau tieng Viet."""
    if s is None or s != s:            # None hoac NaN
        return ''
    s = str(s).lower()
    s = _RE_SDT.sub(' <sdt> ', s)
    s = _RE_SERIAL.sub(' <serial> ', s)
    s = _RE_EMAIL.sub(' <email> ', s)
    s = _RE_KHAC.sub(' ', s)
    return _RE_TRANG.sub(' ', s).strip()


def tao_vectorizer(cfg_tfidf: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=tuple(cfg_tfidf.get('ngram_range', (1, 2))),
        min_df=cfg_tfidf.get('min_df', 2),
        max_features=cfg_tfidf.get('max_features', 20000),
        lowercase=False,               # da ha chu thuong o lam_sach
    )


def dem_tu(s: str) -> int:
    """So tu cua mot mo ta — dung de phan nhom MO TA NGAN khi do thien lech."""
    return len(lam_sach(s).split())
