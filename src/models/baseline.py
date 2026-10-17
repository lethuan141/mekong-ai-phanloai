"""HAI mo hinh co so BAT BUOC.

Khong co mo hinh co so thi con so cua ta khong co y nghia: F1 macro 0,76 co the
la rat tot hoac rat te, tuy vao viec doan bua cho ra bao nhieu.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.text import tao_vectorizer


def co_so_nhan_da_so(hat_giong: int) -> DummyClassifier:
    """MUC SAN: luon doan nhan xuat hien nhieu nhat.

    Mo hinh cua ta PHAI cao hon con so nay, neu khong thi no khong hoc duoc gi.
    """
    return DummyClassifier(strategy='most_frequent', random_state=hat_giong)


def co_so_tfidf_logistic(cfg: dict, hat_giong: int) -> Pipeline:
    """MUC CHUAN: TF-IDF + hoi quy logistic.

    Rat kho vuot qua bang mo hinh phuc tap tren tap du lieu vai nghin dong —
    hay do truoc khi tin vao mo hinh lon.
    """
    return Pipeline([
        ('tfidf', tao_vectorizer(cfg.get('tfidf', {}))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                   random_state=hat_giong)),
    ])
