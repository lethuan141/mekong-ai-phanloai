FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Chay bang nguoi dung khong dac quyen
RUN useradd -m -u 10001 app

COPY src ./src
COPY configs ./configs
COPY data ./data

# HUAN LUYEN MO HINH NGAY TRONG LUC DUNG ANH.
# Nho vay goi dong goi TU DU: nguoi khac clone repo va chay 2 lenh la co
# dich vu du doan hoat dong, khong can tep mo hinh dinh kem.
# Danh doi: dung anh lau hon (~40 giay) va anh lon hon.
RUN mkdir -p artifacts reports \
 && python -m src.models.train --config configs/train.yaml \
 && chown -R app artifacts reports

USER app
ENV MODEL_PATH=artifacts/model_v1.joblib
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=25s --retries=6 \
  CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
