FROM python:3.11-slim

WORKDIR /code

# System deps needed by scikit-learn / xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
# artifacts/ لازم تكون موجودة جنب الـ Dockerfile قبل الـ build (ناتجة من train.py على Kaggle)
COPY artifacts ./artifacts

ENV ARTIFACTS_DIR=artifacts
ENV STATIC_DIR=static
# Railway/Render بيحقنوا PORT تلقائي؛ 8000 قيمة افتراضية للتشغيل المحلي
ENV PORT=8000

# مستخدم غير root — أساسي لأي نشر Production
RUN useradd --create-home appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# shell form عشان ${PORT} يتقرأ وقت التشغيل (Railway/Render بيغيروه)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2 --proxy-headers --forwarded-allow-ips="*"
