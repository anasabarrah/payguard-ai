FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home app && mkdir -p data && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 8501
# One process serves the dashboard, the demo shop (/shop/) and the PayPal webhook
# (/webhooks/paypal). Render sets $PORT; default to Streamlit's port when run locally.
CMD streamlit run server.py --server.port ${PORT:-8501} --server.address 0.0.0.0
