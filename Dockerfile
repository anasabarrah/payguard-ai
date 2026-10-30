FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 nginx gettext-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /app/start.sh
RUN useradd --create-home app && mkdir -p data && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 10000
ENV PORT=10000
CMD ["/app/start.sh"]
