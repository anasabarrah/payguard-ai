#!/bin/sh
set -eu
export PAYGUARD_DB="${PAYGUARD_DB:-/app/data/ledger.db}"
mkdir -p /tmp/nginx-client /tmp/nginx-proxy /tmp/nginx-fastcgi /tmp/nginx-uwsgi /tmp/nginx-scgi
uvicorn shop_api:app --host 127.0.0.1 --port 8000 &
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 &
envsubst '${PORT}' < /app/nginx.conf.template > /tmp/nginx.conf
exec nginx -c /tmp/nginx.conf -g 'daemon off;'
