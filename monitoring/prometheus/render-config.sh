#!/bin/sh
set -eu

case "${COMFYUI_METRICS_TARGET:-}" in
  ""|*[!A-Za-z0-9._:-]*)
    echo "COMFYUI_METRICS_TARGET must be a hostname:port value" >&2
    exit 1
    ;;
esac

sed "s|__COMFYUI_METRICS_TARGET__|${COMFYUI_METRICS_TARGET}|g" \
  /etc/prometheus/prometheus.yml.template \
  > /tmp/prometheus.yml

exec /bin/prometheus "$@"
