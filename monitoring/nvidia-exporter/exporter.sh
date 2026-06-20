#!/usr/bin/env bash
set -euo pipefail

METRICS_FILE=/tmp/nvidia_metrics.prom
PORT=9835
INTERVAL=10

collect() {
  local output
  output=$(nvidia-smi \
    --query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.max.sm,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits 2>/dev/null) || return

  while IFS=',' read -r name driver temp power sm_clock sm_max gpu_util mem_util; do
    name=$(printf '%s' "$name" | xargs)
    driver=$(printf '%s' "$driver" | xargs)
    temp=$(printf '%s' "$temp" | xargs)
    power=$(printf '%s' "$power" | xargs)
    sm_clock=$(printf '%s' "$sm_clock" | xargs)
    sm_max=$(printf '%s' "$sm_max" | xargs)
    gpu_util=$(printf '%s' "$gpu_util" | xargs)
    mem_util=$(printf '%s' "$mem_util" | xargs)

    local labels="gpu=\"${name}\",driver=\"${driver}\""

    {
      printf '# HELP nvidia_temperature_celsius GPU temperature in Celsius\n'
      printf '# TYPE nvidia_temperature_celsius gauge\n'
      printf 'nvidia_temperature_celsius{%s} %s\n' "$labels" "$temp"

      printf '# HELP nvidia_power_draw_watts GPU power draw in Watts\n'
      printf '# TYPE nvidia_power_draw_watts gauge\n'
      printf 'nvidia_power_draw_watts{%s} %s\n' "$labels" "$power"

      printf '# HELP nvidia_sm_clock_mhz GPU SM clock speed in MHz\n'
      printf '# TYPE nvidia_sm_clock_mhz gauge\n'
      printf 'nvidia_sm_clock_mhz{%s} %s\n' "$labels" "$sm_clock"

      printf '# HELP nvidia_sm_clock_max_mhz GPU maximum SM clock speed in MHz\n'
      printf '# TYPE nvidia_sm_clock_max_mhz gauge\n'
      printf 'nvidia_sm_clock_max_mhz{%s} %s\n' "$labels" "$sm_max"

      printf '# HELP nvidia_gpu_utilization_percent GPU compute utilization percent\n'
      printf '# TYPE nvidia_gpu_utilization_percent gauge\n'
      printf 'nvidia_gpu_utilization_percent{%s} %s\n' "$labels" "$gpu_util"

      printf '# HELP nvidia_memory_utilization_percent GPU memory bandwidth utilization percent\n'
      printf '# TYPE nvidia_memory_utilization_percent gauge\n'
      printf 'nvidia_memory_utilization_percent{%s} %s\n' "$labels" "$mem_util"
    } >"${METRICS_FILE}.tmp"

    mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
  done <<<"$output"
}

serve() {
  python3 - <<'PY'
import http.server, os, pathlib, time

PORT = int(os.environ.get("PORT", 9835))
METRICS_FILE = os.environ.get("METRICS_FILE", "/tmp/nvidia_metrics.prom")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = pathlib.Path(METRICS_FILE).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"metrics not yet collected\n")
    def log_message(self, *args):
        pass

http.server.HTTPServer(("", PORT), Handler).serve_forever()
PY
}

collect
serve &

while true; do
  sleep "$INTERVAL"
  collect
done
