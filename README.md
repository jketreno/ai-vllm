# ai-vllm

Docker Compose stack for running:

- `vllm-engine` on port `8000`
- `open-webui` on port `8080`

This project is set up to expose an OpenAI-compatible vLLM endpoint and connect Open WebUI to it.

## Requirements

- Docker with the Compose plugin
- NVIDIA Container Toolkit / GPU-enabled Docker runtime
- Access to a supported NVIDIA GPU
- Hugging Face model access for `openai/gpt-oss-120b`

## Files

- `docker-compose.yml`: service definitions for Open WebUI and vLLM
- `.env`: Open WebUI environment variables, including LDAP settings

## Services

### `vllm-engine`

- Image: `nvcr.io/nvidia/vllm:26.04-py3`
- Port: `8000`
- GPU access enabled with `gpus: all`
- Uses `ipc: host` and increased ulimits as recommended by vLLM
- Mounts the local Hugging Face cache from `~/.cache/huggingface`

### `open-webui`

- Image: `ghcr.io/open-webui/open-webui:main`
- Port: `8080`
- Persists data in the named Docker volume `open-webui`
- Loads configuration from `.env`
- Configured to talk to the local vLLM-backed OpenAI-compatible endpoint

## Start

Bring the stack up in the project directory:

```bash
docker compose up -d
```

Recreate a single service after config changes:

```bash
docker compose up -d --force-recreate open-webui
docker compose up -d --force-recreate vllm-engine
```

## Logs

Follow all logs:

```bash
docker compose logs -f
```

Follow one service:

```bash
docker compose logs -f open-webui
docker compose logs -f vllm-engine
```

## Endpoints

- Open WebUI: `http://localhost:8080`
- vLLM OpenAI-compatible API: `http://localhost:8000/v1`

## LDAP

Open WebUI LDAP settings live in `.env`.

Important details for this setup:

- Open WebUI reads `LDAP_*` variables, not `OWEBUI_LDAP_*`
- `ENABLE_LDAP=True` enables LDAP auth
- `LDAP_USE_TLS=False` is correct for plain LDAP on port `389`
- If your directory uses LDAPS, switch to port `636` and set `LDAP_USE_TLS=True`

## Validate Configuration

Render the effective Compose config:

```bash
docker compose config
```

Check that Open WebUI received the LDAP env vars:

```bash
docker exec open-webui sh -lc 'env | grep "^ENABLE_LDAP\|^LDAP_" | sort'
```

## Stop

```bash
docker compose down
```