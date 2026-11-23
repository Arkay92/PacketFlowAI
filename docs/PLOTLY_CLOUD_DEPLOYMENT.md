# PacketFlowAI — Plotly Cloud deployment

This folder is a self-contained deployment bundle for the PacketFlowAI Signal Room. It includes the Dash visual layer, read-only operational API, CSS assets, and the PacketFlowAI source modules required by the dashboard.

## Plotly Cloud / Dash Enterprise

1. Upload the **contents** of this folder.
2. Select Python 3.11 or newer.
3. Use `app.py` as the application file and `server` as the WSGI variable (`app:server`).
4. The cloud WSGI app captures real inbound HTTP/Dash traffic into an in-memory SQLite database when its application image is read-only. Visitor addresses are pseudonymised immediately and raw addresses are never stored. Set `PACKETFLOWAI_EVENT_DATABASE` to a writable SQLite path only if telemetry must survive a worker restart.

The platform installs `requirements.txt` automatically. The health endpoint is `/health`, and Prometheus-format metrics remain available at `/metrics/prometheus`.

## Local package smoke test

```bash
python app.py
```

Then open `http://127.0.0.1:8050`.

## Rebuild this folder

Run the packager from the repository root:

```bash
python scripts/prepare_cloud.py
```

The script always rebuilds this directory from the current source tree so cloud code cannot silently drift from the main application.
