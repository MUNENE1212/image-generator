# Deploy on a VPS

Two paths, both produce the same end state — Streamlit on `https://your-domain`
with persistent storage and auto-renewing TLS:

- **A. Docker + Caddy** (recommended) — `docker compose up -d`, Caddy fetches a
  cert automatically.
- **B. Bare-metal + systemd + nginx** — for users who'd rather not run Docker.

## Server specs

The app itself is light because all GPU compute happens on Replicate / Fal.

| Resource | Minimum | Comfortable |
|---|---|---|
| vCPUs | 1 | 2 |
| RAM | 1 GB | 2 GB |
| Disk | 10 GB | 20 GB |
| OS | Debian 12 / Ubuntu 22.04+ | Same |

If you bake the **eval extra** into the image (PyTorch + insightface), bump
RAM to 4 GB and disk to 30 GB. Without the eval extra, the metric badges in
the UI just don't render — everything else works.

Cheapest providers as of 2026: **Hetzner CX11 (~$5/mo)**, Vultr, Linode,
DigitalOcean. Hetzner's image gallery has Docker pre-installed which saves a
step.

## DNS

Point an A record for your domain (e.g. `imggen.example.com`) at the VPS's
IPv4 address **before** starting the stack. Caddy / certbot do an HTTP-01
challenge that needs the DNS to already resolve.

---

## Path A — Docker + Caddy

### One-time setup

```bash
# On the VPS, as root or with sudo:
apt-get update && apt-get install -y docker.io docker-compose-plugin git curl
git clone https://github.com/<your-handle>/image-generator /opt/image-generator
cd /opt/image-generator

# Secrets + domain
cp .env.example .env
nano .env   # set REPLICATE_API_TOKEN, FAL_KEY, REPLICATE_DESTINATION,
            # and DOMAIN=imggen.example.com

# Bring it up
docker compose up -d --build
docker compose logs -f caddy   # watch the cert get issued (~30s)
```

After Caddy logs `certificate obtained successfully`, visit
`https://imggen.example.com`.

### Updates

```bash
cd /opt/image-generator
git pull
docker compose up -d --build
```

The `imggen-data` named volume survives rebuilds — your `runs.duckdb` and
all generated images persist across deploys.

### Backups

```bash
# Create a tarball of the data volume.
docker run --rm \
  -v image-generator_imggen-data:/data \
  -v "$PWD":/backup \
  alpine tar czf /backup/imggen-$(date +%F).tar.gz -C /data .
```

Schedule via cron / systemd-timer. Push to S3 / R2 / Backblaze B2 with `rclone`.

---

## Path B — Bare-metal + systemd + nginx

### One-time setup

```bash
# Create a dedicated user and clone there.
sudo adduser --system --group --home /opt/image-generator imggen
sudo -u imggen git clone https://github.com/<your-handle>/image-generator /opt/image-generator
cd /opt/image-generator

# Install uv + project deps.
sudo -u imggen bash -c '
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv sync
'

# Secrets
sudo cp .env.example .env
sudo chown imggen:imggen .env
sudo chmod 600 .env
sudoedit .env   # fill in tokens

# systemd unit
sudo cp deploy/imggen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now imggen
sudo systemctl status imggen   # confirm "active (running)"

# nginx + TLS
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/imggen
sudoedit /etc/nginx/sites-available/imggen   # replace imggen.example.com
sudo ln -s /etc/nginx/sites-available/imggen /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d imggen.example.com
```

### Updates

```bash
sudo -u imggen bash -c '
  cd /opt/image-generator
  git pull
  ~/.local/bin/uv sync
'
sudo systemctl restart imggen
```

---

## Hardening (both paths)

1. **Firewall** — allow only 22, 80, 443:
   ```bash
   ufw allow 22 && ufw allow 80 && ufw allow 443 && ufw enable
   ```
2. **SSH** — disable password auth, key-only.
3. **Spend cap** — set `IMGGEN_DAILY_SPEND_CAP_USD` in `.env`. The app
   reads it; you'll need to wire enforcement into `services/generation.py`
   if you expect public traffic (currently advisory-only).
4. **BYO API key for public deploys** — if anyone on the internet can hit
   the app with your `REPLICATE_API_TOKEN`, they can drain your credits.
   Either gate access (HTTP basic auth via Caddy / nginx) or modify the
   sidebar to take a user-supplied key textbox.

   Caddy basic auth one-liner (drops into the Caddyfile):
   ```
   {$DOMAIN} {
       basicauth {
           you $2a$14$<bcrypt-hash-from-`caddy hash-password`>
       }
       reverse_proxy app:8501
   }
   ```

## Cost in practice

A single user generating ~50 images/day at ~$0.005 each = **$7.50/mo on
Replicate** + **$5/mo VPS** = **~$12.50/mo total**. The VPS is the smaller
line item; image generation cost dominates if you actually use the thing.

## Where the data lives

| Item | Docker path | Bare-metal path |
|---|---|---|
| DuckDB | `imggen-data` volume → `/app/data/runs/runs.duckdb` | `/opt/image-generator/data/runs/runs.duckdb` |
| Selfies | `imggen-data` volume → `/app/data/selfies/` | `/opt/image-generator/data/selfies/` |
| Outputs | `imggen-data` volume → `/app/data/outputs/` | `/opt/image-generator/data/outputs/` |
| Trained LoRAs (artifacts) | Stored on Replicate (URL recorded in `trainings` table) | Same |
| Caddy / Let's Encrypt certs | `caddy-data` named volume | `/etc/letsencrypt/` |

## Troubleshooting

- **`certificate obtain failed`** — DNS isn't pointing at the VPS yet, or
  another process is bound to port 80. `dig $DOMAIN` should return your
  server IP; `ss -tlnp | grep :80` should be empty before bringing the stack
  up.
- **`Invalid request: strategy=instant_id requires a selfie`** — the user
  picked an identity-preserving strategy without uploading. Expected.
- **Generations succeed but Gallery is empty after a redeploy** — you used
  bind mounts instead of named volumes and the path on disk wasn't
  preserved. Switch to the named-volume pattern in `docker-compose.yml`
  (the default).
