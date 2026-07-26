# Server Hosting Security Checklist (Single-Tenant VPS)

When deploying a medical AI system to a single-tenant VPS (Virtual Private Server) like AWS EC2, DigitalOcean Droplet, or Hetzner, securing the Linux environment is a strict HIPAA requirement. 

This document serves as a checklist for locking down the server before opening the Twilio Voice WebSockets to the public internet.

---

## 1. Network Security (Firewall & Ports)
By default, VPS providers expose all ports. You must explicitly block everything except web traffic and SSH.

*   **[ ] Enable UFW (Uncomplicated Firewall) or AWS Security Groups.**
*   **[ ] OPEN Port `443` (HTTPS):** Required for Twilio WebSockets and the Dashboard.
*   **[ ] OPEN Port `80` (HTTP):** Only used to auto-redirect traffic to HTTPS.
*   **[ ] OPEN Port `22` (SSH):** Only allow SSH from your specific IP address if possible.
*   **[ ] BLOCKED Port `5432` (PostgreSQL):** The database MUST NOT be accessible from the public internet. Only internal Docker containers should connect to it.
*   **[ ] BLOCKED Port `8000` / `3000`:** Do not expose the raw FastAPI or Next.js ports directly. Force all traffic through the Reverse Proxy.

---

## 2. Docker Network Isolation
Instead of running applications directly on the host OS, deploy them via `docker-compose`.

*   **[ ] Create a Private Docker Network:** Ensure `docker-compose.yml` places the Python Backend, Next.js Frontend, and PostgreSQL on a shared internal network (e.g., `clinic_internal_net`).
*   **[ ] Do not map internal ports to the host:** For PostgreSQL, do not use `ports: ["5432:5432"]` in production. Only expose it to the internal Docker network so the Python backend can access it via `postgres:5432`.

---

## 3. Reverse Proxy & SSL Encryption (Caddy / Nginx)
Twilio will **refuse** to stream microphone audio over a plain `ws://` WebSocket. It requires an encrypted `wss://` (Secure WebSocket) connection.

*   **[ ] Deploy a Reverse Proxy (e.g., Caddy or Nginx):** This sits at the very edge of your server on ports 80/443.
*   **[ ] Auto-SSL:** Use Let's Encrypt to automatically generate and renew SSL certificates for your domain.
*   **[ ] Route Traffic:** The proxy will receive the HTTPS Twilio request, decrypt it, and securely pass it to your internal FastAPI container (port 8000).

---

## 4. Application-Level Defenses
Even if the server is locked down, the code itself must be resilient against injection attacks.

*   **[ ] Twilio Request Validation:** Implement the `Twilio Request Validator` in FastAPI (`api/routes.py`). This verifies the `X-Twilio-Signature` header using your secret `TWILIO_AUTH_TOKEN`, guaranteeing that only Twilio can hit your endpoints.
*   **[ ] `LANGGRAPH_STRICT_MSGPACK=true`:** Because LangGraph serializes the entire conversation state into PostgreSQL, you must set this environment variable in your Python container. It forces LangGraph to only deserialize safe data types (strings, ints, lists) and prevents malicious Python code injection from being executed when a state is loaded.
*   **[ ] Environment Variables:** Never hardcode secrets. Ensure `.env` files are in `.gitignore` and securely injected into the VPS via a CI/CD pipeline or secure vault.

---

## Summary Architecture

```text
[The Public Internet (Twilio)] 
         │ (Encrypted HTTPS / WSS)
         ▼
[Port 443 - Reverse Proxy (Nginx/Caddy)]
         │ (Internal Docker Network)
         ├──► [Port 3000 - Next.js Dashboard]
         └──► [Port 8000 - FastAPI Voice Server]
                   │
                   └──► [Internal Port 5432 - PostgreSQL DB]
```
