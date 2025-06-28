# SSL Certificates

Place your SSL certificates in this directory:

- `cert.pem` - SSL certificate
- `key.pem` - Private key
- `dhparam.pem` - DH parameters (optional)

## Generating Self-Signed Certificates (Development Only)

```bash
# Generate private key
openssl genrsa -out key.pem 2048

# Generate certificate
openssl req -new -x509 -key key.pem -out cert.pem -days 365

# Generate DH parameters (optional, takes time)
openssl dhparam -out dhparam.pem 2048