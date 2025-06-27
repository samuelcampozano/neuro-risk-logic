# NeuroRiskLogic Deployment

This directory contains all files necessary for deploying NeuroRiskLogic in production.

## Quick Start

1. **Copy environment template**:
   ```bash
   cp .env.template .env
   # Edit .env with your production values
   ```

2. **Start services**:
   ```bash
   make up
   ```

3. **Run migrations**:
   ```bash
   make migrate
   ```

## Services

- **API**: FastAPI application (port 8000)
- **Database**: PostgreSQL 15 (port 5432)
- **Nginx**: Reverse proxy (ports 80/443)
- **Retraining**: Automated model retraining service
- **Prometheus**: Metrics collection (port 9090)
- **PgAdmin**: Database management UI (port 5050)

## Common Commands

```bash
# View logs
make logs

# Check status
make status

# Restart services
make restart

# Backup database
make backup

# Stop everything
make down
```

## Production Checklist

- [ ] Set strong passwords in `.env`
- [ ] Configure SSL certificates in `ssl/`
- [ ] Update CORS origins for your domain
- [ ] Set up firewall rules
- [ ] Configure backup automation
- [ ] Set up monitoring alerts
- [ ] Test disaster recovery

## Deployment Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  FastAPI    │────▶│ PostgreSQL  │
│  (Port 80)  │     │ (Port 8000) │     │ (Port 5432) │
└─────────────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │ Retraining  │
                    │   Service   │
                    └─────────────┘
```

## Security Notes

1. Never commit `.env` files
2. Use strong, unique passwords
3. Restrict database access
4. Enable SSL/TLS
5. Regular security updates
6. Monitor access logs

## Troubleshooting

### Database connection issues
```bash
# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U neurorisk_user -d neurorisk_db
```

### API not starting
```bash
# Check API logs
docker-compose logs api

# Rebuild if needed
docker-compose build api
```

### Migration failures
```bash
# Check migration status
docker-compose exec api alembic current

# Create migration manually
docker-compose exec api alembic revision --autogenerate -m "Description"
```