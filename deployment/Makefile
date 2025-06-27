# NeuroRiskLogic Deployment Makefile
# Commands for production deployment

.PHONY: help up down restart logs status backup restore migrate

# Default target
help:
	@echo "NeuroRiskLogic Deployment Commands"
	@echo "================================="
	@echo "up             Start all services"
	@echo "down           Stop all services"
	@echo "restart        Restart all services"
	@echo "logs           View logs"
	@echo "status         Check service status"
	@echo "backup         Backup database"
	@echo "restore        Restore database"
	@echo "migrate        Run database migrations"
	@echo "clean          Clean up volumes and containers"

# Start all services
up:
	docker-compose up -d
	@echo "✅ Services started"
	@echo "API: http://localhost:8000"
	@echo "PgAdmin: http://localhost:5050"
	@echo "Prometheus: http://localhost:9090"

# Stop all services
down:
	docker-compose down

# Restart all services
restart: down up

# View logs
logs:
	docker-compose logs -f

# Check service status
status:
	docker-compose ps

# Backup database
backup:
	@mkdir -p backups
	@docker-compose exec -T db pg_dump -U $(POSTGRES_USER) $(POSTGRES_DB) > backups/neurorisk_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backed up to backups/"

# Restore database from backup
restore:
	@echo "Available backups:"
	@ls -la backups/*.sql
	@echo "Usage: make restore FILE=backups/neurorisk_YYYYMMDD_HHMMSS.sql"

# Run database migrations
migrate:
	docker-compose exec api alembic upgrade head
	@echo "✅ Migrations completed"

# Clean up everything (WARNING: Deletes data!)
clean:
	docker-compose down -v
	@echo "⚠️  All data has been deleted"

# Production deployment
deploy:
	git pull origin main
	docker-compose build
	docker-compose up -d
	@echo "✅ Deployment completed"

# View API logs only
logs-api:
	docker-compose logs -f api

# View retraining logs only
logs-retrain:
	docker-compose logs -f retraining

# Create new migration
migrate-create:
	docker-compose exec api alembic revision --autogenerate -m "$(msg)"

# Rollback migration
migrate-rollback:
	docker-compose exec api alembic downgrade -1