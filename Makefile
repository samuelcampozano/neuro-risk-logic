# NeuroRiskLogic Makefile
# Common development and deployment tasks

.PHONY: help install dev test clean docker-build docker-up train retrain evaluate monitor

# Default target
help:
	@echo "NeuroRiskLogic Development Commands"
	@echo "=================================="
	@echo "install        Install dependencies"
	@echo "dev            Run development server"
	@echo "test           Run tests"
	@echo "clean          Clean temporary files"
	@echo "docker-build   Build Docker images"
	@echo "docker-up      Start Docker services"
	@echo "train          Train initial model"
	@echo "retrain        Run model retraining"
	@echo "evaluate       Evaluate current model"
	@echo "monitor        Generate monitoring report"
	@echo "simulate       Simulate test data"
	@echo "format         Format code with black"
	@echo "lint           Run linting checks"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run development server
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=app --cov-report=html

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ML Pipeline commands
train:
	python scripts/generate_synthetic_data.py -n 1000
	python scripts/train_model.py --visualize

retrain:
	python scripts/retrain_with_real_data.py

retrain-force:
	python scripts/retrain_with_real_data.py --force

evaluate:
	python scripts/evaluate_model.py --plots

monitor:
	python scripts/generate_monitoring_report.py --open

# Development tools
simulate:
	python scripts/dev_tools.py simulate -n 100 -d 30

health-check:
	python scripts/dev_tools.py health

export-data:
	python scripts/dev_tools.py export -o data/export_$(shell date +%Y%m%d).csv

# Code quality
format:
	black app/ scripts/ tests/

lint:
	flake8 app/ scripts/ tests/ --max-line-length=100
	mypy app/ --ignore-missing-imports

# Database
db-init:
	alembic init alembic
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head

db-migrate:
	alembic revision --autogenerate -m "$(msg)"
	alembic upgrade head

db-reset:
	python scripts/dev_tools.py reset

# Production deployment
deploy-prod:
	@echo "Deploying to production..."
	git push origin main
	ssh prod-server 'cd /opt/neurorisk && git pull && docker-compose up -d --build'

# Backup
backup:
	mkdir -p backups
	pg_dump -h localhost -U $(POSTGRES_USER) $(POSTGRES_DB) > backups/neurorisk_$(shell date +%Y%m%d_%H%M%S).sql
	tar -czf backups/models_$(shell date +%Y%m%d_%H%M%S).tar.gz data/models/

# Service management
start-retraining:
	python scripts/automated_retraining.py &

stop-retraining:
	pkill -f "automated_retraining.py"

# Environment setup
setup-env:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# Full development setup
setup-dev: install db-init train
	@echo "Development environment ready!"
	@echo "Run 'make dev' to start the server"