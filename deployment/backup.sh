#!/bin/bash
# Automated backup script for NeuroRiskLogic

# Configuration
BACKUP_DIR="/backups"
DB_CONTAINER="neurorisk_db"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
echo "Starting database backup..."
docker exec -t $DB_CONTAINER pg_dumpall -c -U $POSTGRES_USER > $BACKUP_DIR/neurorisk_db_$TIMESTAMP.sql

# Backup data directory
echo "Backing up data files..."
tar -czf $BACKUP_DIR/neurorisk_data_$TIMESTAMP.tar.gz ../data/

# Backup configuration
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/neurorisk_config_$TIMESTAMP.tar.gz .env nginx.conf prometheus.yml

# Clean old backups
echo "Cleaning old backups..."
find $BACKUP_DIR -name "neurorisk_*" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $TIMESTAMP"

# Optional: Upload to cloud storage
# aws s3 cp $BACKUP_DIR/neurorisk_db_$TIMESTAMP.sql s3://your-bucket/backups/