#!/bin/bash
set -euo pipefail
BACKUP_DIR="/home/ubuntu/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker exec postgres pg_dump -U governance patient_feedback > "$BACKUP_DIR/governance_$TIMESTAMP.sql"
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
echo "Backup completed: governance_$TIMESTAMP.sql"
