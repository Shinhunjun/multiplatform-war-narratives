#!/bin/bash
# Setup Cloud Scheduler to trigger the pipeline daily
#
# Prerequisites:
#   1. gcloud CLI authenticated
#   2. Cloud Run Job "pipeline-etl-daily" already deployed
#   3. Cloud Scheduler API enabled
#
# Usage:
#   chmod +x setup_scheduler.sh
#   ./setup_scheduler.sh <PROJECT_ID> <REGION>

set -e

PROJECT_ID="${1:?Usage: ./setup_scheduler.sh <PROJECT_ID> <REGION>}"
REGION="${2:-us-central1}"
JOB_NAME="pipeline-etl-daily"
SCHEDULER_NAME="trigger-pipeline-daily"
# Run daily at 06:00 UTC (adjust as needed)
SCHEDULE="0 6 * * *"
TIMEZONE="Etc/UTC"

echo "Setting up Cloud Scheduler..."
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Schedule: $SCHEDULE ($TIMEZONE)"

# Enable required APIs
gcloud services enable cloudscheduler.googleapis.com --project="$PROJECT_ID"
gcloud services enable run.googleapis.com --project="$PROJECT_ID"

# Create a service account for the scheduler
SA_NAME="pipeline-scheduler-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Pipeline Scheduler Service Account" \
  --project="$PROJECT_ID" 2>/dev/null || echo "SA already exists"

# Grant Cloud Run Invoker role
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.invoker" \
  --condition=None

# Create the scheduler job
gcloud scheduler jobs create http "$SCHEDULER_NAME" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --schedule="$SCHEDULE" \
  --time-zone="$TIMEZONE" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
  --http-method=POST \
  --oauth-service-account-email="$SA_EMAIL" \
  --description="Triggers daily ETL pipeline for Venezuela narrative analysis"

echo ""
echo "Cloud Scheduler created successfully!"
echo "  Job: $SCHEDULER_NAME"
echo "  Triggers: Cloud Run Job '$JOB_NAME'"
echo "  Schedule: Daily at 06:00 UTC"
echo ""
echo "To test manually:"
echo "  gcloud scheduler jobs run $SCHEDULER_NAME --project=$PROJECT_ID --location=$REGION"
echo ""
echo "Don't forget to set secrets for the Cloud Run Job:"
echo "  gcloud run jobs update $JOB_NAME --region=$REGION \\"
echo "    --set-env-vars REDDIT_CLIENT_ID=xxx,REDDIT_CLIENT_SECRET=xxx \\"
echo "    --set-env-vars GCS_BUCKET=your-bucket"
