#!/bin/bash
# Upload analysis data to GCS bucket for Cloud Run backend.
# Run from the repo root.
set -euo pipefail

BUCKET="gs://mlops-compute-lab-analysis-data"
DATA_DIR="venezuela-us-reddit-discourse/analysis/outputs"

echo "Creating bucket (if not exists)..."
gsutil mb -p mlops-compute-lab -l us-central1 "$BUCKET" 2>/dev/null || true

echo "Uploading sentiment data..."
gsutil -m cp \
  "$DATA_DIR/sentiment/sentiment_by_month.csv" \
  "$DATA_DIR/sentiment/sentiment_by_subreddit.csv" \
  "$DATA_DIR/sentiment/sentiment_by_subreddit_month.csv" \
  "$BUCKET/sentiment/"

echo "Uploading topics data..."
gsutil -m cp \
  "$DATA_DIR/topics/topic_info.csv" \
  "$DATA_DIR/topics/topics_by_subreddit.csv" \
  "$DATA_DIR/topics/topics_over_time.csv" \
  "$DATA_DIR/topics/topics_monthly.parquet" \
  "$BUCKET/topics/"

echo "Uploading clusters data..."
gsutil -m cp \
  "$DATA_DIR/clusters/cluster_summaries.csv" \
  "$DATA_DIR/clusters/cluster_keywords.csv" \
  "$DATA_DIR/clusters/temporal_clusters.csv" \
  "$DATA_DIR/clusters/cluster_assignments.parquet" \
  "$DATA_DIR/clusters/embeddings_2d.npy" \
  "$BUCKET/clusters/"

echo "Uploading visualizations..."
gsutil -m cp \
  "$DATA_DIR/visualizations/cluster_summary_table.csv" \
  "$BUCKET/visualizations/"

echo "Done. Verifying..."
gsutil ls -lh "$BUCKET/**"
