#!/bin/bash
# Wait for comment collection to finish, then merge comments and re-run analysis.
set -euo pipefail

COMMENTS_DIR="/Users/hunjunsin/Desktop/Jun/capstone/tiktok/data-collection/data/comments"
MERGED_FILE="/Users/hunjunsin/Desktop/Jun/capstone/tiktok/data-collection/data/comments/comments_all_merged.json"
VENV="/Users/hunjunsin/Desktop/Jun/capstone/reddit/analysis/.venv/bin/python"

echo "=== Waiting for comment collection to finish ==="
while pgrep -f "main.py comments" > /dev/null 2>&1; do
    count=$(ls "$COMMENTS_DIR"/comments_*.json 2>/dev/null | wc -l)
    echo "  $(date +%H:%M:%S) — Still running... $count comment files so far"
    sleep 60
done
echo "=== Comment collection finished ==="

echo "=== Merging comment files ==="
python3 -c "
import json, os, glob

comments_dir = '$COMMENTS_DIR'
all_comments = []
for f in sorted(glob.glob(os.path.join(comments_dir, 'comments_*.json'))):
    data = json.load(open(f))
    all_comments.extend(data)

print(f'Merged {len(all_comments):,} comments from {len(glob.glob(os.path.join(comments_dir, \"comments_*.json\")))} files')

with open('$MERGED_FILE', 'w') as fh:
    json.dump(all_comments, fh, ensure_ascii=False, indent=2, default=str)
print(f'Saved to $MERGED_FILE')
"

echo "=== Re-running TikTok analysis ==="
cd /Users/hunjunsin/Desktop/Jun/capstone/reddit/analysis
$VENV ../../tiktok/analysis/run_analysis.py

echo "=== Uploading to GCS ==="
BUCKET="gs://mlops-compute-lab-analysis-data"
BASE="/Users/hunjunsin/Desktop/Jun/capstone/reddit/analysis/outputs_tiktok"

gsutil -m cp \
  "$BASE/sentiment/sentiment_by_month.csv" \
  "$BASE/sentiment/sentiment_by_source.csv" \
  "$BASE/sentiment/sentiment_by_source_month.csv" \
  "$BUCKET/outputs_tiktok/sentiment/"

gsutil -m cp \
  "$BASE/topics/topic_info.csv" \
  "$BASE/topics/topics_over_time.csv" \
  "$BASE/topics/monthly_topics_fitted.parquet" \
  "$BUCKET/outputs_tiktok/topics/"

gsutil -m cp \
  "$BASE/tiktok_specific/hashtag_trends.parquet" \
  "$BASE/tiktok_specific/engagement_metrics.parquet" \
  "$BASE/tiktok_specific/region_distribution.parquet" \
  "$BUCKET/outputs_tiktok/tiktok_specific/"

gsutil cp "$BASE/overview.json" "$BUCKET/outputs_tiktok/"

echo "=== ALL DONE ==="
echo "Restart Cloud Run to pick up new data:"
echo "  gcloud run services update backend-api --region us-central1 --update-env-vars=RESTART_TRIGGER=\$(date +%s)"
