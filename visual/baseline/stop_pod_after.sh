#!/usr/bin/env bash
# Stop this RunPod pod when a queue process exits OR at a hard deadline, whichever
# comes first (the user's credit runs out later). At the deadline, experiments are
# terminated (SIGTERM, then SIGKILL). Then: summary → ntfy → HF backup → pod stop.
# Usage: bash visual/baseline/stop_pod_after.sh <queue pid> <deadline epoch> <pgid>...
set -u
cd /workspace/graph-recurrent-VLA
queue_pid=$1
deadline=$2
shift 2
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
notify() { curl -s -m 15 -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" > /dev/null || true; }

while kill -0 "$queue_pid" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do sleep 30; done

if kill -0 "$queue_pid" 2>/dev/null; then
    echo "deadline reached at $(date -u); terminating experiments (process groups: $*)"
    for group in "$@"; do kill -TERM -- "-$group" 2>/dev/null; done
    sleep 30
    for group in "$@"; do kill -KILL -- "-$group" 2>/dev/null; done
    reason="deadline reached, unfinished work stopped"
else
    echo "queue finished at $(date -u)"
    reason="all queued experiments finished"
fi

summary=$(/workspace/envs/baseline/bin/python visual/baseline/summarize.py $base 2>&1 | cut -c1-1500)
echo "summary: $summary"
notify "final: $reason" "$summary"

echo "backing up to HF at $(date -u)"
timeout 900 /workspace/envs/vision/bin/python visual/baseline/backup_to_hf.py kdh1270/visual-baseline-backup 2>&1 | grep -v -i warn | tail -2
backup_status=$?
sync
sleep 10
notify "backup + stopping pod" "HF backup exit=$backup_status (kdh1270/visual-baseline-backup, SUMMARY.md in outputs/visual_baseline). Stopping pod now."

api() {
    curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
        https://api.runpod.io/graphql -d "{\"query\":\"$1\"}"
}
for attempt in 1 2 3 4 5; do
    echo "stop attempt $attempt at $(date -u)"
    api "mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }"
    echo
    sync
    sleep 60
done
