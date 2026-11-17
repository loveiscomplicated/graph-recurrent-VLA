#!/usr/bin/env bash
# E3: 설정 × 세트 큐 (PAR 개 병렬). SPLIT=dev|eval
set -u
cd "$(dirname "$0")/.."
SPLIT=${SPLIT:-dev}
JOBS=logs/v1/e3_jobs_${SPLIT}.txt
: > $JOBS
for cfg in ${CFGS:-none v0conf v1 v1_tcp}; do
  for s in "$SPLIT" "$SPLIT patch" "$SPLIT dropout" "${SPLIT}_box_move"; do
    set -- $s
    line="$cfg $1${2:+ $2}"  # 끝에 공백이 있으면 xargs -L 이 다음 줄을 이어 붙인다
    [ -n "${SKIP:-}" ] && echo "$SKIP" | tr ',' '\n' | grep -qx "$line" && continue
    echo "$line" >> $JOBS
  done
done
run_one() {
  cfg=$1; set_=$2; pert=${3:-}
  source env.sh
  extra=""; [ -n "$pert" ] && extra="--perturb $pert"
  timeout 240m python scripts/e3_run.py --set $set_ --cfg $cfg $extra > logs/v1/e3_${cfg}_${set_}${pert:+_$pert}.log 2>&1
  echo "$(date +%T) done $cfg $set_ $pert exit=$?"
}
export -f run_one
cat $JOBS | xargs -P ${PAR:-2} -L 1 bash -c 'run_one "$@"' _
echo E3_DONE
