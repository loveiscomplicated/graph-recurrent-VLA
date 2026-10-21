#!/usr/bin/env bash
# E1: 5 tracker × re-anchor {none, conf, conf_e5} on eval 40 ep. 3개씩 병렬 (정확도용; timing 은 따로 단독 측정)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v1
JOBS=logs/v1/e1_jobs.txt
: > $JOBS
for re in none conf conf_e5; do
  echo "vision sam2s_$re --backend sam2 --model small --compile 0 --reanchor $re" >> $JOBS
  echo "vision sam2s_c_$re --backend sam2 --model small --compile 1 --reanchor $re" >> $JOBS
  echo "vision sam2bp_$re --backend sam2 --model base_plus --compile 0 --reanchor $re" >> $JOBS
  echo "samurai samurai_s_$re --backend samurai --model small --mode single --compile 0 --reanchor $re" >> $JOBS
  echo "efficienttam etam_ti_$re --backend efficienttam --model ti --compile 0 --reanchor $re" >> $JOBS
done
run_one() {
  venv=$1; tag=$2; shift 2
  export HF_HOME=${VISION_CACHE:-$PWD/cache}/hf TORCH_HOME=${VISION_CACHE:-$PWD/cache}/torch
  source ${VISION_ENVS:-$HOME/envs}/$venv/bin/activate
  timeout 120m python scripts/track_v1.py --set ${SET:-eval} --tag $tag "$@" > logs/v1/e1_${tag}_${SET:-eval}.log 2>&1
  echo "$(date +%T) done $tag exit=$?"
}
export -f run_one
cat $JOBS | xargs -P ${PAR:-3} -L 1 bash -c 'run_one "$@"' _
echo E1_DONE
