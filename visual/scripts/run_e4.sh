#!/usr/bin/env bash
# E4: eval 40 ep pose/hole — GT mask (known / est size), tracker mask (SAM2-small conf)
cd "$(dirname "$0")/.."; source env.sh
(timeout 150m python scripts/e4_pose.py --set eval --masks gt --size known --name eval_gtmask_known > logs/v1/e4_gt_known.log 2>&1
 timeout 150m python scripts/e4_pose.py --set eval --masks gt --size est --name eval_gtmask_est > logs/v1/e4_gt_est.log 2>&1) &
timeout 150m python scripts/e4_pose.py --set eval --masks results/v1/E1/runs/sam2s_conf/eval --size known --name eval_sam2conf_known > logs/v1/e4_trk_known.log 2>&1 &
wait; echo E4_DONE
