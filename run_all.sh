#!/bin/bash
# Full RECAP-Leaf pipeline (resumable).  DATA_DIR defaults to ../data.
set -e
cd "$(dirname "$0")"; DATA=${RECAP_DATA:-../data}; mkdir -p ../work/logs ../work/results
job() { out=../work/$1; shift
  if [ -e "$out" ]; then echo "SKIP $*"; return; fi
  echo "START $* $(date +%T)"; python3 "$@" > ../work/logs/$(echo "$*" | tr ' /.' '___').log 2>&1; echo "END $* $(date +%T)"; }
[ -f $DATA/manifest.csv ] || python3 prepare_data.py $DATA
[ -f $DATA/scenes.json ] || python3 prepare_scenes.py $DATA
[ -f $DATA/regional_subset.txt ] || python3 leaf_masks.py $DATA/manifest.csv
[ -f $DATA/unlabeled.txt ] || python3 prepare_unlabeled.py $DATA
job sets.json embed_sets.py
M=mobilenetv4_conv_small.e2400_r224_in1k
job student_s0.pt train_student.py 0 10
job student_labonly_s0.pt train_student.py 0 10 --labonly
job base_${M}_s0.pt train_baselines.py $M 0 4
job base_source_cnn_s0.pt train_baselines.py source_cnn 0 12
job base_densenet121_s0.pt train_baselines.py densenet121 0 3
job emb_teacher_s0.npz embed.py teacher 0
for m in student student_labonly ce_$M ce_source_cnn ce_densenet121; do job emb_${m}_s0.npz embed.py $m 0; done
job results/cls_runs_all.json exp_classification.py
for b in teacher student; do
  job recap_bank_${b}_s0.npz recapture_frames.py $b 0
  job results/recap_${b}_s0.json exp_recapture.py $b 0
done
for k in 0 1 2; do job results/tta_s0_k$k.json exp_tta.py 0 $k; done
job results/abstention.json exp_abstention.py
job results/efficiency.json exp_efficiency.py
python3 analyze.py
python3 fig_arch.py && python3 fig_capture.py . && python3 fig_results.py ../work/results/summary.json .
echo PIPELINE_DONE
