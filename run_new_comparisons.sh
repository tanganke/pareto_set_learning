#!/bin/bash
# Runner for the additional comparison experiments against recent
# preference-conditioned merging methods (Pareto Merging, PaLoRA) and AdaMerging.
#
#   bash run_new_comparisons.sh <job> <gpu>
#
# Results land in results/<script>/version_<version>/result.csv
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOGDIR=results/_logs
mkdir -p $LOGDIR

job=$1
gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EIGHT="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

run() {
    echo "+ $*"
    "$@"
}

# ---------------------------------------------------------------- baseline runs
# $1 baseline (pm|palora)  $2 partial  $3 version  $4 datasets  $5 lr
#  $6 num_evaluation_samples  $7 model  $8 batch_size
baseline_run() {
    local bl=$1 partial=$2 version=$3 datasets=$4 lr=$5 evalspec=$6 model=$7 bs=$8
    local base=${9:-null}
    run $PY scripts/clip_pareto_baselines.py train=true evaluate=false \
        baseline=$bl version=$version test_datasets="$datasets" model=$model \
        partial=$partial init_lambda=0.3 lr=$lr num_steps=4000 save_interval=1000 \
        batch_size=$bs base_merge_weight=$base
    run $PY scripts/clip_pareto_baselines.py train=false evaluate=true \
        baseline=$bl version=$version test_datasets="$datasets" model=$model \
        partial=$partial num_evaluation_samples=$evalspec batch_size=128 \
        base_merge_weight=$base '+eval_steps=[4000]'
}

# $1 method (ls|epo)  $2 partial  $3 version  $4 datasets  $5 evalspec  $6 model  $7 bs
#  $8 base_merge_weight (optional)
ours_run() {
    local method=$1 partial=$2 version=$3 datasets=$4 evalspec=$5 model=$6 bs=$7
    local base=${8:-null}
    run $PY scripts/clip_pareto_moe_${method}.py train=true evaluate=false \
        version=$version test_datasets="$datasets" model=$model partial=$partial \
        init_lambda=0.3 alpha=1 lr=1e-2 num_steps=4000 save_interval=1000 \
        batch_size=$bs base_merge_weight=$base
    run $PY scripts/clip_pareto_moe_${method}.py train=false evaluate=true \
        version=$version test_datasets="$datasets" model=$model partial=$partial \
        num_evaluation_samples=$evalspec batch_size=128 base_merge_weight=$base \
        '+eval_steps=[4000]'
}

AM_B32=results/clip_adamerging/ViT-B-32/version_0/merge_weight.pt
AM_L14=results/clip_adamerging/ViT-L-14/version_0/merge_weight.pt

case $job in
# ============================ wave 0: learning-rate probe on SUN397-Cars =======
probe_pm_lr3)    baseline_run pm     false 700 "[SUN397,Cars]" 1e-3 equal_weight ViT-B-32 16 ;;
probe_pm_lr2)    baseline_run pm     false 701 "[SUN397,Cars]" 1e-2 equal_weight ViT-B-32 16 ;;
probe_palora_lr3) baseline_run palora false 702 "[SUN397,Cars]" 1e-3 equal_weight ViT-B-32 16 ;;
probe_palora_lr2) baseline_run palora false 703 "[SUN397,Cars]" 1e-2 equal_weight ViT-B-32 16 ;;

# ============================ AdaMerging =======================================
adamerging_b32)
    run $PY scripts/clip_adamerging.py model=ViT-B-32 test_datasets="$EIGHT" \
        version=0 num_steps=500 lr=1e-3 init_lambda=0.3 batch_size=16 eval_batch_size=128
    ;;
adamerging_l14)
    run $PY scripts/clip_adamerging.py model=ViT-L-14 test_datasets="$EIGHT" \
        version=0 num_steps=500 lr=1e-3 init_lambda=0.3 batch_size=4 eval_batch_size=64
    ;;
adamerging_pairs_b32)
    for pair in "[SUN397,Cars]" "[SUN397,DTD]" "[Cars,DTD]"; do
        v=$(echo $pair | tr -d '[],' | tr ' ' '_')
        run $PY scripts/clip_adamerging.py model=ViT-B-32 test_datasets="$pair" \
            version=$v num_steps=500 lr=1e-3 init_lambda=0.3 batch_size=16 eval_batch_size=128
    done
    ;;

# ============================ eight tasks, ViT-B/32 ===========================
pm_mlp_8)      baseline_run pm     true  10 "$EIGHT" ${PM_LR:-1e-3} equal_weight+priority ViT-B-32 16 ;;
pm_all_8)      baseline_run pm     false 11 "$EIGHT" ${PM_LR:-1e-3} equal_weight+priority ViT-B-32 16 ;;
palora_mlp_8)  baseline_run palora true  12 "$EIGHT" ${PALORA_LR:-1e-3} equal_weight+priority ViT-B-32 16 ;;
palora_all_8)  baseline_run palora false 13 "$EIGHT" ${PALORA_LR:-1e-3} equal_weight+priority ViT-B-32 16 ;;
ours_ls_mlp_8) ours_run ls true  10 "$EIGHT" equal_weight+priority ViT-B-32 16 ;;
ours_ls_all_8) ours_run ls false 11 "$EIGHT" equal_weight+priority ViT-B-32 16 ;;

# ============================ two tasks, ViT-B/32 (Pareto fronts) =============
pm_pairs)
    baseline_run pm true  20 "[SUN397,Cars]" ${PM_LR:-1e-3} 11 ViT-B-32 16
    baseline_run pm false 21 "[SUN397,Cars]" ${PM_LR:-1e-3} 11 ViT-B-32 16
    ;;
palora_pairs)
    baseline_run palora true  20 "[SUN397,Cars]" ${PALORA_LR:-1e-3} 11 ViT-B-32 16
    baseline_run palora false 21 "[SUN397,Cars]" ${PALORA_LR:-1e-3} 11 ViT-B-32 16
    ;;
task_arithmetic_pair)
    run $PY scripts/clip_task_arithmetic.py model=ViT-B-32 tasks="[SUN397,Cars]" \
        version=SUN397Cars batch_size=128
    ;;
ours_ls_pairs)
    ours_run ls true  50 "[SUN397,Cars]" 11 ViT-B-32 16
    ours_run ls false 51 "[SUN397,Cars]" 11 ViT-B-32 16
    ;;
ours_epo_pairs)
    ours_run epo true  52 "[SUN397,Cars]" 11 ViT-B-32 16
    ours_run epo false 53 "[SUN397,Cars]" 11 ViT-B-32 16
    ;;
# ======== eight tasks, ViT-B/32, AdaMerging base (the setting used by PM/ReACT) ====
pm_all_8_am)      baseline_run pm     false 31 "$EIGHT" ${PM_LR:-1e-3} equal_weight+priority ViT-B-32 16 $AM_B32 ;;
palora_all_8_am)  baseline_run palora false 33 "$EIGHT" ${PALORA_LR:-1e-3} equal_weight+priority ViT-B-32 16 $AM_B32 ;;
ours_ls_mlp_8_am) ours_run ls true  30 "$EIGHT" equal_weight+priority ViT-B-32 16 $AM_B32 ;;
ours_ls_all_8_am) ours_run ls false 31 "$EIGHT" equal_weight+priority ViT-B-32 16 $AM_B32 ;;

# ======== eight tasks, ViT-L/14 ====================================================
ours_ls_mlp_8_l14)    ours_run ls true  40 "$EIGHT" equal_weight+priority ViT-L-14 8 ;;
ours_ls_all_8_l14)    ours_run ls false 41 "$EIGHT" equal_weight+priority ViT-L-14 8 ;;
ours_ls_all_8_l14_am) ours_run ls false 42 "$EIGHT" equal_weight+priority ViT-L-14 8 $AM_L14 ;;
pm_all_8_l14)         baseline_run pm false 41 "$EIGHT" ${PM_LR:-1e-3} equal_weight+priority ViT-L-14 8 ;;
palora_all_8_l14)     baseline_run palora false 43 "$EIGHT" ${PALORA_LR:-1e-3} equal_weight+priority ViT-L-14 8 ;;

*)
    echo "unknown job: $job" >&2
    exit 1
    ;;
esac
