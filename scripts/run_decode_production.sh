#!/bin/bash
# Cross-engine TV production pipeline: teacher-forced replay oracle.
# This pipeline produced the paper's reference vllm-vs-sglang pools (2026-06-12).
#
# Design (aligned with the noisy-oracle model):
#   - Sampling: native generation, default kernels, bf16, one generate() call
#     per engine (--env replay registers the force processor / sizes the
#     sglang token pool so sample and score share one engine config).
#   - Primary oracle = REPLAY: teacher-forced generation (true decode path,
#     same batch shape as sampling). vllm returns unpolluted raw logprobs;
#     sglang captures pre-forcing top-k inside the processor.
#     Validated: replay == sample-time logprobs bit-exactly (vllm) /
#     own-zero = 0 and ~4e-5 readout agreement (sglang).
#   - Comparison oracle = SEQ: the engines' standard whole-sequence logprob
#     API. Tests the hypothesis that single-forward scoring underestimates
#     the noise of the generative process (no per-step accumulation).
#   - K_REPLAY=8 (replay measured deterministic; repeats are verification),
#     K_SEQ=16. Full top-k saved for the first 2 repeats of each tensor
#     (sigma/top-k dataset). B-chunked replay (chunk 256) = batch-shape
#     sensitivity diagnostic.
#
# Every phase is guarded by its output file => resubmit-safe.
set -uo pipefail

: ${N:=512}
: ${n:=500}
: ${MODEL:=Qwen/Qwen3-0.6B}
: ${TOP_K:=20}
: ${OUTDIR_REL:=results/vllm_vs_sglang_v2}
: ${SEED:=0}
: ${K_REPLAY:=8}
: ${K_SEQ:=16}
: ${SAVE_TOPK:=2}
: ${SEQ_CHUNK:=128}
: ${DIAG_CHUNK:=256}
: ${USER_PROMPT:="Tell me a story about a robot learning to paint."}
: ${RUN_SEQ:=1}
: ${RUN_DIAG:=1}

# Site setup (TACC Vista used the line below; replace with your toolchain):
# . /opt/apps/lmod/lmod/init/profile && module reset && module load gcc/14.2.0 cuda/12.8 python3/3.11.8

# Two separate venvs (vllm and sglang pin conflicting deps); build them with
# scripts/setup_{vllm,sglang}_venv.sh, then point these at the results:
VLLM_VENV=${VLLM_VENV:?set VLLM_VENV to the vllm venv path}
SGLANG_VENV=${SGLANG_VENV:?set SGLANG_VENV to the sglang venv path}
PROJECT=${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}
OUTDIR=$PROJECT/$OUTDIR_REL

mkdir -p $OUTDIR/seq
cd $PROJECT
export PYTHONPATH=$PROJECT:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/path/to/hf_cache   # optional: pin the HF model cache location

run_in_venv() {
    local venv=$1; shift
    bash -c "source $venv/bin/activate && python -u $*"
}
venv_for() { [ "$1" = vllm ] && echo $VLLM_VENV || echo $SGLANG_VENV; }

echo "[$(date -u +%H:%M:%S)] === v2 production: N=$N n=$n K_replay=$K_REPLAY K_seq=$K_SEQ ==="
echo "[$(date -u +%H:%M:%S)] outdir=$OUTDIR prompt='$USER_PROMPT'"

# ── 1. Sample (native generation, replay-parity engine config) ───────────────
if [ ! -f $OUTDIR/X_pi.pt ]; then
    echo "[$(date -u +%H:%M:%S)] sample X_pi via vllm"
    run_in_venv $VLLM_VENV experiments/inference_engines/phase_sample.py \
        --engine vllm --model $MODEL --N $N --n $n --top_k $TOP_K \
        --seed $SEED --env replay --user "\"$USER_PROMPT\"" --out $OUTDIR/X_pi.pt
fi
if [ ! -f $OUTDIR/X_mu.pt ]; then
    echo "[$(date -u +%H:%M:%S)] sample X_mu via sglang"
    run_in_venv $SGLANG_VENV experiments/inference_engines/phase_sample.py \
        --engine sglang --model $MODEL --N $N --n $n --top_k $TOP_K \
        --seed $((SEED+1)) --env replay --user "\"$USER_PROMPT\"" --out $OUTDIR/X_mu.pt
fi

# ── 2a. REPLAY scores first (headline pools complete earliest) ──────────────
for ENG in vllm sglang; do
    SUF=$([ "$ENG" = vllm ] && echo pi || echo mu)
    SEED_E=$([ "$ENG" = vllm ] && echo $SEED || echo $((SEED+1)))
    VENV=$(venv_for $ENG)
    for SIDE in pi mu; do
        if [ ! -f $OUTDIR/score_${SIDE}_${SUF}.done ]; then
            echo "[$(date -u +%H:%M:%S)] $ENG replay-scores X_$SIDE (K=$K_REPLAY)"
            run_in_venv $VENV experiments/inference_engines/phase_score.py \
                --engine $ENG --X_path $OUTDIR/X_$SIDE.pt --K_max $K_REPLAY \
                --model $MODEL --top_k $TOP_K --seed $SEED_E \
                --score_path replay --save_topk_reps $SAVE_TOPK \
                --out $OUTDIR/score_${SIDE}_${SUF}.pt \
                && touch $OUTDIR/score_${SIDE}_${SUF}.done
        fi
    done
done

# combine + analyze replay pools as soon as they exist
for SIDE in pi mu; do
    if [ ! -f $OUTDIR/${SIDE}_pool.pt ]; then
        run_in_venv $VLLM_VENV experiments/inference_engines/phase_combine.py \
            --X_path $OUTDIR/X_$SIDE.pt \
            --score_pi_path $OUTDIR/score_${SIDE}_pi.pt \
            --score_mu_path $OUTDIR/score_${SIDE}_mu.pt \
            --side $SIDE --out $OUTDIR/${SIDE}_pool.pt
    fi
done
run_in_venv $VLLM_VENV experiments/inference_engines/analysis/quick_decode_analysis.py \
    --dir $OUTDIR || true
touch $OUTDIR/REPLAY_DONE

# ── 2b. SEQ comparison oracle + shape diagnostics ────────────────────────────
for ENG in vllm sglang; do
    SUF=$([ "$ENG" = vllm ] && echo pi || echo mu)
    SEED_E=$([ "$ENG" = vllm ] && echo $SEED || echo $((SEED+1)))
    VENV=$(venv_for $ENG)
    for SIDE in pi mu; do
        if [ "$RUN_SEQ" = 1 ] && [ ! -f $OUTDIR/seq/score_${SIDE}_${SUF}.done ]; then
            echo "[$(date -u +%H:%M:%S)] $ENG seq-scores X_$SIDE (K=$K_SEQ)"
            run_in_venv $VENV experiments/inference_engines/phase_score.py \
                --engine $ENG --X_path $OUTDIR/X_$SIDE.pt --K_max $K_SEQ \
                --model $MODEL --top_k $TOP_K --seed $SEED_E \
                --score_path seq --chunk $SEQ_CHUNK --save_topk_reps $SAVE_TOPK \
                --out $OUTDIR/seq/score_${SIDE}_${SUF}.pt \
                && touch $OUTDIR/seq/score_${SIDE}_${SUF}.done
        fi
    done
    if [ "$RUN_DIAG" = 1 ] && [ ! -f $OUTDIR/score_${SUF}_${SUF}_b${DIAG_CHUNK}.done ]; then
        echo "[$(date -u +%H:%M:%S)] $ENG replay shape-diag (chunk=$DIAG_CHUNK) on own pool"
        run_in_venv $VENV experiments/inference_engines/phase_score.py \
            --engine $ENG --X_path $OUTDIR/X_$SUF.pt --K_max 1 \
            --model $MODEL --top_k $TOP_K --seed $SEED_E \
            --score_path replay --chunk $DIAG_CHUNK \
            --out $OUTDIR/score_${SUF}_${SUF}_b${DIAG_CHUNK}.pt \
            && touch $OUTDIR/score_${SUF}_${SUF}_b${DIAG_CHUNK}.done
    fi
done

# ── 3. Combine + acceptance analysis ─────────────────────────────────────────
for V in "seq/"; do
    [ "$RUN_SEQ" != 1 ] && continue
    for SIDE in pi mu; do
        if [ ! -f $OUTDIR/${V}${SIDE}_pool.pt ]; then
            run_in_venv $VLLM_VENV experiments/inference_engines/phase_combine.py \
                --X_path $OUTDIR/X_$SIDE.pt \
                --score_pi_path $OUTDIR/${V}score_${SIDE}_pi.pt \
                --score_mu_path $OUTDIR/${V}score_${SIDE}_mu.pt \
                --side $SIDE --out $OUTDIR/${V}${SIDE}_pool.pt
        fi
    done
    echo "[$(date -u +%H:%M:%S)] analysis (${V:-replay})"
    run_in_venv $VLLM_VENV experiments/inference_engines/analysis/quick_decode_analysis.py \
        --dir $OUTDIR/${V%/} || true
done

cd $OUTDIR && md5sum *.pt seq/*.pt 2>/dev/null > md5sums.txt
touch $OUTDIR/ALL_DONE
echo "[$(date -u +%H:%M:%S)] === DONE ==="
