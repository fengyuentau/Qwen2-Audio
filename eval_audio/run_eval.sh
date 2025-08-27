#!/bin/bash
workspace_dir=/workspace
checkpoint=${workspace_dir}/models/Qwen2.5-Omni-7B
#for ds in "librispeech" "aishell2" "cv15_en" "cv15_zh" "cv15_yue" "cv15_fr" "fluers_zh"
for ds in "fluers_zh"
do
    python -m torch.distributed.launch --use_env \
        --nproc_per_node ${NPROC_PER_NODE:-1} --nnodes 1 \
        evaluate_asr.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 1 \
        --num-workers 2
done
