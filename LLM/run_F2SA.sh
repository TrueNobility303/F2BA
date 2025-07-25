#!/bin/bash

data_dir=${data_dir:-'data_alpaca_0.5'}
freq_lambda=${freq_lambda:-5} 
lr_w=${lr_w:-1e-5}
lr_lambda=${lr_lambda:-1e-1}
alpha=${alpha:-10}
cd=${cd:-1} # whether to use central difference

function main() {
    seed=0

    batch_size=64
    micro_batch_size=8
    val_batch_size=1
    epoch=1
    exp_name=exp_gpt_F2SA_${data_dir}

    num_partitions=`ls -l ${data_dir}/train_*.json|wc -l`
    log_dir=log_${exp_name}/
    save_dir=models/${exp_name}/
    wandb_proj='F2SA'

    mkdir -p ${log_dir}
    mkdir -p ${save_dir}

    local exp_id=${exp_name}_cd-${cd}-lr_w-${lr_w}_lr_lambda-${lr_lambda}_seed-${seed}_alpha-${alpha}_epoch-${epoch}_freq_lambda-${freq_lambda}

          echo "$(date): ${exp_id}"

          accelerate launch  --config_file fsdp_config_gpt2.yaml python/main.py \
              --wandb_run_name ${exp_id} \
              --wandb_project ${wandb_proj} \
              --train-data ${data_dir}/train_\*.json \
              --val-data ${data_dir}/val.json \
              --test-data ${data_dir}/test.json \
              --model openai-community/gpt2 \
              --micro_batch_size ${micro_batch_size} \
              --global_batch_size $batch_size \
              --max-length 512 \
              --tokenizer-name openai-community/gpt2 \
              --optimizer "name=adamw" \
              --init_lr 5e-4 \
              --F2SA_init_lr_u ${lr_w} \
              --F2SA_init_lr_w ${lr_w} \
              --F2SA_init_lr_lambda ${lr_lambda} \
              --freq_lambda ${freq_lambda} \
              --lr_scheduler "name=cosine, min_lr=0.0" \
              --validation_model_mode "train" \
              --F2SA_validation_sampler stochastic \
              --sampler stochastic \
              --pseudo_random \
              --logging_conf_file conf/common.log_conf \
              --init_alpha ${alpha} \
              --tau 1 \
              --seed ${seed} \
              --epoch $epoch \
              --num_outer_iter 1 \
              --bf16 \
              --model-type gpt \
              --use_wandb \
              --val_batch_size ${val_batch_size} \
              --eval_frequency 50 \
              --num_partitions ${num_partitions} \
              --response_loss_only \
              --save_dir ${save_dir} \
              --csv_file ${log_dir}/${exp_id}.csv \
              > ${log_dir}/${exp_id}.log \
              2> ${log_dir}/${exp_id}.err
}

main "$@"
