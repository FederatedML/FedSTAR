#!/bin/bash

read_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
    -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

server() {

    eval $(read_yaml config.yml "cfg_")

    python3 $(pwd)/src/server.py \
    `# Server Parameters` \
    --server_address=$cfg_global_SERVER_ADDRESS \
    --rounds=$cfg_server_FEDERATED_ROUNDS \
    --num_clients=$cfg_global_NUM_CLIENTS \
    --min_sample_size=$cfg_server_NUM_CLIENTS_PER_ROUND \
    --gpu_memory=$cfg_server_GPU_MEMORY_SIZE \
    --eval_step=$cfg_server_EVALUATION_STEP \
    --train_epochs=$cfg_server_LOCAL_TRAIN_STEP \
    `# Model Parameters` \
    --batch_size=$cfg_model_BATCH_SIZE \
    --learning_rate=$cfg_model_LR \
    `# Dataset Parameters` \
    --dataset_dir=$cfg_dataset_DATA_DIR \
    --seed=$cfg_global_SEED
}

client() {
    
    set -e
    eval $(read_yaml config.yml "cfg_")

    python3 $(pwd)/src/clients.py \
            `# Client Parameters` \
            --server_address=$cfg_global_SERVER_ADDRESS \
            --num_clients=$cfg_global_NUM_CLIENTS \
            --gpu_memory=$cfg_clients_GPU_MEMORY_SIZE \
            `# Dataset Parameters` \
            --dataset_dir=$cfg_dataset_DATA_DIR \
            --seed=$cfg_global_SEED \
            `# Model Parameters` \
            --batch_size=$cfg_model_BATCH_SIZE \
            `# Semi-Supervised Parameters` \
            --l_per=$cfg_dataset_LABELLED_PER \
            --u_per=$cfg_dataset_UNLABELED_PER \
            --fedstar=$cfg_global_FEDSTAR \
            --class_distribute=$cfg_global_C_DIST
}

# Execute Experiment
fuser -k 10001/tcp  2> /dev/null
server & (sleep 1; client)