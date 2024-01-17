#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=CoOpUPLTrainer

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
CLASS_EQULE=$7  # CLASS_EQULE True of False
TAG="rn50_random_init"
LAMBDA_S=$8
LAMBDA_Q=$9


for SEED in 1 # {1..3}
do
    DIR=./output_transductive/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_ls${LAMBDA_S}_lqA_EQULE_${CLASS_EQULE}_${CONF_THRESHOLD}_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    #if [ -d "$DIR" ]; then
    #    echo "Results are available in ${DIR}. Skip this job"
    #else
    echo "Run this job and save the output to ${DIR}"
    python coopupl_train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --n_shots ${SHOTS} \
    --lambda_s ${LAMBDA_S} \
    --lambda_q ${LAMBDA_Q} \
    TRAINER.CoOpUPLTrainer.N_CTX ${NCTX} \
    TRAINER.CoOpUPLTrainer.CSC ${CSC} \
    TRAINER.CoOpUPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS 16 \
    DATASET.CLASS_EQULE ${CLASS_EQULE}
    #fi
done


# #!/bin/bash

# cd ..

# # custom config
# DATA=./data
# TRAINER=ZeroshotCLIP
# DATASET=$1
# CFG=$2  # rn50, rn101, vit_b32 or vit_b16

# python sstrain.py \
# --root ${DATA} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/HHTrainer/${CFG}.yaml \
# --output-dir output/${TRAINER}/${CFG}/zeroshot/${DATASET} \
# --eval-only