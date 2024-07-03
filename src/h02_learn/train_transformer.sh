TASK=$1
MODEL_ARCH=$2
DATA_DIR=$3
SAVE_DIR=$4
RANDOM_SEED=$5
EXTRA_FLAGS=$6

WANDB_NAME=${DATA_DIR} \
fairseq-train --task ${TASK} \
	${DATA_DIR} \
	--save-dir ${SAVE_DIR} \
	--arch ${MODEL_ARCH} --share-decoder-input-output-embed \
	--dropout 0.1 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
	--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
	--tokens-per-sample 512 --sample-break-mode none \
	--max-tokens 32768 --update-freq 1 \
	--fp16 \
	--max-update 100000 --max-epoch 100 --patience 3  \
	--seed ${RANDOM_SEED} \
	--keep-last-epochs 1 \
	--user-dir src/h02_learn/ --wandb-project fairseq-training \
	${EXTRA_FLAGS}
