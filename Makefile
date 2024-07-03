LANGUAGE := es
VOCAB_SIZE := 32000
MAX_TOKENS_TRAIN := -1
DATASET := wiki40b
TASK := language_modeling
SEED:= 7
MODEL := transformer
TOKENIZER := bpe

MODEL_ARCH := $(if $(filter-out $(MODEL), transformer),transformer_lm_big,transformer_lm)

DATA_DIR_BASE:=data
DATA_DIR:=$(DATA_DIR_BASE)/$(DATASET)/$(LANGUAGE)

DATA_RAW_DIR:=$(DATA_DIR)/raw
DATA_RAW_TRAIN_FILE:=$(DATA_RAW_DIR)/train.txt
DATA_RAW_VAL_FILE:=$(DATA_RAW_DIR)/validation.txt
DATA_RAW_TEST_FILE:=$(DATA_RAW_DIR)/test.txt
DATA_WIKI40B_RAW_TRAIN_FILE:=$(DATA_DIR_BASE)/wiki40b/$(LANGUAGE)/raw/train.txt
DATA_WIKI40B_RAW_VAL_FILE:=$(DATA_DIR_BASE)/wiki40b/$(LANGUAGE)/raw/validation.txt
DATA_WIKI40B_RAW_TEST_FILE:=$(DATA_DIR_BASE)/wiki40b/$(LANGUAGE)/raw/test.txt

DATA_SENTENCEPIECED_DIR:=$(DATA_DIR)/$(TOKENIZER)/sentencepiece
DATA_SENTENCEPIECED_MODEL_PREFFIX:=$(DATA_SENTENCEPIECED_DIR)/spm
DATA_SENTENCEPIECED_MODEL_FILE:=$(DATA_SENTENCEPIECED_MODEL_PREFFIX).model
DATA_SENTENCEPIECED_TRAIN_FILE:=$(DATA_SENTENCEPIECED_DIR)/train.txt
DATA_SENTENCEPIECED_VAL_FILE:=$(DATA_SENTENCEPIECED_DIR)/validation.txt
DATA_SENTENCEPIECED_TEST_FILE:=$(DATA_SENTENCEPIECED_DIR)/test.txt

ifneq ($(filter $(TOKENIZER),rawwords),)
	DATA_SRC_TRAIN_FILE:=$(DATA_RAW_TRAIN_FILE)
	DATA_SRC_VAL_FILE:=$(DATA_RAW_VAL_FILE)
	DATA_SRC_TEST_FILE:=$(DATA_RAW_TEST_FILE)
else
	DATA_SRC_TRAIN_FILE:=$(DATA_SENTENCEPIECED_TRAIN_FILE)
	DATA_SRC_VAL_FILE:=$(DATA_SENTENCEPIECED_VAL_FILE)
	DATA_SRC_TEST_FILE:=$(DATA_SENTENCEPIECED_TEST_FILE)
endif

DATA_SUBSAMPLED_DIR:=$(DATA_DIR)/$(TOKENIZER)/subsampled/$(MAX_TOKENS_TRAIN)
DATA_SUBSAMPLED_TRAIN_FILE:=$(DATA_SUBSAMPLED_DIR)/train.txt

DATA_PREPROCESSED_DIR:=$(DATA_DIR)/$(TOKENIZER)/preprocess/$(MAX_TOKENS_TRAIN)
DATA_PREPROCESSED_TEST_FILE:=$(DATA_PREPROCESSED_DIR)/test.bin

CHECKPOINT_DIR_BASE:=checkpoint/$(DATASET)
CHECKPOINT_DIR:=$(CHECKPOINT_DIR_BASE)/$(LANGUAGE)/$(TOKENIZER)
CHECKPOINT_MODEL_DIR:=$(CHECKPOINT_DIR)/$(TASK)/$(MODEL)/$(MAX_TOKENS_TRAIN)/
CHECKPOINT_MODEL_LAST_FILE:=$(CHECKPOINT_MODEL_DIR)/checkpoint_last.pt
CHECKPOINT_MODEL_BEST_FILE:=$(CHECKPOINT_MODEL_DIR)/checkpoint_best.pt

PREPROCESS_DATA_FLAGS := $(if $(filter-out $(TOKENIZER), rawwords),,--nwordssrc $(VOCAB_SIZE))

####### Make commands #######

all: get_data sentencepiece_data subsample_data preprocess_data train_model

all_cpu: get_data sentencepiece_data subsample_data preprocess_data

get_data: $(DATA_RAW_TEST_FILE)

ifneq ($(filter $(TOKENIZER),rawwords),)
sentencepiece_data:
else
sentencepiece_data: $(DATA_SENTENCEPIECED_MODEL_FILE) $(DATA_SENTENCEPIECED_TRAIN_FILE) $(DATA_SENTENCEPIECED_VAL_FILE) $(DATA_SENTENCEPIECED_TEST_FILE)
endif

subsample_data: $(DATA_SUBSAMPLED_TRAIN_FILE)

preprocess_data: $(DATA_PREPROCESSED_TEST_FILE)

train_model: $(CHECKPOINT_MODEL_LAST_FILE)

####### Actual commands #######

$(CHECKPOINT_MODEL_LAST_FILE):
	bash src/h02_learn/train_transformer.sh $(TASK) $(MODEL_ARCH) $(DATA_PREPROCESSED_DIR) $(CHECKPOINT_MODEL_DIR) $(SEED)

$(DATA_PREPROCESSED_TEST_FILE):
	rm -f $(DATA_PREPROCESSED_DIR)/*
	fairseq-preprocess --only-source  --trainpref $(DATA_SUBSAMPLED_TRAIN_FILE) \
    --validpref $(DATA_SRC_VAL_FILE) --testpref $(DATA_SRC_TEST_FILE) \
    --destdir $(DATA_PREPROCESSED_DIR) --workers 20 $(PREPROCESS_DATA_FLAGS)

$(DATA_SUBSAMPLED_TRAIN_FILE):
	mkdir -p $(DATA_SUBSAMPLED_DIR)
	python src/h01_data/subsample_data.py --max-tokens $(MAX_TOKENS_TRAIN) --src-file $(DATA_SRC_TRAIN_FILE) --tgt-file $(DATA_SUBSAMPLED_TRAIN_FILE)

$(DATA_SENTENCEPIECED_TEST_FILE):
	python fairseq/scripts/spm_encode.py --model=$(DATA_SENTENCEPIECED_MODEL_FILE) --output_format=piece < $(DATA_RAW_TEST_FILE) > $(DATA_SENTENCEPIECED_TEST_FILE)

$(DATA_SENTENCEPIECED_VAL_FILE):
	python fairseq/scripts/spm_encode.py --model=$(DATA_SENTENCEPIECED_MODEL_FILE) --output_format=piece < $(DATA_RAW_VAL_FILE) > $(DATA_SENTENCEPIECED_VAL_FILE)

$(DATA_SENTENCEPIECED_TRAIN_FILE):
	python fairseq/scripts/spm_encode.py --model=$(DATA_SENTENCEPIECED_MODEL_FILE) --output_format=piece < $(DATA_RAW_TRAIN_FILE) > $(DATA_SENTENCEPIECED_TRAIN_FILE)

$(DATA_SENTENCEPIECED_MODEL_FILE):
	mkdir -p $(DATA_SENTENCEPIECED_DIR)
	python fairseq/scripts/spm_train.py --input=$(DATA_RAW_TRAIN_FILE) --vocab_size=$(VOCAB_SIZE) --character_coverage=1.0 \
		--model_prefix=$(DATA_SENTENCEPIECED_MODEL_PREFFIX) --model_type=bpe --train_extremely_large_corpus=true

$(DATA_WIKI40B_RAW_TEST_FILE):
	rm -f $(DATA_WIKI40B_RAW_TRAIN_FILE) $(DATA_WIKI40B_RAW_VAL_FILE) $(DATA_WIKI40B_RAW_TEST_FILE)
	TF_CPP_MIN_LOG_LEVEL=3 tokenize_wiki_40b --language $(LANGUAGE) --tgt-dir $(DATA_RAW_DIR) --break-text-mode document --dont-tokenize
