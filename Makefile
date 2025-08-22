init_wandb:
	poetry run wandb login
	export WANDB_PROJECT=cs336-ass1

download_artifacts:
	poetry run wandb artifact get tokens_tinystoriesV2_valid_valid:v0
	poetry run wandb artifact get tokens_tinystoriesV2_valid_train:v0

train_bpe:
	poetry run python cs336_basics/bootstrap.py train_bpe \
		--input_path data/TinyStoriesV2-GPT4-train.txt \
		--out_dir_path data/vocab_tinystoriesV2_train \
		--num_processors 8

tokenize:
	poetry run python cs336_basics/bootstrap.py tokenize \
		--input_path data/TinyStoriesV2-GPT4-valid.txt \
		--vocab_dir_path data/vocab_tinystoriesV2_train \
		--out_dir_path data/tokens_tinystoriesV2_valid \
		--num_processors 4

train_model:
	poetry run python cs336_basics/bootstrap.py train_model \
		--train_dir_path artifacts/tokens_tinystoriesV2_valid_train:v0 \
		--valid_dir_path artifacts/tokens_tinystoriesV2_valid_valid:v0 \
		--checkpoint_path artifacts/checkpoints \
		--num_epochs 1

chat:
	poetry run python cs336_basics/bootstrap.py chat \
		--checkpoint_path artifacts/checkpoints \
		--vocab_dir_path data/vocab_tinystoriesV2_train \
		--max_gen_len 10 \
		--temperature 0.6 \
		--top_p 0.5

upload_model:
	poetry run wandb artifact put artifacts/checkpoints --type model

.PHONY: init_wandb download_artifacts train_bpe tokenize train_model chat upload_model