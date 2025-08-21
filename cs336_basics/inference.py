from dataclasses import dataclass
import os
import pickle
from typing import Iterable

from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.nn_model import TransformerLM
from cs336_basics.nn_utils import top_p_sampling
from cs336_basics.tokenizer import BpeTokenizer
from cs336_basics.train import TrainParams
from torch import nn
import torch


@dataclass
class ChatParams:
    checkpoint_path: str
    vocab_dir_path: str
    special_tokens: list[str] = ('<|endoftext|>',)

    max_gen_len: int = 256
    temperature: float = 1.0
    top_p: float = 0.0

    device: str = 'cuda'


def chat(params: ChatParams):
    model, _ = init_model(params.checkpoint_path, params.device)
    tokenizer = init_tokenizer(params.vocab_dir_path, params.special_tokens)

    print("=" * 50)
    print("💬  简易终端对话 (输入 exit 退出)")
    print("=" * 50)

    round = 1
    while True:
        prompt = input(f"\n🟢 你({round}): ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("\n👋 对话结束，再见！")
            break

        print("🤖 模型:", end=" ", flush=True)
        for chunk in inference(prompt, model, tokenizer, params):
            print(chunk, end="", flush=True)
        print()  # 输出换行
        round += 1

def init_model(checkpoint_path: str | os.PathLike, device: str) -> tuple[nn.Module, TrainParams]:

    if os.path.isdir(checkpoint_path):
        for filename in os.listdir(checkpoint_path):
            if filename.endswith(".pt"):
                checkpoint_path = os.path.join(checkpoint_path, filename)
                break
        else:
            raise FileNotFoundError(f"No checkpoint files found in directory {checkpoint_path}")

    model_weights, _, _, model_info = load_checkpoint(checkpoint_path)
    params = model_info["train_param"]
    model = TransformerLM(
        vocab_size=params.vocab_size,
        context_length=params.context_length,
        d_model=params.d_model,
        d_ff=params.d_ff,
        num_layers=params.num_layers,
        num_heads=params.num_heads,
        rope_theta=params.rope_theta,
        device=device
    )
    model.load_state_dict(model_weights)
    model.eval()

    print(f"模型已从 {checkpoint_path} 加载。")
    print(f"模型训练设定 {params} 。")
    return model, params

def init_tokenizer(vocab_dir_path: str, special_tokens: list[str]) -> BpeTokenizer:
    with open(f"{vocab_dir_path}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"{vocab_dir_path}/merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tokenizer = BpeTokenizer(vocab, merges, special_tokens=special_tokens)
    print(f"分词器已从 {vocab_dir_path} 初始化。")
    return tokenizer

@torch.no_grad()
def inference(
    prompt: str,
    model: nn.Module,
    tokenizer: BpeTokenizer,
    chat_params: ChatParams
) -> Iterable[str]:

    token_count = 0
    current_token = -1

    inputs = tokenizer.encode(prompt)
    inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0) 
    while True:
        if token_count >= chat_params.max_gen_len:
            break

        outputs = model(inputs)
        probs = nn.functional.softmax(outputs[:, -1, :] / chat_params.temperature, dim=-1)

        if chat_params.top_p > 0:
            probs = top_p_sampling(probs, chat_params.top_p)

        current_token = torch.multinomial(probs, num_samples=1).item()
        token_count += 1
        if current_token == 0:
            break

        yield tokenizer.decode([current_token])
        # append current_token to inputs
        inputs = torch.cat([inputs, torch.tensor([[current_token]], dtype=torch.long)], dim=-1)