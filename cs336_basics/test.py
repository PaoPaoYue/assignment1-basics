# import pickle
# from cs336_basics.data_loader import RandomStartBatchSampler, TokensDataset
# from cs336_basics.tokenizer import BpeTokenizer
# from cs336_basics.train import TrainParams
# import numpy as np
# from requests import get
# from torch.utils.data import DataLoader

# with open("data/vocab_tinystoriesV2_valid/vocab.pkl", "rb") as f:
#     vocab = pickle.load(f)
# with open("data/vocab_tinystoriesV2_valid/merges.pkl", "rb") as f:
#     merges = pickle.load(f)


# tokenizer = BpeTokenizer(vocab, merges, special_tokens=['<|endoftext|>'])

# mm = np.load('data/tokens_tinystoriesV2_valid/tokens.npy')
# print(len(mm))


# def get_dataloaders(
#     params: TrainParams
# ) -> tuple[DataLoader, DataLoader]:
#     train_ds = TokensDataset(data=np.load(f"{params.train_dir_path}/tokens.npy", mmap_mode='r'), context_length=params.context_length)
#     valid_ds = TokensDataset(data=np.load(f"{params.valid_dir_path}/tokens.npy", mmap_mode='r'), context_length=params.context_length)
#     train_loader = DataLoader(
#         train_ds,
#         batch_sampler=RandomStartBatchSampler(len(train_ds), batch_size=params.batch_size),
#         pin_memory=True,
#         num_workers=params.loader_num_workers,
#         persistent_workers=(params.loader_num_workers > 0),
#     )
#     valid_loader = DataLoader(
#         valid_ds,
#         batch_size=params.batch_size,
#         shuffle=False,
#         num_workers=params.loader_num_workers,
#         pin_memory=True,
#         drop_last=False,
#         persistent_workers=(params.loader_num_workers > 0),
#     )
#     return train_loader, valid_loader

# train_loader, valid_loader = get_dataloaders(TrainParams(
#     train_dir_path="data/tokens_tinystoriesV2_train",
#     valid_dir_path="data/tokens_tinystoriesV2_valid",
#     context_length=256,
#     batch_size=64,
#     loader_num_workers=4,
#     checkpoint_path="data/checkpoints"
# ))

# for i, batch in enumerate(train_loader):
#     print(batch)
#     if i > 10:
#         break

# import numpy as np
# import os

# # 读取原始数据
# mm = np.load('data/tokens_tinystoriesV2_valid/tokens.npy')

# # 计算训练集大小（9:1）
# train_size = int(len(mm) * 0.9)

# # 如果第 90% 个位置不是 0，则向前找到最近的 0
# while train_size > 0 and mm[train_size - 1] != 0:
#     train_size -= 1

# train_set = mm[:train_size]
# valid_set = mm[train_size:]

# # 创建保存目录
# os.makedirs('tokens_tinystoriesV2_valid_train', exist_ok=True)
# os.makedirs('tokens_tinystoriesV2_valid_valid', exist_ok=True)

# # 保存
# np.save('tokens_tinystoriesV2_valid_train/tokens.npy', train_set)
# np.save('tokens_tinystoriesV2_valid_valid/tokens.npy', valid_set)

# print(f"Train size: {len(train_set)}, Valid size: {len(valid_set)}")
# print(f"最后一个 train 元素: {train_set[-1]}")
