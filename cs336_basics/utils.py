from dataclasses import fields

import numpy as np

def dict_to_params(args, param_cls):
    field_names = {f.name for f in fields(param_cls)}
    return param_cls(**{k: v for k, v in vars(args).items() if k in field_names})

def zarrs_1d_to_npy(zarr_list, out_path):
    """
    将多个 1D Zarr 数组顺序合并保存成一个大 .npy（mmap 友好）。
    
    参数：
        zarr_list:  Zarr 数组对象列表（已打开，shape 都是 1D）
        out_path:   输出 .npy 文件路径
    """
    # 计算总长度 & dtype
    total_len = sum(len(z) for z in zarr_list)
    dtype = zarr_list[0].dtype
    
    # 创建目标 memmap 文件
    mm = np.lib.format.open_memmap(out_path, mode='w+',
                                   dtype=dtype, shape=(total_len,))
    
    # 顺序写入
    offset = 0
    for z in zarr_list:
        chunk_size = z.chunks[0]  # 每次读取一个块大小
        for start in range(0, len(z), chunk_size):
            end = min(start + chunk_size, len(z))
            mm[offset:offset + (end - start)] = z[start:end]
            offset += (end - start)
    
    del mm  # flush 到磁盘