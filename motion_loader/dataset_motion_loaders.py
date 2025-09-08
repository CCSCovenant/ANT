# -*- coding: utf-8 -*-
from datasets import get_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import os, hashlib, torch
from models.submodules.text_encoders import get_text_encoder            # NEW
from models.submodules.text_processors import create_text_processor, TextProcessorStrategy  # NEW
from tqdm import tqdm

# -------------------------- 1. 通用 Collate --------------------------
def collate_fn(batch):
    """
    batch[i] 结构:
        旧: (caption, motion, m_len, *other)
        新: (caption, motion, m_len, *other, raw_embed)  (当启用缓存)
    排序关键仍保持 m_len，对索引不造成影响。
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


# -------------------------- 2. 缓存包装数据集 --------------------------
class CachedTextWrapper(Dataset):
    """
    对原数据集进行包装，给每个样本附加 text_encoder 的“冻结输出 raw_embeds”。
    若本地已存在缓存且 hash 一致则直接加载，否则自动计算并持久化。
    """
    def __init__(self, base_ds: Dataset, opt, split: str, mode: str = 'train'):
        self.base        = base_ds
        self.opt         = opt
        self.split       = split
        self.encoder_tp  = opt.text_encoder_type.lower()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CAP_IDX = 2 if mode in ["eval", "gt_eval"] else 0

        # ---------- 2.1 目录&文件 ----------
        cache_root = getattr(opt, "text_cache_root", "./text_cache")
        ds_name    = getattr(opt, "dataset_name", "data")
        self.cache_dir  = os.path.join(cache_root, ds_name, split, self.encoder_tp)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "raw_embeds.pt")
        self.hash_file  = os.path.join(self.cache_dir, "hash.txt")

        # ---------- 2.2 数据完整性校验 ----------
        new_hash = self._calc_dataset_hash()
        old_hash = self._read_hash()

        # ---------- 2.3 获取 / 创建 cache ----------
        if (not opt.use_text_cache) or (old_hash != new_hash):
            print(f"[Cache]  缓存不存在或已过期({self.split}). 重新构建……")
            self.raw_cache = self._build_cache()
            if opt.use_text_cache:
                torch.save(self.raw_cache, self.cache_file)
                with open(self.hash_file, "w") as f: f.write(new_hash)
        else:
            print(f"[Cache]  读取已有缓存: {self.cache_file}")
            self.raw_cache = torch.load(self.cache_file, map_location="cpu")

    # ----------------- 计算数据集 hash -----------------
    def _calc_dataset_hash(self) -> str:
        m = hashlib.md5()
        for idx in tqdm(range(len(self.base)),desc="Calculating dataset hash"):
            cap = self.base[idx][self.CAP_IDX]                          # caption
            cap_str = " ".join(cap) if isinstance(cap, list) else str(cap)
            m.update(cap_str.encode("utf-8"))
        return m.hexdigest()

    def _read_hash(self):
        if not os.path.exists(self.hash_file):
            return ""
        return open(self.hash_file, "r").read().strip()

    # ----------------- 构建缓存 -----------------
    def _build_cache(self):
        # 1) 冻结 text_encoder
        text_encoder, _ = get_text_encoder(self.opt, device=self.device)
        text_encoder.eval()
        # 2) 创建只负责 no-grad 输出 raw 的 strategy
        dummy_proj = torch.nn.Identity()        # 不使用
        dummy_ln   = torch.nn.Identity()
        strategy: TextProcessorStrategy = create_text_processor(
            self.opt, text_encoder, dummy_proj, dummy_ln
        )
        raw_list = []
        with torch.no_grad():
            for idx in tqdm(range(len(self.base)),desc="Building text cache"):
                cap = self.base[idx][self.CAP_IDX]
                raw = strategy.get_raw_embeds([cap] if isinstance(cap, str) else cap)  # -> tensor
                raw = raw[0]
                raw_list.append(raw.cpu())
        return raw_list

    # ----------------- Dataset 接口 -----------------
    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.opt.use_text_cache:
            # 直接把 raw_embed 追加到末尾，保持原顺序不变
            return (*item, self.raw_cache[idx])
        else:
            return item


# -------------------------- 3. 外部工厂函数 --------------------------
def get_dataset_loader(opt, batch_size, mode='eval', split='test', accelerator=None):
    """
    根据 opt.use_text_cache 决定是否为数据集加一层缓存包装。
    """
    base_ds = get_dataset(opt, split, mode, accelerator)      # 原始数据集
    dataset = CachedTextWrapper(base_ds, opt, split, mode) if opt.use_text_cache else base_ds
    dl_kwargs = dict(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 32,
        drop_last   = True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    if mode in ['eval', 'gt_eval']:
        dl_kwargs["collate_fn"] = collate_fn
    else:
        dl_kwargs["persistent_workers"] = True

    return DataLoader(**dl_kwargs)
