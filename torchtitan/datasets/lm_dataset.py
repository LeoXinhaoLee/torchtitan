# Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt/datasets.py
# Except we don't pad the last block and don't use overlapping eval
# And we return both the input and the target
import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import RandomSampler, DistributedSampler


class RandomFaultTolerantSampler(RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # generator = torch.Generator().manual_seed(seed)
        # super().__init__(*args, generator=generator, **kwargs)
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called before hand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        self.restarting = True

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len, drop_last=True, llama=False, bos_token_id=None):
        """tokens should be a numpy array"""
        self.seq_len = seq_len
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // seq_len) * seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.seq_len)
        self.llama = llama
        self.bos_token_id = bos_token_id

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        idx = idx % self.ntokens
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.ntokens - 1 - start_idx)
        # data = torch.as_tensor(self.tokens[start_idx : (start_idx + seq_len + 1)].astype(np.int32))
        data = torch.as_tensor(self.tokens[start_idx : (start_idx + seq_len + 1)].astype(np.int64))
        if self.llama:
            return {
                "input_tokens": data[:-1],
                "target_tokens": data[1:].clone(),
                "loss_masks": (data[1:] != self.bos_token_id).to(torch.float32),
            }
        else:
            return {
                "input_tokens": data[:-1],
                "target_tokens": data[1:].clone(),
                "loss_masks": torch.ones_like(data[:-1], dtype=torch.float32),
            }
