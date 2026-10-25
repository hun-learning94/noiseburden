"""Drop only the final incomplete effective batch, retaining sampler order."""
from itertools import islice
from torch.utils.data import BatchSampler


class CompleteUpdateBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, accumulation):
        super().__init__(sampler, batch_size=batch_size, drop_last=True)
        assert accumulation >= 1
        self.complete_microbatches = super().__len__() // accumulation * accumulation

    def __len__(self):
        return self.complete_microbatches

    def __iter__(self):
        return islice(super().__iter__(), self.complete_microbatches)
