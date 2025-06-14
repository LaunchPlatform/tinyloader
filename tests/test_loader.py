import typing
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import tinygrad
import tqdm

from tinyloader.loader import load
from tinyloader.loader import load_with_workers
from tinyloader.loader import Loader
from tinyloader.loader import SharedMemoryShim


class RandomLoader(Loader):
    def __init__(self, data_size: tuple[int, ...], label_size: tuple[int, ...]):
        self.data_size = data_size
        self.label_size = label_size

    def make_request(self, item: float) -> typing.Any:
        return item

    def load(self, request: float) -> tuple[np.typing.NDArray, ...]:
        return np.random.normal(request, 1.0, size=self.data_size), np.random.normal(
            request, 1.0, size=self.label_size
        )

    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
        x, y = response
        return tinygrad.Tensor(x).contiguous().realize(), tinygrad.Tensor(
            y
        ).contiguous().realize()


def test_load():
    data_size = (64, 64)
    label_size = (4,)
    n = 1000
    loader = RandomLoader(data_size=data_size, label_size=label_size)
    count = 0
    for x, y in load(loader, range(n)):
        assert x.numpy().shape == data_size
        assert y.numpy().shape == label_size
        count += 1
    assert count == n


def test_load_with_workers():
    data_size = (64, 64)
    label_size = (4,)
    num_worker = 4
    n = 100
    loader = RandomLoader(data_size=data_size, label_size=label_size)
    count = 0
    with load_with_workers(loader, range(n), num_worker) as generator:
        for x, y in generator:
            assert x.numpy().shape == data_size
            assert y.numpy().shape == label_size
            count += 1
    assert count == n


def test_share_memory_shim():
    data_size = (3, 512, 512)
    label_size = (4,)
    num_worker = 8
    n = 1000
    count = 0
    with SharedMemoryManager() as smm:
        loader = SharedMemoryShim(
            RandomLoader(data_size=data_size, label_size=label_size),
            smm=smm,
            memory_pool_block_count=num_worker,
        )
        with load_with_workers(loader, range(n), num_worker) as generator:
            for x, y in tqdm.tqdm(generator):
                assert x.numpy().shape == data_size
                assert y.numpy().shape == label_size
                count += 1
    assert count == n
