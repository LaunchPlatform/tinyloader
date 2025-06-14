import multiprocessing.context
import typing
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import tinygrad

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
    loader = RandomLoader(data_size=(64, 64), label_size=(4,))
    for x, y in load(loader, [1, 2, 3]):
        print(x.numpy(), y.numpy())


def test_load_with_workers():
    loader = RandomLoader(data_size=(64, 64), label_size=(4,))
    with load_with_workers(loader, [1, 2, 3], 4) as generator:
        for x, y in generator:
            print(x.numpy(), y.numpy())


def test_share_memory_shim():
    multiprocessing.context.set_spawning_popen("spawn")
    with SharedMemoryManager() as smm:
        loader = SharedMemoryShim(
            RandomLoader(data_size=(64, 1024, 1024, 3), label_size=(4,)), smm=smm
        )
        with load_with_workers(loader, range(1000), 4) as generator:
            for x, y in generator:
                print(x.numpy(), y.numpy())
