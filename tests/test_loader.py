import typing

import numpy as np
import tinygrad

from tinyloader.loader import load
from tinyloader.loader import Loader


class NormalLoader(Loader):
    def make_request(self, item: float) -> typing.Any:
        return item

    def load(self, request: float) -> tuple[np.typing.NDArray, ...]:
        return np.random.normal(request, 1.0, size=(3, 5)), np.random.normal(
            request, 1.0, size=(4,)
        )

    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
        x, y = response
        return tinygrad.Tensor(x).contiguous().realize(), tinygrad.Tensor(
            y
        ).contiguous().realize()


def test_simple_loader():
    loader = NormalLoader()

    for x, y in load(loader, [1, 2, 3]):
        print(x, y)
