import abc
import collections
import contextlib
import dataclasses
import itertools
import logging
import multiprocessing
import pathlib
import typing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import tinygrad


class DataLoader(abc.ABC):
    @property
    def buf_sizes(self) -> tuple[int, ...] | dict[str, int]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_request(self, index: int) -> typing.Any:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, request: typing.Any):
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self, res):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class SharedNDArray:
    shape: tuple[int]
    dtype: np.typing.DTypeLike
    buffer: SharedMemory

    def to_ndarray(self) -> np.typing.NDArray:
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.buffer.buf,
        )


def share_ndarray(array: np.ndarray, buffer: SharedMemory) -> SharedNDArray:
    if array.nbytes != buffer.size:
        raise ValueError(
            f"Expected data ndarray size {array.nbytes} should be equal to {buffer.size}"
        )
    shared_ndarray = np.ndarray(shape=array.shape, dtype=array.dtype, buffer=buffer.buf)
    shared_ndarray[:] = array[:]
    return SharedNDArray(
        shape=shared_ndarray.shape,
        dtype=shared_ndarray.dtype,
        buffer=buffer,
    )


class SharedMemoryShim:
    def __init__(
        self,
        loader: DataLoader,
        shared_buffers: tuple[SharedMemory, ...] | dict[str, SharedMemory],
    ):
        self.loader = loader
        self.shared_buffers = shared_buffers

    def load(self, request: typing.Any):
        result = self.loader.load(request)
        if isinstance(result, tuple):
            if len(result) != self.shared_buffers:
                raise ValueError(
                    f"Expected load function result length to be {len(self.shared_buffers)} but got {len(result)} "
                    "instead"
                )
            return tuple(
                share_ndarray(array=array, buffer=shared_buffer)
                for array, shared_buffer in zip(result, self.shared_buffers)
            )
        else:
            raise ValueError(f"Unexpected load function result type {type(result)}")


@contextlib.contextmanager
def with_indexes(
    loader: DataLoader, num_worker: int, indexes: typing.Sequence[int]
) -> typing.Generator[typing.Any]:
    pool = multiprocessing.Pool(num_worker)
    with SharedMemoryManager() as smm:
        memory_pool: dict[int, list[SharedMemory]] = collections.defaultdict(list)

        def pop_buffer(size: int):
            if memory_pool[size]:
                return memory_pool[size].pop(0)
            else:
                return smm.SharedMemory(size)

        buf_sizes = loader.buf_sizes
        if isinstance(buf_sizes, tuple):
            shared_buffers = tuple(map(pop_buffer, buf_sizes))
        else:
            raise ValueError(f"Unexpected buf_sizes type {type(buf_sizes)}")

        shim = SharedMemoryShim(loader=loader)

        yield map(
            loader.post_process,
            pool.imap(loader.load, map(loader.make_request, indexes)),
        )
