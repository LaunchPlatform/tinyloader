import abc
import collections
import contextlib
import dataclasses
import multiprocessing
import typing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import tinygrad


class DataLoader(abc.ABC):
    @property
    def buf_sizes(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_request(self, item: typing.Any) -> typing.Any:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, request: typing.Any) -> type[np.typing.NDArray]:
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self, res):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class SharedNDArray:
    shape: tuple[int, ...]
    dtype: np.typing.DTypeLike
    buffer: SharedMemory

    def to_ndarray(self) -> np.typing.NDArray:
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.buffer.buf,
        )


@dataclasses.dataclass(frozen=True)
class LoadRequestSharedBuffer:
    request: typing.Any
    buffers: tuple[SharedMemory, ...] | None


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


class SharedMemoryShim(DataLoader):
    def __init__(self, loader: DataLoader, smm: SharedMemoryManager):
        self.loader = loader
        self._buf_sizes: tuple[int, ...] | None = None
        self._smm: SharedMemoryManager = smm
        self._memory_pool: dict[int, list[SharedMemory]] = collections.defaultdict(list)

    def _pop_buf(self, size: int) -> SharedMemory:
        if self._memory_pool[size]:
            return self._memory_pool[size].pop(0)
        else:
            return self._smm.SharedMemory(size)

    def make_request(self, item: typing.Any) -> LoadRequestSharedBuffer:
        request = self.loader.make_request(item)
        buffers = None
        if self._buf_sizes is not None:
            buffers = tuple(map(self._pop_buf, self._buf_sizes))
        return LoadRequestSharedBuffer(
            request=request,
            buffers=buffers,
        )

    def load(self, request: LoadRequestSharedBuffer):
        result = self.loader.load(request.request)
        if request.buffers is None:
            # This is our first load, let's do it without the shared memory
            return result
        if isinstance(result, tuple):
            if len(result) != request.buffers:
                raise ValueError(
                    f"Expected load function result length to be {len(request.buffers)} but got {len(result)} "
                    "instead"
                )
            return tuple(
                share_ndarray(array=array, buffer=shared_buffer)
                for array, shared_buffer in zip(result, request.buffers)
            )
        else:
            raise ValueError(f"Unexpected load function result type {type(result)}")

    def post_process(self, res):
        pass


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
