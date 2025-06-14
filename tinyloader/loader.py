import abc
import collections
import contextlib
import dataclasses
import functools
import itertools
import multiprocessing
import typing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import tinygrad


class Loader(abc.ABC):
    @abc.abstractmethod
    def make_request(self, item: typing.Any) -> typing.Any:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, request: typing.Any) -> tuple[np.typing.NDArray, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
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


class SharedMemoryShim(Loader):
    def __init__(self, loader: Loader, smm: SharedMemoryManager):
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

    def post_process(
        self, response: tuple[np.typing.NDArray | SharedNDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
        if any(map(lambda item: isinstance(item, np.ndarray), response)):
            self._buf_sizes = tuple(map(len, response))
            return self.loader.post_process(response)
        new_resp = []
        for shared in response:
            array = shared.to_ndarray()
            new_resp.append(array)
            self._memory_pool[array.nbytes].append(shared.buffer)
        return self.loader.post_process(tuple(new_resp))

    def __reduce__(self):
        return self.__class__, (self.loader, None), None


def load(loader: Loader, items: typing.Sequence[typing.Any]):
    yield from map(
        loader.post_process, map(loader.load, map(loader.make_request, items))
    )


@contextlib.contextmanager
def load_with_workers(
    loader: Loader,
    items: typing.Sequence[typing.Any],
    num_worker: int | None = None,
) -> typing.Generator[typing.Any]:
    with multiprocessing.Pool(num_worker) as pool:
        items_iter = iter(items)

        def generate():
            # Load first item without multiprocessing to get the buffer size in case the shared memory loader is
            # used
            yield from load(loader, [next(items_iter)])
            yield from map(
                loader.post_process,
                pool.imap(loader.load, map(loader.make_request, items_iter)),
            )

        yield generate()
