import abc
import contextlib
import dataclasses
import logging
import multiprocessing
import queue
import typing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import tinygrad

logger = logging.getLogger(__name__)


class Loader(abc.ABC):
    @abc.abstractmethod
    def make_request(self, item: typing.Any) -> typing.Any:
        """Called to make data loading request to potentially passing to the workers. Ideally the return value should
        be easily pickable otherwise it might be very slow.

        :param item: The item to generate the loading request for
        :return: a pickable value for the worker process or the current process to load
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, request: typing.Any) -> tuple[np.typing.NDArray, ...]:
        """Called to load data for the given item. Potentially called from a worker process.

        :param request: Request for loading the data
        :return: The loaded data, should be a tuple of numpy's ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
        """Called to convert numpy's ndarray returned from the `load` method into tinygrad's Tensor for training or
        testing purpose. This method will be called from the process which invokes the loading generator.

        :param response: Response ndarray values returned by the `load` method
        :return: A tuple of tinygrad Tensor for training / testing or other purpose
        """
        raise NotImplementedError

    def shutdown(self):
        """Called to shutdown resources associated with the loader. Like, abort async operations or release files and
        etc.

        """


@dataclasses.dataclass(frozen=True)
class SharedBuffer:
    index: int
    view: slice
    buffer: SharedMemory
    actual_block_size: int

    @property
    def buf(self) -> memoryview:
        return self.buffer.buf[self.view]


@dataclasses.dataclass(frozen=True)
class SharedNDArray:
    shape: tuple[int, ...]
    dtype: np.typing.DTypeLike
    buffer: SharedBuffer

    def to_ndarray(self) -> np.typing.NDArray:
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.buffer.buf,
        )


@dataclasses.dataclass(frozen=True)
class LoadRequestSharedBuffer:
    request: typing.Any
    buffers: tuple[SharedBuffer, ...] | None


def share_ndarray(array: np.ndarray, buffer: SharedBuffer) -> SharedNDArray:
    if array.nbytes != buffer.actual_block_size:
        raise ValueError(
            f"Expected data ndarray size {array.nbytes} should be equal to {buffer.actual_block_size}"
        )
    shared_ndarray = np.ndarray(shape=array.shape, dtype=array.dtype, buffer=buffer.buf)
    shared_ndarray[:] = array[:]
    return SharedNDArray(
        shape=shared_ndarray.shape,
        dtype=shared_ndarray.dtype,
        buffer=buffer,
    )


class MemoryPool:
    def __init__(
        self,
        smm: SharedMemoryManager,
        block_size: int,
        block_count: int,
        alignment_size: int = 8,
    ):
        self.actual_block_size = block_size
        self.block_size = max(self.actual_block_size, alignment_size)
        self.block_count = block_count
        self._shared_memory = smm.SharedMemory(self.block_size * self.block_count)
        self._queue = queue.Queue(block_count)
        for i in range(block_count):
            self._queue.put(i)

    def pop(self) -> SharedBuffer:
        index = self._queue.get()
        offset = index * self.block_size
        shared_buffer = SharedBuffer(
            index=index,
            view=slice(offset, offset + self.block_size),
            buffer=self._shared_memory,
            actual_block_size=self.actual_block_size,
        )
        logger.debug("Pop shared buffer %s", shared_buffer)
        return shared_buffer

    def push(self, shared_buffer: SharedBuffer):
        if shared_buffer.actual_block_size != self.actual_block_size:
            raise ValueError(
                f"Push shared buffer with a wrong block size back to a wrong memory pool, expected {self.actual_block_size} but got {shared_buffer.actual_block_size}"
            )
        self._queue.put(shared_buffer.index)
        logger.debug("Pushed shared buffer %s", shared_buffer)

    def shutdown(self):
        self._queue.shutdown(immediate=True)


class SharedMemoryShim(Loader):
    def __init__(
        self, loader: Loader, smm: SharedMemoryManager, memory_pool_block_count: int
    ):
        self.loader = loader
        self._buf_sizes: tuple[int, ...] | None = None
        self._smm: SharedMemoryManager = smm
        self._memory_pools: dict[int, MemoryPool] = {}
        self._memory_pool_block_count = memory_pool_block_count

    def _pop_buf(self, size: int) -> SharedBuffer:
        pool = self._memory_pools[size]
        shared_buffer = pool.pop()
        logger.debug("Pop shared buffer %s from pool", shared_buffer)
        return shared_buffer

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
            if len(result) != len(request.buffers):
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
            self._buf_sizes = tuple(map(lambda item: item.nbytes, response))
            for item in response:
                self._memory_pools[item.nbytes] = MemoryPool(
                    smm=self._smm,
                    block_size=item.nbytes,
                    block_count=self._memory_pool_block_count,
                )
            return self.loader.post_process(response)
        new_resp = []
        for shared in response:
            array = shared.to_ndarray()
            new_resp.append(array)
            self._memory_pools[shared.buffer.actual_block_size].push(shared.buffer)
            logger.debug("Return shared buffer %s to pool", shared.buffer)
        return self.loader.post_process(tuple(new_resp))

    def __reduce__(self):
        # avoid pickling SharedMemoryManager, only care about the underlying loader in `load` method anyway
        return self.__class__, (self.loader, None, 0), None

    def shutdown(self):
        for mem_pool in self._memory_pools.values():
            mem_pool.shutdown()


def load(
    loader: Loader, items: typing.Sequence[typing.Any]
) -> typing.Generator[tuple[tinygrad.Tensor, ...], None, None]:
    """Load data immediately in current process/thread

    :param loader: the `Loader` to load data from
    :param items: sequence or generator providing items for loader to load
    :return: A generator for yielding loaded data
    """
    yield from map(
        loader.post_process, map(loader.load, map(loader.make_request, items))
    )


@contextlib.contextmanager
def load_with_workers(
    loader: Loader,
    items: typing.Sequence[typing.Any],
    num_worker: int | None = None,
    shared_memory_enabled: bool = False,
    memory_pool_block_count: int | None = None,
) -> typing.Generator[
    typing.Generator[tuple[tinygrad.Tensor, ...], None, None], None, None
]:
    """
    Load data with background workers

    :param loader: The `Loader` to load data from
    :param items: sequence or generator providing items for loader to load
    :param num_worker: number of workers to run in the background
    :param shared_memory_enabled: should we load data into shared memory to speed it up or not
    :param memory_pool_block_count: count of memory blocks for shared memory pool, if None is provided, num_worker
                                    will be used
    :return: A generator for yielding loaded data
    """
    shared_memory_ctx = contextlib.nullcontext()
    if shared_memory_enabled:
        shared_memory_ctx = SharedMemoryManager()
    with shared_memory_ctx as smm, multiprocessing.Pool(num_worker) as pool:
        items_iter = iter(items)

        actual_loader = loader
        if shared_memory_enabled:
            actual_loader = SharedMemoryShim(
                loader=loader,
                smm=smm,
                memory_pool_block_count=memory_pool_block_count
                if memory_pool_block_count is not None
                else num_worker,
            )

        def generate() -> typing.Generator[tuple[tinygrad.Tensor, ...], None, None]:
            # Load first item without multiprocessing to get the buffer size in case the shared memory loader is
            # used
            yield from load(actual_loader, [next(items_iter)])
            yield from map(
                actual_loader.post_process,
                pool.imap(
                    actual_loader.load, map(actual_loader.make_request, items_iter)
                ),
            )

        try:
            yield generate()
        finally:
            actual_loader.shutdown()
