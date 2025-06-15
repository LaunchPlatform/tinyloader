# tinyloader
A minimalist multiprocessing data loader for tinygrad

## Why?

With [PyTorch](https://pytorch.org), you have [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) to help load data in the background.
But what about [tinygrad](https://github.com/tinygrad/tinygrad/)?
We want to load data efficiently using multiprocessing, but it turns out to be more challenging than expected.
This is mainly because pickling large amounts of data is extremely slow, often making it slower than a single-process approach.
To solve this problem, we built a simple, minimalist library to efficiently load data in background processes into shared memory, avoiding slow pickling.

## How?

To install tinyloader, simply run:

```bash
pip install tinyloader
```

Then, for example, you can define your own loader for loading video file like this:

```bash
import pathlib

import numpy as np
from tinyloader.loader import Loader

class VideoLoader(Loader):
    def make_request(self, item: pathlib.Path) -> typing.Any:
        # This function will be called in the main process (where load or load_with_worker is invoked)
        # It's for creating the request for loading data in a worker process.
        # Therefore, the returned value ideally should be easily pickable so that it can
        # be transferred to another process easily
        return item

    def load(self, request: pathlib.Path) -> tuple[np.typing.NDArray, ...]:
        # This function will be called from a background worker process if multiprocessing is used,
        x, y = load_video(request)
        # both x and y need to be numpy.ndarray
        return x, y

    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[tinygrad.Tensor, ...]:
        x, y = response
        # This function will be called from the main process (where load or load_with_worker is invoked)
        # before yielding back to the for loop. We need to transform the response returned from the `load`
        # method into a tinygrad Tensor. Be careful that the underlying memory buffer for the passed in
        # response could be shared memory, and it will be reused by other worker after this function
        # returns, so you need to copy the data before return
        return tinygrad.Tensor(x).contiguous().realize(), tinygrad.Tensor(
            y
        ).contiguous().realize()

```

Next, you can use the `load` function to load the data like this:

```python
from tinyloader.loader import load

video_loader = VideoLoader()
for x, y in load(loader=loader, items=["0.mp4", "1.mp4", ...]):
    # ... use x and y for training or testing
    pass

```

The `load` function runs everything in the same process without multiprocessing.
To speed up with multiprocessing, you can use `load_with_workers` instead like this:

```python
from tinyloader.loader import load_with_workers

video_loader = VideoLoader()
for x, y in load_with_workers(loader=loader, items=["0.mp4", "1.mp4", ...], num_worker=8):
    # ... use x and y for training or testing
    pass

```

When this works fine, but if you are loading huge amount of data such as video files, it could be even slower than a single process approach due to picking large `ndarray` is very slow.
To solve the problem, you can use the `SharedMemoryShim`.

```python
from multiprocessing.managers import SharedMemoryManager
from tinyloader.loader import load_with_workers
from tinyloader.loader import SharedMemoryShim

num_workers = 8

with SharedMemoryManager() as smm:
    loader = SharedMemoryShim(
        VideoLoader(),
        smm=smm,
        memory_pool_block_count=num_workers,
    )
    with load_with_workers(loader, ["0.mp4", "1.mp4", ...], num_workers) as generator:
        for x, y in generator:
            # ... use x and y for training or testing
            pass

```
