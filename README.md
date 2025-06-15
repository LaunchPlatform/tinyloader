# tinyloader
A minimalist multiprocessing data loader for tinygrad

## Why?

With [PyTorch](https://pytorch.org), you have [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) to help load data in the background.
But what about [tinygrad](https://github.com/tinygrad/tinygrad/)?
We want to load data efficiently using multiprocessing, but it turns out to be more challenging than expected.
This is mainly because pickling large amounts of data is extremely slow, often making it slower than a single-process approach.
To solve this problem, we built a simple, minimalist library to efficiently load data in background processes into shared memory, avoiding slow pickling.
