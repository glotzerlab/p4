import multiprocessing as mp
import os
# from tqdm import tqdm
# import time

# def process_item(item):
#     time.sleep(0.1)  # Simulate some work
#     return item * 2

# if __name__ == '__main__':
#     items = range(100)
#     with mp.Pool() as pool:
#         # Using imap_unordered for potentially faster results if order doesn't
#         # matter
#         results = list(
#             tqdm(pool.imap_unordered(process_item, items), total=len(items))
#         )
#     # print(results)

def func(a_item, b_chunk):
    print(f"{a_item = }")
    print(f"{b_chunk = }")
    # total = a_item
    # for b_item in b_chunk:
    #     total += b_item ** 2

def subdivide(array: list, n: int):
    """Subdivide an array into some number of chunks of consecutive items.

    Disclaimer: the body of this function was written by ChatGPT.

    Parameters
    ----------
    array : list
        The array to subdivide
    n : int
        The number of chunks to subdivide the array into.

    Returns
    -------
    subarrays
        An array of sections of the input array.
    """
    k, m = divmod(len(array), n)
    return [array[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    n_processes = os.process_cpu_count()

    a = list(range(n_processes))
    b = list(range(100))

    b_chunks = subdivide(b, n_processes)
    args = zip(a, b_chunks)

    with mp.Pool() as pool:
        pool.starmap(func, args)
        # I want each process to call func() on a single element from `a` and a subset or "chunk" from `b`