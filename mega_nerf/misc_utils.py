import os

from tqdm import tqdm


def main_print(log) -> None:
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        print(log)


def main_tqdm(inner):
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        return tqdm(inner)
    else:
        return inner
