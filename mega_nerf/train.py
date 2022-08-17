from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from mega_nerf.opts import get_opts_base
from mega_nerf.runner import Runner


def _get_train_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).train()
    else:
        Runner(hparams).train()


if __name__ == '__main__':
    main(_get_train_opts())
