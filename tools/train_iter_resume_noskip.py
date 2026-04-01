#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train script that resumes iter-based runs without fast-forwarding dataloader.

This is a local workaround for MMEngine iter-based resume cases that can spend
too long in ``Advance dataloader N steps`` and never return to actual training.
It preserves checkpoint resume for model / optimizer / scheduler / iter state,
but skips the dataloader fast-forward loop.
"""

import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.loops import IterBasedTrainLoop

from mmseg.registry import RUNNERS


def _patch_iter_loop() -> None:
    def run_without_dataloader_advance(self):
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Resume workaround enabled: skip dataloader advance at iter={self._iter} '
                'and continue training directly.',
                logger='current',
                level=logging.WARNING)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    IterBasedTrainLoop.run = run_without_dataloader_advance


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a segmentor and resume iter-based runs without dataloader skip.')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in xxx=yyy format')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    _patch_iter_loop()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    cfg.resume = args.resume

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
