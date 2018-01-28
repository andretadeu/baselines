#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser


def train(env_id, num_timesteps, seed, rank):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )
    env.close()


def main():
    parser = atari_arg_parser()
    parser.add_argument('--log-dir', help='Log directory where all logs will be written', default=None)
    parser.add_argument('--log-formats', help='Formats in which the logs will be written.', default=None)
    args = parser.parse_args()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(args.log_dir, args.log_formats)
    else:
        logger.configure(log_dir=args.log_dir, format_strs=[])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, rank=rank)


if __name__ == '__main__':
    main()
