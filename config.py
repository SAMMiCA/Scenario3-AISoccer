import argparse

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Using CPU or GPU
    parser.add_argument("--use-cpu", action="store_true", default=False)
    # Output Filename
    parser.add_argument("--commit_num", type=str, default="0", help="commit number?")
    # Environment
    # parser.add_argument("--scenario", type=str, default="simple_", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--seed", type=int, default=1, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp")
    # parser.add_argument("--update-freq", type=int, default=100, help="number of timesteps trainer should be updated ")
    parser.add_argument("--no-comm", action="store_true", default=False) # for analysis purposes
    parser.add_argument("--critic-lstm", action="store_true", default=False)
    parser.add_argument("--actor-lstm", action="store_true", default=False)
    parser.add_argument("--centralized-actor", action="store_true", default=False)
    parser.add_argument("--with-comm-budget", action="store_true", default=False)
    parser.add_argument("--analysis", type=str, default="", help="type of analysis") # time, pos, argmax
    parser.add_argument("--commit-num", type=str, default="", help="name of the experiment")
    parser.add_argument("--sync-sampling", action="store_true", default=False)
    parser.add_argument("--tracking", action="store_true", default=True)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../../examples/ai28_player/saved_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./saved_policy/", help="directory in which training state and model are loaded")
    parser.add_argument("--test-actor-q", action="store_true", default=False)
    # Evaluation
    parser.add_argument("--graph", action="store_true", default=True)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--metrics-filename", type=str, default="", help="name of metrics filename")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="../../examples/ai28_player/benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="../../examples/ai28_player/learning_curves/", help="directory where plot data is saved")
    arglist, _ = parser.parse_known_args()
    return arglist
