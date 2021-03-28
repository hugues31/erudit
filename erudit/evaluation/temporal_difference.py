from erudit import Environment

def temporal_difference(
    pi,
    env: Environment,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.3,
    n_episodes=500,
):
    pass