from gym.envs.registration import registry, register, make, spec

# Sliding Puzzle
# ----------------------------------------
register(
        id='SlidingPuzzle-v0',
        entry_point='gym_sliding_puzzle.envs:SlidingPuzzleEnv',
        max_episode_steps=1000,
        )
