"""Fixed-action agents for probing reward rules.

Each agent returns (start_state, moves): the cube it starts on and the
quarter turns it takes, one per action.
"""

from algorithms import ALGORITHMS, parse
from cube import SOLVED, CubeEnv, apply, is_solved
from rewards import HighWaterReward, NaiveStageReward, StaySolvedReward
from stages import stages_complete


def farming_agent():
    """Cross complete (top layer turned), then D, D', D, D', ... for 20 actions."""
    start = apply(SOLVED, "U")
    return start, ["D", "D'"] * 10


TWIST = parse(ALGORITHMS["twist_corner"])


def _play(state, moves):
    for move in moves:
        state = apply(state, move)
    return state


def method_sequence_agent():
    """Stage 7 of the method: two top corners are twisted (stages 1-6
    complete). The agent fixes them with twist_corner and U turns; the
    lower layers are scrambled along the way and restored at the end."""
    # Build the start: twist one corner 2x, U, another corner 4x, U back.
    # Six twist_corners in total leave the lower layers intact.
    start = _play(SOLVED, TWIST * 2 + ["U"] + TWIST * 4 + ["U'"])
    assert stages_complete(start) == 6 and not is_solved(start)

    # Solve: 4x on the first corner, U, 2x on the second, U back.
    moves = TWIST * 4 + ["U"] + TWIST * 2 + ["U'"]
    assert is_solved(_play(start, moves))
    return start, moves


def run_episode(agent, rule):
    """Play the agent's moves in a CubeEnv; return the per-action rewards."""
    start, moves = agent()
    env = CubeEnv()
    env.restore(start)
    rule.reset(start)
    rewards = []
    for move in moves:
        env.step(move)
        rewards.append(rule.reward(env.state))
    return rewards


RULES = {"Naive": NaiveStageReward, "StaySolved": StaySolvedReward, "HighWater": HighWaterReward}
AGENTS = {"farming_agent": farming_agent, "method_sequence_agent": method_sequence_agent}


if __name__ == "__main__":
    print(f"{'agent':<24}" + "".join(f"{name:>12}" for name in RULES))
    for agent_name, agent in AGENTS.items():
        totals = [sum(run_episode(agent, rule())) for rule in RULES.values()]
        print(f"{agent_name:<24}" + "".join(f"{t:>12.1f}" for t in totals))
