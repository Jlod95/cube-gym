"""Reward rules. Call reset(start_state) at episode start, then
reward(state) after every action."""

from cube import is_solved
from stages import stages_complete

STAGE_PAY = 0.1
SOLVE_BONUS = 1.0


class NaiveStageReward:
    """Pays STAGE_PAY whenever stages_complete goes up since the last action."""

    def reset(self, start_state):
        self.prev = stages_complete(start_state)

    def reward(self, state):
        stages = stages_complete(state)
        paid = STAGE_PAY if stages > self.prev else 0.0
        self.prev = stages
        return paid


class StaySolvedReward(NaiveStageReward):
    """Naive, but any drop in stages_complete voids all later rewards."""

    def reset(self, start_state):
        super().reset(start_state)
        self.violated = False

    def reward(self, state):
        if stages_complete(state) < self.prev:
            self.violated = True
        paid = super().reward(state)
        return 0.0 if self.violated else paid


class HighWaterReward:
    """Pays STAGE_PAY per stage above the best reached so far, plus
    SOLVE_BONUS the first time the cube is solved. Drops cost nothing."""

    def reset(self, start_state):
        self.record = stages_complete(start_state)
        self.solved_before = is_solved(start_state)

    def reward(self, state):
        stages = stages_complete(state)
        paid = 0.0
        if stages > self.record:
            paid += STAGE_PAY * (stages - self.record)
            self.record = stages
        if not self.solved_before and is_solved(state):
            paid += SOLVE_BONUS
            self.solved_before = True
        return paid
