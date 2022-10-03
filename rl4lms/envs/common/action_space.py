from typing import List
from gym.spaces.discrete import Discrete


class ActionSpace(Discrete):
    def __init__(self, actions: List[str]):
        self.actions = actions
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}
        super().__init__(len(self.actions))

    def __post_init__(self):
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}

    def action_to_ix(self, action: str) -> int:
        return self._action_to_ix[action]

    def ix_to_action(self, ix: int) -> str:
        return self._ix_to_action[ix]

    def size(self) -> int:
        return self.n

    def __repr__(self):
        return f"Discrete Action Space with {self.size()} actions: {self.actions}"
