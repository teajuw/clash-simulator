from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

from ..mechanics.mechanic_base import BaseMechanic

if TYPE_CHECKING:
    from ..entities import Troop


@dataclass
class HogRiderJump(BaseMechanic):
    """Hog Rider jumps the river at a bridge over ~0.4 seconds.

    In real CR the jump is a visible arc animation lasting roughly
    10-12 game ticks (0.5s at 20fps). During the jump, the Hog
    can still be targeted and damaged by towers.
    """
    jump_duration_ms: float = 400.0  # 0.4 seconds to cross river

    # Internal state
    is_jumping: bool = False
    has_jumped: bool = False
    jump_progress: float = 0.0  # 0.0 to 1.0
    jump_start: Optional[Tuple[float, float]] = field(default=None, repr=False)
    jump_end: Optional[Tuple[float, float]] = field(default=None, repr=False)

    def on_tick(self, entity, dt_ms: int) -> None:
        from ..entities import Troop

        if not isinstance(entity, Troop) or self.has_jumped:
            return

        # Continue jump in progress
        if self.is_jumping:
            self.jump_progress += dt_ms / self.jump_duration_ms
            if self.jump_progress >= 1.0:
                # Jump complete
                entity.position.x = self.jump_end[0]
                entity.position.y = self.jump_end[1]
                self.is_jumping = False
                self.has_jumped = True
            else:
                # Interpolate position
                t = self.jump_progress
                entity.position.x = self.jump_start[0] + (self.jump_end[0] - self.jump_start[0]) * t
                entity.position.y = self.jump_start[1] + (self.jump_end[1] - self.jump_start[1]) * t
            return

        # Check if we should start jumping
        cx = entity.position.x
        cy = entity.position.y

        on_left_bridge = 2.0 <= cx < 5.0
        on_right_bridge = 13.0 <= cx < 16.0
        if not (on_left_bridge or on_right_bridge):
            return

        # Player 0 approaches river from below (y increasing toward 15)
        # Player 1 approaches river from above (y decreasing toward 16)
        should_jump = (
            (entity.player_id == 0 and 14.5 <= cy <= 15.5) or
            (entity.player_id == 1 and 16.5 <= cy <= 17.5)
        )

        if should_jump:
            self.is_jumping = True
            self.jump_progress = 0.0
            self.jump_start = (cx, cy)
            # Land on the other side of the river
            if entity.player_id == 0:
                self.jump_end = (cx, 18.0)
            else:
                self.jump_end = (cx, 14.0)
