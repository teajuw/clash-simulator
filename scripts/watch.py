#!/usr/bin/env python3 -u
"""Watch a trained agent play against the rule bot in ASCII.

Usage:
    python scripts/watch.py --model models/selfplay_overnight_elo1670
    python scripts/watch.py --model models/ppo_hog26_500000 --speed 0.5
    python scripts/watch.py  # random actions
"""

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print = functools.partial(print, flush=True)

import numpy as np

import clasher.config as cfg
cfg.VERBOSE = False

from clasher.env import (
    ClashRoyaleEnv, DECK, CARD_TO_IDX, ELIXIR_COST,
    N_REGIONS, region_to_position,
)

# Card display symbols (2 chars each)
CARD_SYMBOLS = {
    "HogRider": "🐗",
    "Musketeer": "🔫",
    "IceGolem": "🧊",
    "IceSpirit": "❄️",
    "Cannon": "💣",
    "Fireball": "🔥",
    "Skeletons": "💀",
    "TheLog": "🪵",
}

# ASCII card symbols (no emoji, terminal safe)
CARD_ASCII = {
    "HogRider": "HR",
    "Musketeer": "MK",
    "IceGolem": "IG",
    "IceSpirit": "IS",
    "Cannon": "CN",
    "Fireball": "FB",
    "Skeletons": "SK",
    "TheLog": "LG",
}


def render_arena(obs: np.ndarray, action: np.ndarray, player_hand: list,
                 p0_elixir: float, p1_elixir: float, game_time: float,
                 reward: float, step: int) -> str:
    """Render ASCII arena from observation vector."""

    # Parse tower HP from obs
    p0_king = obs[2] * 4824
    p0_left = obs[3] * 3052
    p0_right = obs[4] * 3052
    p1_king = obs[5] * 4824
    p1_left = obs[6] * 3052
    p1_right = obs[7] * 3052

    # Build 18x15 grid (player 0's half + bridge row)
    # We'll show a simplified 9x8 grid (2:1 compression)
    grid = [['  ' for _ in range(9)] for _ in range(16)]

    # Place towers
    grid[14][1] = f"{'L' if p0_left > 0 else 'x'}{int(p0_left/305):1d}"
    grid[14][7] = f"{'R' if p0_right > 0 else 'x'}{int(p0_right/305):1d}"
    grid[15][4] = f"{'K' if p0_king > 0 else 'x'}{int(p0_king/482):1d}"
    grid[1][1] = f"{'L' if p1_left > 0 else 'x'}{int(p1_left/305):1d}"
    grid[1][7] = f"{'R' if p1_right > 0 else 'x'}{int(p1_right/305):1d}"
    grid[0][4] = f"{'K' if p1_king > 0 else 'x'}{int(p1_king/482):1d}"

    # Place units from obs (slots 21-170, 5 features each)
    for i in range(30):
        base = 21 + i * 5
        card_id = obs[base]
        if card_id == 0 and obs[base + 1] == 0:
            continue  # empty slot
        x = obs[base + 1] * 18
        y = obs[base + 2] * 32
        hp_pct = obs[base + 3]
        owner = obs[base + 4]

        # Map to grid coordinates
        gx = min(8, max(0, int(x / 2)))
        gy = min(15, max(0, 15 - int(y / 2)))

        if hp_pct <= 0:
            continue

        # Card type from normalized ID
        card_idx = int(round(card_id * 9))
        if card_idx == 8:
            continue  # princess tower (already shown)
        if card_idx == 9:
            continue  # king tower (already shown)

        symbol = "○" if owner < 0.5 else "●"
        card_names = list(CARD_ASCII.values())
        if card_idx < len(card_names):
            symbol = card_names[card_idx][0].lower() if owner < 0.5 else card_names[card_idx][0].upper()

        if grid[gy][gx] == '  ':
            grid[gy][gx] = f"{symbol} "

    # Action display
    card_choice = int(action[0])
    region = int(action[1])
    action_str = "WAIT"
    if card_choice > 0:
        slot = card_choice - 1
        if slot < len(player_hand):
            card = player_hand[slot]
            pos = region_to_position(region)
            action_str = f"{CARD_ASCII.get(card, '??')} → ({pos.x:.0f},{pos.y:.0f})"

    # Hand display
    hand_str = " ".join(
        f"[{CARD_ASCII.get(c, '??')}:{ELIXIR_COST.get(c, 0)}]"
        for c in player_hand
    )

    # Elixir bar
    e_filled = int(p0_elixir)
    e_bar = "█" * e_filled + "░" * (10 - e_filled)

    lines = [
        f"┌───────────────────┐  Step: {step}  Time: {game_time:.0f}s",
        f"│ P1 King  {p1_king:4.0f}      │  Action: {action_str}",
    ]

    # River at row 8
    for row_idx, row in enumerate(grid):
        prefix = "│"
        suffix = "│"
        line = prefix + "".join(row) + suffix
        if row_idx == 8:
            line = "├~~~~~~~RIVER~~~~~~~┤"
        lines.append(line)

    lines.extend([
        f"│ P0 King  {p0_king:4.0f}      │",
        f"└───────────────────┘",
        f"  Elixir: [{e_bar}] {p0_elixir:.1f}",
        f"  Hand: {hand_str}",
        f"  Reward: {reward:+.3f}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--speed", type=float, default=0.3, help="Seconds between frames")
    parser.add_argument("--backend", type=str, default="auto")
    args = parser.parse_args()

    env = ClashRoyaleEnv(opponent="rule_bot", backend=args.backend)

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded model: {args.model}")
    else:
        print("No model — using random actions")

    obs, info = env.reset(seed=42)
    total_reward = 0
    step = 0

    try:
        while True:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Get game state for display
            players = env.battle.players
            p0 = players[0]

            frame = render_arena(
                obs, action,
                player_hand=p0.hand,
                p0_elixir=p0.elixir,
                p1_elixir=players[1].elixir,
                game_time=env.battle.time,
                reward=reward,
                step=step,
            )

            # Clear screen and draw
            print("\033[2J\033[H" + frame)

            if terminated:
                winner = info.get("winner")
                print(f"\n  GAME OVER — Winner: P{winner}  Total Reward: {total_reward:+.2f}")
                break

            time.sleep(args.speed)

    except KeyboardInterrupt:
        print(f"\n  Stopped at step {step}. Total reward: {total_reward:+.2f}")


if __name__ == "__main__":
    main()
