#!/usr/bin/env python3 -u
"""AlphaStar-style league training with 3 agent types.

Launches 3 parallel processes:
  - Main Agent: trains against full pool, focused play
  - Main Exploiter: trains ONLY against latest Main Agent, finds weaknesses
  - League Exploiter: trains against random pool members, diverse strategies

All share a single snapshots/ directory.

Usage:
    python scripts/train_league.py --timesteps 50000000
    python scripts/train_league.py --timesteps 50000000 --resume models/my_model
"""

import argparse
import functools
import os
import signal
import subprocess
import sys
import time

print = functools.partial(print, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume main agent from this model")
    parser.add_argument("--snapshot-dir", type=str, default="snapshots")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    # Ensure snapshot dir exists
    os.makedirs(args.snapshot_dir, exist_ok=True)

    print("=" * 65)
    print("  CLASH ROYALE RL — LEAGUE TRAINING (AlphaStar-style)")
    print("=" * 65)
    print(f"  Total steps per agent: {args.timesteps:,}")
    print(f"  Snapshot pool: {args.snapshot_dir}/")
    print(f"  Save every: {args.save_every} episodes")
    print()
    print("  MAIN  — best overall player, PFSP vs pool, ent=0.025")
    print("  MMAX  — minimax exploiter, uses main's critic, ent=0.03")
    print("  XPLR  — entropy explorer, high→low schedule, ent=0.10→0.02")
    print("  LEAG  — league generalist, uniform random pool, ent=0.05")
    print("=" * 65)
    print()

    # Build commands for each agent
    base_cmd = [
        sys.executable, "-u",
        os.path.join(os.path.dirname(__file__), "train_league_agent.py"),
    ]

    agents = [
        {
            "name": "MAIN",
            "role": "main",
            "ent_coef": 0.025,
            "seed": 42,
            "log_prefix": "[MAIN]",
            "resume": args.resume,
        },
        {
            "name": "EXPLOITER",
            "role": "minimax_exploiter",
            "ent_coef": 0.03,
            "seed": 123,
            "log_prefix": "[MMAX]",
            "resume": None,
        },
        {
            "name": "EXPLORER",
            "role": "entropy_explorer",
            "ent_coef": 0.10,
            "seed": 789,
            "log_prefix": "[XPLR]",
            "resume": None,
        },
        {
            "name": "LEAGUE",
            "role": "league_exploiter",
            "ent_coef": 0.05,
            "seed": 456,
            "log_prefix": "[LEAG]",
            "resume": None,
        },
    ]

    processes = []
    log_files = []

    for i, agent in enumerate(agents):
        log_path = f"training_{agent['role']}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)

        cmd = base_cmd + [
            "--role", agent["role"],
            "--ent-coef", str(agent["ent_coef"]),
            "--seed", str(agent["seed"]),
            "--timesteps", str(args.timesteps),
            "--snapshot-dir", args.snapshot_dir,
            "--save-every", str(args.save_every),
            "--log-every", str(args.log_every),
            "--log-prefix", agent["log_prefix"],
        ]
        if agent["resume"]:
            cmd += ["--resume", agent["resume"]]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        processes.append(proc)
        print(f"  Started {agent['name']} (PID {proc.pid}) → {log_path}")

    print()
    print("  All 3 agents running. Logs:")
    print("    tail -f training_main.log")
    print("    tail -f training_main_exploiter.log")
    print("    tail -f training_league_exploiter.log")
    print()
    print("  Combined view:")
    print("    tail -f training_*.log")
    print()
    print("  Press Ctrl+C to stop all agents.")
    print()

    # Stream all outputs interleaved
    import selectors
    sel = selectors.DefaultSelector()
    for i, proc in enumerate(processes):
        sel.register(proc.stdout, selectors.EVENT_READ, data=i)

    agent_names = ["MAIN", "EXPL", "LEAG"]
    last_ep_group = [0, 0, 0]  # track which epoch group each agent last printed
    lines_in_group = 0

    try:
        active = len(processes)
        while active > 0:
            events = sel.select(timeout=1.0)
            for key, _ in events:
                idx = key.data
                line = key.fileobj.readline()
                if line:
                    line = line.rstrip()
                    # Write to individual log file
                    log_files[idx].write(line + "\n")
                    log_files[idx].flush()

                    # Detect epoch from line (ep=XXX)
                    import re
                    ep_match = re.search(r'ep=(\d+)', line)
                    if ep_match:
                        ep = int(ep_match.group(1))
                        if ep > last_ep_group[idx]:
                            last_ep_group[idx] = ep
                            lines_in_group += 1
                            # Add blank line after all 3 agents report same epoch
                            if lines_in_group >= 3:
                                lines_in_group = 0

                    # Print to terminal
                    print(f"{line}")
                    if lines_in_group == 0 and ep_match:
                        print()  # blank line between groups
                else:
                    sel.unregister(key.fileobj)
                    active -= 1

    except KeyboardInterrupt:
        print("\n  Stopping all agents...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait(timeout=10)
        print("  All agents stopped.")

    finally:
        for lf in log_files:
            lf.close()

    # Print final status
    print()
    print("=" * 65)
    print("  LEAGUE TRAINING COMPLETE")
    for i, proc in enumerate(processes):
        status = "OK" if proc.returncode == 0 else f"exit={proc.returncode}"
        print(f"  {agents[i]['name']}: {status}")
    print("=" * 65)


if __name__ == "__main__":
    main()
