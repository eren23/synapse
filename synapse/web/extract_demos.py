#!/usr/bin/env python3
"""Extract real PushT expert demonstrations from lerobot/pusht on HuggingFace."""

import json
import os

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "trajectories")
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading lerobot/pusht dataset...")
path = hf_hub_download("lerobot/pusht", "data/chunk-000/file-000.parquet", repo_type="dataset")
table = pq.read_table(path)
df = table.to_pandas()

print(f"Total frames: {len(df)}, episodes: {df['episode_index'].nunique()}")

# Find episodes with highest success (reward)
ep_stats = df.groupby("episode_index").agg(
    length=("frame_index", "count"),
    max_reward=("next.reward", "max"),
    mean_reward=("next.reward", "mean"),
    success=("next.success", "any"),
).reset_index()

# Pick 5 good episodes: successful, reasonable length (80-200 frames)
good = ep_stats[(ep_stats["success"]) & (ep_stats["length"] >= 60) & (ep_stats["length"] <= 200)]
good = good.sort_values("mean_reward", ascending=False)

if len(good) < 3:
    # Fallback: just pick top episodes by reward
    good = ep_stats.sort_values("mean_reward", ascending=False)

selected = good.head(5)["episode_index"].tolist()
print(f"Selected episodes: {selected}")

for i, ep_idx in enumerate(selected):
    ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")

    actions = []
    agent_states = []

    for _, row in ep.iterrows():
        a = row["action"]
        s = row["observation.state"]
        actions.append([round(float(a[0]), 2), round(float(a[1]), 2)])
        agent_states.append([round(float(s[0]), 2), round(float(s[1]), 2)])

    demo = {
        "episode_index": int(ep_idx),
        "actions": actions,
        "agent_states": agent_states,
        "length": len(actions),
        "seed": int(ep_idx),  # use episode index as seed for reproducibility
    }

    out_path = os.path.join(OUT_DIR, f"demo_{i}.json")
    with open(out_path, "w") as f:
        json.dump(demo, f)

    print(f"  Demo {i}: episode {ep_idx}, {len(actions)} frames → {out_path}")

print("Done!")
