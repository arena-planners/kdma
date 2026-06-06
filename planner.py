"""KDMA planner adapter for the arena_planners bridge.

Network output is target velocity (vx, vy) in the goal-aligned ego frame;
upstream env converts that delta to acceleration, here we use it as velocity directly.
Returns the native holonomic (vx, vy) in the world frame; the bridge applies
diff-drive projection when the target robot isn't holonomic.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from arena_planners.geometry import lookahead_on_path
from arena_planners.sdk import load_manifest, main_loop

from networks import ActorNetwork

_WEIGHTS = pathlib.Path(__file__).parent / "model" / "kdma_policy.ckpt"

_NEIGHBORHOOD_RADIUS: float = 5.0
_MAX_SPEED: float = 2.5
_LOOKAHEAD: float = 2.0

_actor: ActorNetwork | None = None


def _get_actor() -> ActorNetwork:
    global _actor
    if _actor is None:
        actor = ActorNetwork()
        ckpt = torch.load(str(_WEIGHTS), map_location="cpu", weights_only=False)
        actor_state = {
            k.removeprefix("model.actor."): v
            for k, v in ckpt["model"].items()
            if k.startswith("model.actor.")
        }
        actor.load_state_dict(actor_state, strict=True)
        actor.eval()
        _actor = actor
    return _actor


def step(features: dict) -> list[float]:
    actor = _get_actor()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None or robot_state is None:
        return [0.0, 0.0]

    px, py = float(robot_pose[0]), float(robot_pose[1])
    vx, vy = float(robot_state[2]), float(robot_state[3])

    global_plan = features.get("global_plan")
    goal_pose = features.get("goal_pose")
    target: tuple[float, float] | None = None
    if global_plan is not None and len(global_plan) > 0:
        target = lookahead_on_path(global_plan, robot_pose, lookahead=_LOOKAHEAD)
    if target is None and goal_pose is not None and len(goal_pose) >= 2:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    dpx, dpy = gx - px, gy - py
    dist = min(float(np.hypot(dpx, dpy)), _NEIGHBORHOOD_RADIUS)
    heading = float(np.arctan2(dpy, dpx))
    c, s = float(np.cos(-heading)), float(np.sin(-heading))
    v_rot_x = c * vx - s * vy
    v_rot_y = s * vx + c * vy
    agent_feat = np.array([dist, v_rot_x, v_rot_y], dtype=np.float32)

    peds = features.get("pedestrians") or []
    neighbor_feats: list[list[float]] = []
    for ped in peds:
        npx_w, npy_w = float(ped[1]) - px, float(ped[2]) - py
        nvx_w, nvy_w = float(ped[3]) - vx, float(ped[4]) - vy
        if np.hypot(npx_w, npy_w) > _NEIGHBORHOOD_RADIUS:
            continue
        neighbor_feats.append([
            c * npx_w - s * npy_w,
            s * npx_w + c * npy_w,
            c * nvx_w - s * nvy_w,
            s * nvx_w + c * nvy_w,
        ])

    if neighbor_feats:
        neighbor_tensor = torch.tensor(neighbor_feats, dtype=torch.float32)
    else:
        neighbor_tensor = torch.zeros((1, 4), dtype=torch.float32)
    agent_tensor = torch.tensor(agent_feat, dtype=torch.float32)

    with torch.no_grad():
        dist_obj = actor(agent_tensor, neighbor_tensor)
        action_rot = dist_obj.mean.cpu().numpy()

    cw, sw = float(np.cos(heading)), float(np.sin(heading))
    out_vx = cw * float(action_rot[0]) - sw * float(action_rot[1])
    out_vy = sw * float(action_rot[0]) + cw * float(action_rot[1])

    speed = float(np.hypot(out_vx, out_vy))
    if speed > _MAX_SPEED:
        scale = _MAX_SPEED / speed
        out_vx *= scale
        out_vy *= scale

    return [out_vx, out_vy]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _actor
    _actor = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
