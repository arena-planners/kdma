"""KDMA planner adapter for the arena_planners bridge.

Native rate ~8.33 Hz (1/0.12); launch with planner_rate_hz:=8.333.

The network outputs a goal-frame velocity target. Upstream integrates it as an
acceleration, but the per-step fps factors cancel, so it reduces to a direct
velocity assignment: rotate the output to world frame and clamp to MAX_SPEED.
The accumulator is fed back as the velocity feature, matching upstream's
self.velocity (commanded, not odom). Returns world-frame (vx, vy); the bridge
projects to diff-drive when the target robot is not holonomic.
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
_velocity: list[float] = [0.0, 0.0]


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
    global _velocity
    actor = _get_actor()

    robot_pose = features.get("robot_pose")
    if robot_pose is None:
        return [0.0, 0.0]

    px, py = float(robot_pose[0]), float(robot_pose[1])
    vx, vy = _velocity[0], _velocity[1]

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

    peds = features.get("pedestrians")
    if peds is None:
        peds = []
    neighbor_feats: list[list[float]] = []
    for ped in peds:
        npx_w, npy_w = float(ped[1]) - px, float(ped[2]) - py
        # relative velocity vs. accumulator velocity (matches upstream self.velocity)
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

    # rotate goal-frame velocity target to world frame (R(+heading))
    cw, sw = float(np.cos(heading)), float(np.sin(heading))
    vx_target = cw * float(action_rot[0]) - sw * float(action_rot[1])
    vy_target = sw * float(action_rot[0]) + cw * float(action_rot[1])

    _velocity[0] = vx_target
    _velocity[1] = vy_target

    speed = float(np.hypot(_velocity[0], _velocity[1]))
    if speed > _MAX_SPEED:
        scale = _MAX_SPEED / speed
        _velocity[0] *= scale
        _velocity[1] *= scale

    return [_velocity[0], _velocity[1]]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _velocity
    _velocity = [0.0, 0.0]


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
