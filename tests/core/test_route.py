"""restorebot.core.navigation の止め位置・経路計画・経路追従・撮影時の後退（設計書 8 章、#16）の単体テスト。"""

import math

import numpy as np
import pytest

from restorebot.core.config import M1_BASE_POSE, NAV, REACH, ROBOT
from restorebot.core.navigation import (
    PathFollower,
    approach_pose,
    build_grid,
    in_reach,
    plan_path,
    plan_route,
    retreat_pose,
    robot_clearance,
    robot_corners,
    wrap_angle,
)
from restorebot.core.scene import HOME_V2
from restorebot.core.types import Pose2D

SCENE = HOME_V2
DESK = SCENE.surfaces["desk"]
SIDE = SCENE.surfaces["side_table"]
GRID = build_grid(SCENE, ROBOT.circumradius + NAV.inflation_margin, NAV.resolution)


def _free(xy):
    return not GRID.occupied_at(xy)


# ---- 台車の外形と、家具・壁との距離 ----


def test_circumradius():
    assert ROBOT.circumradius == pytest.approx(math.hypot(0.47, 0.40))


def test_robot_corners_follow_pose():
    corners = robot_corners(Pose2D(1.0, 2.0, math.pi / 2), ROBOT)
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    assert (min(xs), max(xs)) == pytest.approx((1.0 - 0.40, 1.0 + 0.40))  # 90° 回ると左右が x 方向
    assert (min(ys), max(ys)) == pytest.approx((2.0 - 0.47, 2.0 + 0.47))


def test_clearance_at_m1_pose_is_the_furniture_gap():
    assert robot_clearance(M1_BASE_POSE, SCENE, ROBOT) == pytest.approx(ROBOT.furniture_gap, abs=1e-6)


def test_clearance_negative_when_overlapping_or_outside_room():
    assert robot_clearance(Pose2D(0.8, 0.85, 0.0), SCENE, ROBOT) < 0  # 机に重なる
    assert robot_clearance(Pose2D(1.9, 0.0, 0.0), SCENE, ROBOT) < 0  # 壁の外にはみ出す


# ---- 止め位置 ----


def _check_approach(pose, target_xy, surface):
    assert pose is not None
    assert robot_clearance(pose, SCENE, ROBOT) == pytest.approx(ROBOT.furniture_gap, abs=1e-6)  # 家具との隙間 10cm
    assert in_reach(pose, target_xy, REACH)
    # 家具の方を向いている：台車の前方 (隙間 + 少し) は天板の上
    d = ROBOT.center_to_edge + 0.05
    fx, fy = pose.x + d * math.cos(pose.yaw), pose.y + d * math.sin(pose.yaw)
    assert SCENE.surface_at((fx, fy)) is surface


def test_approach_desk_front_from_front():
    target = (0.75, 0.72)
    pose = approach_pose(target, "desk", SCENE, start=Pose2D(0.75, -0.6, 0.0))
    _check_approach(pose, target, DESK)
    assert pose.yaw == pytest.approx(math.pi / 2)
    assert pose.x == pytest.approx(0.75)  # 物体が正面に来る


def test_approach_side_table_only_from_east():
    for target in [(-1.45, -0.55), (-1.45, -0.05)]:
        pose = approach_pose(target, "side_table", SCENE, start=M1_BASE_POSE)
        _check_approach(pose, target, SIDE)
        assert abs(wrap_angle(pose.yaw - math.pi)) < 1e-9  # -x を向く（棚の東側から）


def test_approach_chooses_side_nearest_to_start():
    target = (0.45, 0.72)  # 机の左寄り：正面からも左側からも届く
    from_left = approach_pose(target, "desk", SCENE, start=Pose2D(-1.0, 0.7, 0.0))
    from_front = approach_pose(target, "desk", SCENE, start=Pose2D(0.45, -0.7, 0.0))
    _check_approach(from_left, target, DESK)
    _check_approach(from_front, target, DESK)
    assert from_left.yaw == pytest.approx(0.0)  # 左側（-x）から +x を向く
    assert from_front.yaw == pytest.approx(math.pi / 2)


def test_approach_shifts_laterally_near_wall():
    # 壁際の家具の端の物体：正面にそろえると台車が南の壁にはみ出すので、横にずらす（届く範囲の中で）
    from dataclasses import replace

    from restorebot.core.scene import Placement, Surface

    table = Surface(center=(-1.0, -1.2), size=(0.4, 0.6), key="table", z=0.5, front="+x")
    scene = replace(
        SCENE,
        surfaces={"table": table},
        tidy={k: Placement("table", -1.0, -1.2) for k in SCENE.tidy},
    )
    target = (-0.95, -1.38)
    pose = approach_pose(target, "table", scene, start=Pose2D(0.5, 0.0, 0.0))
    assert pose is not None
    assert pose.y > target[1]  # 台車は物体の正面より北にずれる（南の壁に当たらない）
    assert robot_clearance(pose, scene, ROBOT) == pytest.approx(ROBOT.furniture_gap, abs=1e-6)
    assert in_reach(pose, target, REACH)


def test_approach_returns_none_when_unreachable():
    # 机の奥（手前の縁から 50cm）は、どの辺からも届かない
    assert approach_pose((0.8, 1.05), "desk", SCENE, start=M1_BASE_POSE) is None


# ---- 格子地図と A* ----


def test_grid_inflates_walls_and_furniture():
    r = ROBOT.circumradius + NAV.inflation_margin
    assert _free((0.0, -0.5))  # 部屋の中央の南
    assert not _free((0.8, 0.85))  # 机
    assert not _free((0.8, DESK.center[1] - DESK.size[1] / 2 - r + 0.05))  # 机から膨張の内側
    assert _free((0.8, DESK.center[1] - DESK.size[1] / 2 - r - 0.05))  # 膨張のすぐ外
    assert not _free((0.0, -1.5 + r - 0.05))  # 南の壁の近く
    assert not _free((5.0, 0.0))  # 部屋の外


def test_plan_path_avoids_inflated_obstacles():
    path = plan_path(GRID, (0.75, -0.42), (-0.33, -0.30))
    assert path is not None and len(path) >= 2
    assert path[0] == pytest.approx((0.75, -0.42), abs=GRID.resolution)
    assert path[-1] == pytest.approx((-0.33, -0.30), abs=GRID.resolution)
    for a, b in zip(path, path[1:], strict=False):  # 間引いた後の線分も、膨張した障害物を通らない
        for s in np.linspace(0, 1, 20):
            assert _free((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))


def test_plan_path_none_when_goal_blocked():
    assert plan_path(GRID, (0.0, -0.5), (0.8, 0.85)) is None


# ---- 経路（下がる・A*・寄せる）----


def _check_route(route, start, goal):
    assert route is not None
    assert route[-1] == goal
    pts = [start, *route]
    for a, b in zip(pts, pts[1:], strict=False):
        for s in np.linspace(0, 1, 30):
            p = Pose2D(a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), b.yaw if s > 0 else a.yaw)
            # 台車の外形は、どこでも家具・壁から min_gap 以上（膨張の外では向きによらず当たらない）
            if not _free((p.x, p.y)):
                assert robot_clearance(p, SCENE, ROBOT) >= NAV.min_gap - 1e-9


def test_route_desk_to_side_table_and_back():
    goal = approach_pose((-1.45, -0.10), "side_table", SCENE, start=M1_BASE_POSE)
    route = plan_route(SCENE, GRID, M1_BASE_POSE, goal)
    _check_route(route, M1_BASE_POSE, goal)
    # 最初は向きを変えずに下がる、最後は到着時の向きでまっすぐ寄せる
    assert route[0].yaw == pytest.approx(M1_BASE_POSE.yaw)
    assert route[0].y < M1_BASE_POSE.y and route[0].x == pytest.approx(M1_BASE_POSE.x)
    assert route[-2].yaw == pytest.approx(goal.yaw)
    back = plan_route(SCENE, GRID, goal, M1_BASE_POSE)
    _check_route(back, goal, M1_BASE_POSE)


def test_route_between_free_poses_is_direct():
    start, goal = Pose2D(-0.2, -0.5, 0.0), Pose2D(0.5, -0.5, math.pi / 2)
    route = plan_route(SCENE, GRID, start, goal)
    assert route[-1] == goal and len(route) <= 2


def test_route_none_when_goal_unreachable():
    assert plan_route(SCENE, GRID, M1_BASE_POSE, Pose2D(0.8, 0.85, 0.0)) is None


# ---- 撮影時の後退 ----


def test_retreat_pose_backs_off_straight():
    p = retreat_pose(M1_BASE_POSE, 0.2, SCENE)
    assert p == Pose2D(M1_BASE_POSE.x, pytest.approx(M1_BASE_POSE.y - 0.2), M1_BASE_POSE.yaw)


def test_retreat_pose_none_when_hitting_wall():
    assert retreat_pose(Pose2D(0.0, -0.95, math.pi / 2), 0.2, SCENE) is None


# ---- 経路追従 ----


def _simulate(route, start, dt=1 / 60, t_max=60.0, **kw):
    f = PathFollower(route, **kw)
    pose, t, vmax, wmax = start, 0.0, 0.0, 0.0
    while t < t_max:
        cmd = f.command(pose)
        if cmd is None:
            return pose, t, vmax, wmax
        vx, vy, wz = cmd
        vmax, wmax = max(vmax, math.hypot(vx, vy)), max(wmax, abs(wz))
        pose = Pose2D(pose.x + vx * dt, pose.y + vy * dt, wrap_angle(pose.yaw + wz * dt))
        t += dt
    raise AssertionError("到着しなかった")


def test_follower_reaches_goal_within_tolerance_and_limits():
    goal = approach_pose((-1.45, -0.10), "side_table", SCENE, start=M1_BASE_POSE)
    route = plan_route(SCENE, GRID, M1_BASE_POSE, goal)
    pose, t, vmax, wmax = _simulate(route, M1_BASE_POSE)
    assert math.hypot(pose.x - goal.x, pose.y - goal.y) <= 0.01
    assert abs(wrap_angle(pose.yaw - goal.yaw)) <= 0.017
    assert vmax <= 0.5 + 1e-9 and wmax <= 1.0 + 1e-9
    assert t < 30.0


def test_follower_stops_rotating_before_final_approach():
    # 寄せの開始点では、到着時の向きにそろってから寄せる（家具の近くで回らない）
    goal = approach_pose((-1.45, -0.10), "side_table", SCENE, start=M1_BASE_POSE)
    route = plan_route(SCENE, GRID, M1_BASE_POSE, goal)
    f = PathFollower(route)
    pose, dt = M1_BASE_POSE, 1 / 60
    for _ in range(int(60 / dt)):
        cmd = f.command(pose)
        if cmd is None:
            break
        if f.index == len(route) - 1:  # 最後の寄せの区間
            assert abs(wrap_angle(pose.yaw - goal.yaw)) <= f.waypoint_yaw_tol + 0.02
        vx, vy, wz = cmd
        pose = Pose2D(pose.x + vx * dt, pose.y + vy * dt, wrap_angle(pose.yaw + wz * dt))


def test_follower_single_goal():
    pose, _, _, _ = _simulate([Pose2D(0.3, 0.0, 0.5)], Pose2D(0.0, 0.0, 0.0))
    assert (pose.x, pose.y) == pytest.approx((0.3, 0.0), abs=0.01)
