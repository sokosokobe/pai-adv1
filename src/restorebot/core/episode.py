"""全体の制御：1 回の実行（エピソード）の状態遷移と失敗の扱い（設計書 2 章）。

Protocol（WorldState・CeilingCamera・Robot・Simulation・VlmClient）だけを使い、sim の実装を差し込む。
単体テストではダミーを差し込む。今は M1（台車を動かさない）の範囲。M2 の移動は #17 で足す。

M1 の流れ：
    撮影姿勢で今の写真 → 隠れの判定（撮り直しは max_retakes 回まで。それでも隠れていればその写真で進める）
    → VLM → タスクの列（近い順）→ タスクごとに「待機姿勢 → 掴む（失敗したら max_grasp_retries 回やり直す）→ 置く」
    → 撮影姿勢で片付けた後の写真 → 待機姿勢
写真を撮るときは、アームを台車の後ろへ回した撮影姿勢にして、天板を隠さない（#14。M1 では台車を下げない）。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from restorebot.core.config import NAV, ROBOT
from restorebot.core.navigation import approach_pose, build_grid, in_reach, plan_route, retreat_pose
from restorebot.core.occlusion import needs_retake, robot_mask
from restorebot.core.perception import EstimatedWorld
from restorebot.core.report import EpisodeLog, TaskRecord, detection_to_dict
from restorebot.core.scene import SceneSpec
from restorebot.core.tasks import plan_tasks, to_instructions
from restorebot.core.types import (
    Action,
    CeilingCamera,
    Observation,
    Robot,
    Simulation,
    Status,
    StepResult,
    WorldState,
)
from restorebot.core.vlm.client import VlmClient
from restorebot.core.vlm.detect import detect_moves


@dataclass
class EpisodeConfig:
    milestone: str = "m1"
    max_retakes: int = 2
    max_grasp_retries: int = 1
    vlm_retries: int = 2
    verify: bool = False  # 片付けた後の確認と、残った物体の 1 回のやり直し（設計書 10 章）
    action_timeout_s: float = 60.0  # 1 つの動作（掴む・置く・姿勢の変更）の時間切れ（シミュレーションの時間）
    dt: float = 1.0 / 60.0  # Action.step に渡す時間（物理の 1 ステップ）


UNKNOWN_POSITION = "物体の位置が分からない（天板の上にない、または画像から推定できない）"


def encode_png(rgb) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


class Episode:
    def __init__(
        self,
        scene: SceneSpec,
        world: WorldState,
        camera: CeilingCamera,
        robot: Robot,
        vlm: VlmClient,
        sim: Simulation,
        config: EpisodeConfig | None = None,
        *,
        meta: dict | None = None,
        truth: WorldState | None = None,
    ) -> None:
        """world はロボットを動かすための世界（真値、または推定版の EstimatedWorld）。

        truth は記録（置いた後の位置、推定の誤差）のための真値。省略すると world を使う。
        """
        if (config or EpisodeConfig()).milestone not in ("m1", "m2"):
            raise ValueError(f"未知の milestone です: {config.milestone}")
        self.scene = scene
        self.world = world
        self.truth = truth or world
        self.estimated = isinstance(world, EstimatedWorld)
        self.camera = camera
        self.robot = robot
        self.vlm = vlm
        self.sim = sim
        self.config = config or EpisodeConfig()
        self.meta = dict(meta or {})
        self.reference: Observation | None = None  # いつもの写真（teach() で撮る、または外から与える）
        self.t = 0.0  # シミュレーションの経過時間 [s]
        self.log = EpisodeLog()
        self.moves = self.config.milestone == "m2"  # 台車を動かすか
        self.grid = build_grid(scene, ROBOT.circumradius + NAV.inflation_margin, NAV.resolution) if self.moves else None

    def teach(self) -> Observation:
        """いつもの写真を撮る（ロボットが写らない状態で呼ぶ）。"""
        self.reference = self.camera.capture()
        return self.reference

    # ------------------------------------------------------------
    def _run(self, action: Action) -> StepResult:
        """Action を終わるまで 1 ステップずつ進める。action_timeout_s を超えたら失敗。"""
        action.start()
        elapsed = 0.0
        while elapsed < self.config.action_timeout_s:
            result = action.step(self.config.dt)
            if result.status is not Status.RUNNING:
                return result
            dt = self.sim.step()
            elapsed += dt
            self.t += dt
        return StepResult(Status.FAILED, f"{action.name}：{self.config.action_timeout_s:.0f} 秒で終わらなかった")

    def _arm_to(self, posture: str) -> None:
        r = self._run(self.robot.arm_to(posture))
        if r.status is not Status.SUCCEEDED:
            self.log.event(self.t, "arm_failed", r.reason or "", posture=posture)

    def _capture_current(self) -> Observation:
        """撮影姿勢で今の写真を撮り、ロボットが天板を隠していれば撮り直す（max_retakes 回まで）。"""
        cfg = self.config
        for attempt in range(cfg.max_retakes + 1):
            self._arm_to("photo")
            obs = self.camera.capture()
            hidden = needs_retake(robot_mask(obs.depth, self.reference.depth), self.scene)
            if not hidden:
                return obs
            if attempt < cfg.max_retakes:
                retreated = False
                if self.moves:  # M2：まっすぐ下がって撮り直す（設計書 7.2・8.4）
                    back = retreat_pose(self.world.robot_pose(), NAV.shoot_retreat, self.scene)
                    if back is not None:
                        r = self._run(self.robot.navigate([back]))
                        retreated = r.status is Status.SUCCEEDED
                self.log.event(
                    self.t,
                    "retake",
                    "ロボットが天板の物体を置ける範囲を隠している",
                    attempt=attempt + 1,
                    retreated=retreated,
                )
            else:
                self.log.event(self.t, "still_hidden", f"{cfg.max_retakes} 回撮り直しても隠れている。この写真で進める")
        return obs

    def _go_to(self, xy, surface: str | None) -> str | None:
        """M2：家具 surface の上の点 xy を掴む・置くための止め位置へ移動する。失敗の理由を返す（成功なら None）。

        今の位置からすでに楽に届くなら動かない。
        """
        base = self.world.robot_pose()
        if in_reach(base, xy):
            return None
        if surface is None:
            return "物体が天板の上にない"
        goal = approach_pose(xy, surface, self.scene, start=base)
        if goal is None:
            return "止め位置がない（届かない）"
        route = plan_route(self.scene, self.grid, base, goal)
        if route is None:
            return "経路がない"
        r = self._run(self.robot.navigate(route))
        if r.status is not Status.SUCCEEDED:
            return r.reason or "移動に失敗"
        self.log.event(self.t, "arrived", "", pose=[goal.x, goal.y, goal.yaw])
        return None

    def _do_task_m2(self, task) -> TaskRecord:
        rec = TaskRecord(task.key, tuple(task.place_position), tuple(task.pick.position), started_s=self.t)
        for _ in range(self.config.max_grasp_retries + 1):
            self._arm_to("standby")
            obj = self.world.object_state(task.key)  # 掴み損ねて物体が動いたかもしれないので、毎回取り直す
            if obj.surface is None:
                rec.status, rec.reason = "skipped", UNKNOWN_POSITION
                return rec
            reason = self._go_to(obj.position[:2], obj.surface)
            if reason is not None:
                rec.status, rec.reason = "skipped", reason
                return rec
            rec.pick_attempts += 1
            r = self._run(self.robot.pick(obj))
            if r.status is Status.SUCCEEDED:
                break
            rec.reason = r.reason
            self.log.event(self.t, "pick_failed", r.reason or "", key=task.key, attempt=rec.pick_attempts)
        else:
            self._arm_to("standby")
            rec.status = "skipped"
            return rec
        reason = self._go_to(task.place_position[:2], task.place_surface)  # 物を持ったまま運ぶ（#15）
        if reason is not None:
            rec.status, rec.reason = "failed", reason
            return rec
        r = self._run(self.robot.place(task.place_position, task.place_yaw))
        if r.status is Status.SUCCEEDED:
            rec.status, rec.reason = "succeeded", None
        else:
            rec.status, rec.reason = "failed", r.reason
            self._arm_to("standby")
        return rec

    def _do_task(self, task, base) -> TaskRecord:
        if self.moves:
            return self._do_task_m2(task)
        rec = TaskRecord(task.key, tuple(task.place_position), tuple(task.pick.position), started_s=self.t)
        pick_xy, place_xy = task.pick.position[:2], task.place_position[:2]
        if task.pick.surface is None:
            rec.status, rec.reason = "skipped", UNKNOWN_POSITION
            return rec
        if not (in_reach(base, pick_xy) and in_reach(base, place_xy)):
            rec.status, rec.reason = "skipped", "M1 の止め位置から届かない"
            return rec
        for _ in range(self.config.max_grasp_retries + 1):
            self._arm_to("standby")
            rec.pick_attempts += 1
            r = self._run(self.robot.pick(self.world.object_state(task.key)))
            if r.status is Status.SUCCEEDED:
                break
            rec.reason = r.reason
            self.log.event(self.t, "pick_failed", r.reason or "", key=task.key, attempt=rec.pick_attempts)
        else:
            self._arm_to("standby")
            rec.status = "skipped"
            return rec
        r = self._run(self.robot.place(task.place_position, task.place_yaw))
        if r.status is Status.SUCCEEDED:
            rec.status, rec.reason = "succeeded", None
        else:
            rec.status, rec.reason = "failed", r.reason
            self._arm_to("standby")
        return rec

    def _estimate_errors(self, task) -> tuple[float | None, float | None]:
        """推定版の世界のとき、掴む物体の推定（task.pick）と真値の差（位置 [m]、箱の向き [deg]、180° の範囲）。"""
        if not self.estimated or task.pick.surface is None:
            return None, None
        t = self.truth.object_state(task.key)
        pos = math.dist(task.pick.position[:2], t.position[:2])
        if self.scene.object(task.key).shape != "cube":
            return pos, None
        d = (task.pick.yaw - t.yaw) % math.pi
        return pos, math.degrees(min(d, math.pi - d))

    def _run_tasks(self, tasks, base, *, round_: int) -> None:
        for task in tasks:
            est_err, yaw_err = self._estimate_errors(task)
            rec = self._do_task(task, base)
            rec.round = round_
            rec.estimate_error_m, rec.estimate_yaw_error_deg = est_err, yaw_err
            if self.estimated and rec.status == "succeeded":  # 置いた物体は、置いた位置にあるとみなす
                self.world.note_placed(task.key, task.place_position, task.place_yaw)
            state = self.truth.object_state(task.key)
            rec.final_position = tuple(state.position)
            rec.error_m = math.dist(state.position[:2], task.place_position[:2])
            rec.duration_s = self.t - rec.started_s
            self.log.tasks.append(rec)
            self.log.event(self.t, "task", rec.reason or "", key=rec.key, status=rec.status, round=round_)

    def _detect(self, current: Observation):
        cam = self.scene.camera
        return detect_moves(
            self.vlm,
            encode_png(self.reference.rgb),
            encode_png(current.rgb),
            object_names=[o.name for o in self.scene.objects],
            width=cam.width,
            height=cam.height,
            retries=self.config.vlm_retries,
        )

    def _perceive(self, obs: Observation, moves) -> None:
        """推定版の世界のとき、VLM が答えた物体の位置・向きを今の写真から推定し直す（#23）。"""
        if not self.estimated:
            return
        failed = self.world.update(obs, moves)
        estimates = {
            k: {"position": list(e.position), "yaw": e.yaw, "n_pixels": e.n_pixels}
            for k, e in self.world.estimates.items()
        }
        self.log.event(self.t, "perception", f"{len(failed)} 個の推定に失敗", failed=failed, estimates=estimates)

    def _verify_and_retry(self, after: Observation, base) -> None:
        log = self.log
        detection = self._detect(after)
        log.verify = {"detection": detection_to_dict(detection), "instructions": [], "dropped": []}
        if not detection.ok:
            log.event(self.t, "verify_failed", "確認の VLM の問い合わせに失敗。1 回目の結果で終える")
            return
        self._perceive(after, detection.moves)
        instructions, dropped = to_instructions(detection.moves, self.scene, self.reference, self.scene.camera)
        log.verify["instructions"] = [
            {"key": i.key, "target_position": list(i.target_position), "target_surface": i.target_surface}
            for i in instructions
        ]
        log.verify["dropped"] = dropped
        keys = [i.key for i in instructions]
        log.event(self.t, "verify", "残りなし" if not keys else "やり直す：" + "・".join(keys), remaining=keys)
        if not instructions:
            return
        tasks = plan_tasks(instructions, self.world, self.scene, self.world.robot_pose())
        self._run_tasks(tasks, base, round_=2)
        log.photos["final"] = self._capture_current()

    def run(self) -> EpisodeLog:
        """状態遷移を最後まで進め、記録を返す。"""
        if self.reference is None:
            raise RuntimeError("いつもの写真がありません（teach() を先に呼ぶか、reference を与える）")
        log = self.log
        base = self.world.robot_pose()
        log.meta = {**self.meta, "milestone": self.config.milestone, "base_pose": [base.x, base.y, base.yaw]}
        log.photos["reference"] = self.reference
        log.event(self.t, "start")

        current = self._capture_current()
        log.photos["current"] = current
        cam = self.scene.camera
        detection = detect_moves(
            self.vlm,
            encode_png(self.reference.rgb),
            encode_png(current.rgb),
            object_names=[o.name for o in self.scene.objects],
            width=cam.width,
            height=cam.height,
            retries=self.config.vlm_retries,
        )
        log.detection = detection_to_dict(detection)
        log.event(self.t, "vlm", f"{len(detection.moves)} 個の予測", ok=detection.ok)
        if not detection.ok:
            log.finish("failed", "VLM の問い合わせに失敗")
            self._arm_to("standby")
            log.sim_time_s = self.t
            return log
        self._perceive(current, detection.moves)

        instructions, dropped = to_instructions(detection.moves, self.scene, self.reference, cam)
        log.instructions = [
            {"key": i.key, "target_position": list(i.target_position), "target_surface": i.target_surface}
            for i in instructions
        ]
        log.dropped = dropped
        tasks = plan_tasks(instructions, self.world, self.scene, base)
        log.event(self.t, "plan", " → ".join(t.key for t in tasks))

        self._run_tasks(tasks, base, round_=1)

        if self.config.verify:
            # 確認：片付けた後の写真をいつもの写真と比べ、まだ動いている物体だけを 1 回やり直す（設計書 10 章）
            after = self._capture_current()
            log.photos["after"] = after
            self._verify_and_retry(after, base)
        else:
            self._arm_to("photo")
            log.photos["after"] = self.camera.capture()
        self._arm_to("standby")

        # 成否は物体ごとの最後の結果で決める（1 回目に失敗しても、やり直しで戻せれば成功）
        last = {t.key: t for t in log.tasks}
        succeeded = sum(t.status == "succeeded" for t in last.values())
        if succeeded == len(last):
            log.finish("succeeded")
        elif succeeded > 0:
            log.finish("partial", f"{len(last) - succeeded} 個のタスクが失敗")
        else:
            log.finish("failed", "すべてのタスクが失敗")
        log.sim_time_s = self.t
        log.event(self.t, "end", log.status)
        return log
