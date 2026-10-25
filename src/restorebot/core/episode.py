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

from restorebot.core.navigation import in_reach
from restorebot.core.occlusion import needs_retake, robot_mask
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
    action_timeout_s: float = 60.0  # 1 つの動作（掴む・置く・姿勢の変更）の時間切れ（シミュレーションの時間）
    dt: float = 1.0 / 60.0  # Action.step に渡す時間（物理の 1 ステップ）


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
    ) -> None:
        if (config or EpisodeConfig()).milestone != "m1":
            raise NotImplementedError("今は M1（台車を動かさない）だけ。M2 は #17 で足す")
        self.scene = scene
        self.world = world
        self.camera = camera
        self.robot = robot
        self.vlm = vlm
        self.sim = sim
        self.config = config or EpisodeConfig()
        self.meta = dict(meta or {})
        self.reference: Observation | None = None  # いつもの写真（teach() で撮る、または外から与える）
        self.t = 0.0  # シミュレーションの経過時間 [s]
        self.log = EpisodeLog()

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
                self.log.event(self.t, "retake", "ロボットが天板の物体を置ける範囲を隠している", attempt=attempt + 1)
            else:
                self.log.event(self.t, "still_hidden", f"{cfg.max_retakes} 回撮り直しても隠れている。この写真で進める")
        return obs

    def _do_task(self, task, base) -> TaskRecord:
        rec = TaskRecord(task.key, tuple(task.place_position), tuple(task.pick.position), started_s=self.t)
        pick_xy, place_xy = task.pick.position[:2], task.place_position[:2]
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

        instructions, dropped = to_instructions(detection.moves, self.scene, self.reference, cam)
        log.instructions = [
            {"key": i.key, "target_position": list(i.target_position), "target_surface": i.target_surface}
            for i in instructions
        ]
        log.dropped = dropped
        tasks = plan_tasks(instructions, self.world, self.scene, base)
        log.event(self.t, "plan", " → ".join(t.key for t in tasks))

        for task in tasks:
            rec = self._do_task(task, base)
            state = self.world.object_state(task.key)
            rec.final_position = tuple(state.position)
            rec.error_m = math.dist(state.position[:2], task.place_position[:2])
            rec.duration_s = self.t - rec.started_s
            log.tasks.append(rec)
            log.event(self.t, "task", rec.reason or "", key=rec.key, status=rec.status)

        self._arm_to("photo")
        log.photos["after"] = self.camera.capture()
        self._arm_to("standby")

        succeeded = sum(t.status == "succeeded" for t in log.tasks)
        if succeeded == len(log.tasks):
            log.finish("succeeded")
        elif succeeded > 0:
            log.finish("partial", f"{len(log.tasks) - succeeded} 個のタスクが失敗")
        else:
            log.finish("failed", "すべてのタスクが失敗")
        log.sim_time_s = self.t
        log.event(self.t, "end", log.status)
        return log
