"""真下向きの天井カメラの幾何（投影・逆投影）。

画像の座標は左上が原点で、u（x）が右、v（y）が下向きのピクセル。カメラは真下を向き、
画像の右がワールドの +x、画像の上がワールドの +y になる（#5 で画像を保存して確認済み）。
深度は distance_to_image_plane [m]（光軸方向の距離。真下向きなのでカメラとの高さの差）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DownCamera:
    """真下を向いたピンホールカメラ。

    focal_length と horizontal_aperture は同じ単位（USD の Camera と同じく mm など）。画素は正方形。
    """

    position: tuple[float, float, float]
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float

    @property
    def fx(self) -> float:
        return self.focal_length / self.horizontal_aperture * self.width

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """ワールド座標の点 → 画像上の座標 (u, v) [px]。"""
        x0, y0, z0 = self.position
        depth = z0 - point[2]
        if depth <= 0:
            raise ValueError("カメラより上の点は投影できません")
        return (self.cx + self.fx * (point[0] - x0) / depth, self.cy - self.fy * (point[1] - y0) / depth)

    def unproject(self, uv: Sequence[float], depth: float) -> tuple[float, float, float]:
        """画像上の座標 (u, v) [px] と深度 [m]（distance_to_image_plane）→ ワールド座標の点。"""
        if depth <= 0:
            raise ValueError(f"深度は正の値が必要です: {depth}")
        x0, y0, z0 = self.position
        return (x0 + (uv[0] - self.cx) * depth / self.fx, y0 - (uv[1] - self.cy) * depth / self.fy, z0 - depth)

    def meters_per_pixel(self, z: float) -> float:
        """高さ z の水平面上で、1 ピクセルが何 m に当たるか。"""
        return (self.position[2] - z) / self.fx
