"""restorebot.core.camera（真下向きの天井カメラの投影・逆投影）の単体テスト。"""

import pytest

from restorebot.core.camera import DownCamera
from restorebot.core.scene import HOME_V2

CAM = DownCamera(position=(0.0, 0.0, 2.5), width=1024, height=768, focal_length=18.0, horizontal_aperture=36.0)


def test_intrinsics_from_focal_length_and_aperture():
    assert CAM.fx == pytest.approx(18.0 / 36.0 * 1024)
    assert CAM.fy == pytest.approx(CAM.fx)  # 正方画素
    assert (CAM.cx, CAM.cy) == (512.0, 384.0)


def test_point_below_camera_projects_to_image_center():
    assert CAM.project((0.0, 0.0, 0.0)) == pytest.approx((512.0, 384.0))


def test_image_orientation_right_is_plus_x_and_up_is_plus_y():
    u0, v0 = CAM.project((0.0, 0.0, 0.0))
    u_px, v_px = CAM.project((0.5, 0.0, 0.0))
    u_py, v_py = CAM.project((0.0, 0.5, 0.0))
    assert u_px > u0 and v_px == pytest.approx(v0)  # +x は画像の右
    assert v_py < v0 and u_py == pytest.approx(u0)  # +y は画像の上（v が小さい）


def test_point_above_camera_cannot_be_projected():
    with pytest.raises(ValueError):
        CAM.project((0.0, 0.0, 3.0))


@pytest.mark.parametrize("point", [(0.0, 0.0, 0.0), (0.8, 0.72, 0.79), (-1.5, -0.1, 0.55), (1.9, -1.4, 0.0)])
def test_unproject_inverts_project(point):
    cam = HOME_V2.camera
    u, v = cam.project(point)
    depth = cam.position[2] - point[2]  # distance_to_image_plane（真下向きなので高さの差）
    assert cam.unproject((u, v), depth) == pytest.approx(point)


def test_unproject_rejects_non_positive_depth():
    with pytest.raises(ValueError):
        CAM.unproject((10.0, 10.0), 0.0)


def test_meters_per_pixel_on_floor():
    # 高さ 2.5m、fx = 512px → 床の上で 2.5 / 512 m/px
    assert CAM.meters_per_pixel(0.0) == pytest.approx(2.5 / 512)


def test_home_camera_sees_whole_floor():
    cam = HOME_V2.camera
    hx, hy = HOME_V2.room_half
    for corner in [(-hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0), (hx, -hy, 0.0)]:
        u, v = cam.project(corner)
        assert 0 <= u <= cam.width and 0 <= v <= cam.height
