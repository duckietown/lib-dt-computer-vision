import io

import numpy as np

from dt_computer_vision.camera import CameraModel, Pixel
from dt_computer_vision.camera.types import EQUIDISTANT, PLUMB_BOB

# NOTE: this is from the physical `pdrone24` Duckiedrone, October 2026
test_camera = {
    "width": 480,
    "height": 640,
    "K": [[313.48439395812346, 0.0, 264.1988043657829],
          [0.0, 313.53866521639895, 341.679223053381],
          [0.0, 0.0, 1.0]],
    "D": [-0.025000690096569123,
          -0.019833877811529205,
          0.013135857557684198,
          -0.0045645648684453375],
    "P": [[313.48439395812346, 0.0, 264.1988043657829, 0.0],
          [0.0, 313.53866521639895, 341.679223053381, 0.0],
          [0.0, 0.0, 1.0, 0.0]],
    "H": None,
    "distortion_model": EQUIDISTANT,
}

# rectified pixels, from the image center out to a corner region
test_pixels = [(264, 342), (300, 300), (100, 500), (420, 120), (60, 60)]


def _distort(camera: CameraModel, u: float, v: float) -> np.ndarray:
    # equidistant model: theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    x = (u - camera.P[0, 2]) / camera.P[0, 0]
    y = (v - camera.P[1, 2]) / camera.P[1, 1]
    r = np.hypot(x, y)
    if r < 1e-12:
        return np.array([camera.cx, camera.cy])
    theta = np.arctan(r)
    theta_d = theta * (1 + sum(k * theta ** (2 * (i + 1)) for i, k in enumerate(camera.D)))
    return np.array([camera.fx * theta_d / r * x + camera.cx,
                     camera.fy * theta_d / r * y + camera.cy])


def test_is_fisheye():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    assert camera.is_fisheye
    assert camera.D.shape == (4,)


def test_default_is_plumb_bob():
    native = {k: v for k, v in test_camera.items() if k != "distortion_model"}
    native["D"] = test_camera["D"] + [0.0]
    camera: CameraModel = CameraModel.from_native_objects(native)
    assert camera.distortion_model == PLUMB_BOB
    assert not camera.is_fisheye


def test_rectify_pixel():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    for u, v in test_pixels:
        distorted = _distort(camera, u, v)
        rectified = camera.rectifier.rectify_pixel(Pixel(*distorted))
        assert np.allclose(rectified.as_array(), [u, v], atol=1e-3)


def test_rectify_image():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    rectified = camera.rectifier.rectify(image)
    assert rectified.shape == image.shape
    for u, v in test_pixels:
        distorted = _distort(camera, u, v)
        mapped = [camera.rectifier.mapx[v, u], camera.rectifier.mapy[v, u]]
        assert np.allclose(mapped, distorted, atol=1e-2)


def test_native_round_trip():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    native: dict = camera.to_native_objects()
    assert native["distortion_model"] == EQUIDISTANT
    assert len(native["D"]) == 4
    camera2: CameraModel = CameraModel.from_native_objects(native)
    assert camera2.is_fisheye
    assert np.all(camera2.K == test_camera["K"])
    assert np.all(camera2.D == test_camera["D"])
    assert np.all(camera2.P == test_camera["P"])


def test_ros_round_trip():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    stream = io.StringIO()
    camera.to_ros_calibration(stream)
    stream.seek(0)
    camera2: CameraModel = CameraModel.from_ros_calibration(stream)
    assert camera2.is_fisheye
    assert camera2.width == test_camera["width"]
    assert camera2.height == test_camera["height"]
    assert np.allclose(camera2.K, test_camera["K"])
    assert np.allclose(camera2.D, test_camera["D"])
    assert np.allclose(camera2.P, test_camera["P"])


def test_cropped():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    cropped: CameraModel = camera.cropped(top=40, right=10, bottom=20, left=30)
    assert cropped.is_fisheye
    assert (cropped.width, cropped.height) == (440, 580)
    assert np.all(cropped.D == test_camera["D"])
    assert np.allclose([cropped.cx, cropped.cy], [camera.cx - 30, camera.cy - 40])
    for u, v in [(220, 290), (100, 400), (350, 80)]:
        distorted = _distort(cropped, u, v)
        rectified = cropped.rectifier.rectify_pixel(Pixel(*distorted))
        assert np.allclose(rectified.as_array(), [u, v], atol=1e-3)


def test_scaled():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    scaled: CameraModel = camera.scaled(0.5)
    assert scaled.is_fisheye
    assert (scaled.width, scaled.height) == (240, 320)
    assert np.all(scaled.D == test_camera["D"])
    assert np.allclose([scaled.fx, scaled.fy], [camera.fx / 2, camera.fy / 2])
    for u, v in [(132, 171), (50, 250), (210, 60)]:
        distorted = _distort(scaled, u, v)
        rectified = scaled.rectifier.rectify_pixel(Pixel(*distorted))
        assert np.allclose(rectified.as_array(), [u, v], atol=1e-3)


def _load_with_alpha(alpha: float) -> CameraModel:
    stream = io.StringIO()
    CameraModel.from_native_objects(test_camera).to_ros_calibration(stream)
    stream.seek(0)
    return CameraModel.from_ros_calibration(stream, alpha=alpha)


def test_ros_alpha_keeps_full_image():
    # alpha = 1 keeps all of the source image in view: the midpoints of its four edges
    # rectify to inside the image, and the widest one lands on the border (to half a pixel)
    W, H = test_camera["width"], test_camera["height"]
    loaded: CameraModel = _load_with_alpha(1.0)
    assert loaded.is_fisheye
    assert np.allclose(loaded.D, test_camera["D"])
    assert np.allclose(loaded.P[:, :3], loaded.K)
    source = dict(test_camera, P=loaded.P.tolist())
    camera: CameraModel = CameraModel.from_native_objects(source)
    margins = []
    for u, v in [(W / 2, 0), (W, H / 2), (W / 2, H), (0, H / 2)]:
        x, y = camera.rectifier.rectify_pixel(Pixel(u, v)).as_array()
        margins.extend([x, W - x, y, H - y])
    assert abs(min(margins)) < 0.5


def test_ros_alpha_trades_view_for_zoom():
    # a smaller alpha crops more of the source image, so the focal length grows
    full, half = _load_with_alpha(1.0), _load_with_alpha(0.5)
    assert half.fx > full.fx
    assert half.fy > full.fy
