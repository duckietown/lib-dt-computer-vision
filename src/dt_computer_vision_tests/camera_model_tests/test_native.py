import numpy as np

from dt_computer_vision.camera import CameraModel

# NOTE: this is from the real `myrobot` duckiebot at TTIC, March 2022
test_camera = {
    "width": 640,
    "height": 480,
    "K": [[295.79606866959824, 0.0, 321.2621599038631],
          [0.0, 299.5389048862878, 241.73616515312332],
          [0.0, 0.0, 1.0]],
    "D": [-0.23543978771661125,
          0.03637781479419574,
          -0.0033069818601306755,
          -0.0012140708179525926,
          0.0],
    "P": [[201.14027404785156, 0.0, 319.5586620845679, 0.0],
          [0.0, 239.74398803710938, 237.60151004037834, 0.0],
          [0.0, 0.0, 1.0, 0.0]],
    # NOTE: this homography is computed in the 20-entrinsics-calibration jupyter notebook
    "H": [[-3.25205447e-02,  1.19331664e-01,  3.93352735e-01],
          [-4.72717010e-01, -9.44920613e-05, -1.91892118e-02],
          [-1.92169544e-01,  3.63163818e+00,  1.00000000e+00]],
}


def test_from_native():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    assert camera.width == test_camera["width"]
    assert camera.height == test_camera["height"]
    assert np.all(camera.K == test_camera["K"])
    assert np.all(camera.D == test_camera["D"])
    assert np.all(camera.P == test_camera["P"])
    assert np.all(camera.R == np.eye(3))
    assert np.all(camera.H == test_camera["H"])


def test_to_native():
    camera: CameraModel = CameraModel.from_native_objects(test_camera)
    native: dict = camera.to_native_objects()
    assert native["width"] == test_camera["width"]
    assert native["height"] == test_camera["height"]
    assert np.all(native["K"] == test_camera["K"])
    assert np.all(native["D"] == test_camera["D"])
    assert np.all(native["P"] == test_camera["P"])
    assert np.all(native["R"] == np.eye(3))
    assert np.all(native["H"] == test_camera["H"])
