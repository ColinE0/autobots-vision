"""Focus metric (tools/focus.py).

The number has to move the right way, and it has to move MORE than the noise
between two frames of the same scene, or turning the barrel tells you nothing.
"""
import cv2
import numpy as np

from tools.focus import bar, sharpness


def checkerboard():
    img = np.zeros((240, 320, 3), np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    return img


def test_blur_scores_lower_than_sharp():
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 4)
    assert sharpness(blurred) < sharpness(sharp)


def test_the_gap_is_large_not_marginal():
    # A metric that separates focus from blur by a few percent is unusable
    # when you are turning a barrel by hand.
    sharp = sharpness(checkerboard())
    blurred = sharpness(cv2.GaussianBlur(checkerboard(), (9, 9), 4))
    assert sharp > blurred * 3


def test_centre_region_ignores_the_corners():
    # Soft corners are a property of cheap glass at every focus setting.
    # Scoring them flattens the peak being hunted.
    img = np.zeros((240, 320, 3), np.uint8)
    img[0:30, 0:30] = checkerboard()[0:30, 0:30]      # detail in a corner only
    assert sharpness(img, centre=True) < sharpness(img, centre=False)


def test_a_flat_frame_scores_about_zero():
    assert sharpness(np.full((240, 320, 3), 128, np.uint8)) < 1.0


def test_bar_is_safe_before_a_peak_exists():
    assert bar(0.0, 0.0) == ''
    assert len(bar(5.0, 10.0, width=40)) == 20
