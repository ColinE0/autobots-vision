"""Label formatting in tools/test_camera.py.

Small, but it is the line a human reads off the console while aiming a prop,
and the percentage is the number they act on.
"""
from tools.test_camera import format_labels


def test_no_detections_reads_as_a_dash():
    assert format_labels([]) == '-'


def test_one_detection_carries_its_area_as_a_percentage():
    assert format_labels([('red_light', 0.0123)]) == 'red_light(1.2%)'


def test_several_are_comma_separated_in_the_order_given():
    line = format_labels([('stop_sign', 0.21), ('red_light', 0.004)])
    assert line == 'stop_sign(21.0%), red_light(0.4%)'


def test_a_blob_under_a_tenth_of_a_percent_still_shows_a_number():
    # 0.05% rounds to 0.1%, not to 0.0%: a reading of zero next to a live
    # detection reads as a bug rather than as a small blob.
    assert format_labels([('green_light', 0.0005)]) == 'green_light(0.1%)'
