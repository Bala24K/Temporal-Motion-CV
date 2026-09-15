"""
Unit tests for recognition/qevd_dataset.py dataset loader and sequence categorizer.
"""
import numpy as np
import pytest
from body_model.skeleton import SkeletonFrame
from pose.landmarks import Landmark, PoseFrame
from body_model.sequence import SkeletonSequence
from recognition.qevd_dataset import categorize_exercise_from_sequence


def test_categorize_exercise_squat():
    seq = SkeletonSequence()
    sf1 = SkeletonFrame(timestamp=0.0)
    sf1.joint_angles["left_knee"] = 175.0
    sf1.joint_angles["right_knee"] = 175.0

    sf2 = SkeletonFrame(timestamp=0.1)
    sf2.joint_angles["left_knee"] = 110.0
    sf2.joint_angles["right_knee"] = 110.0

    seq.push(sf1)
    seq.push(sf2)

    cat = categorize_exercise_from_sequence(seq, [PoseFrame(timestamp=0.0)])
    assert cat == "squat"


def test_categorize_exercise_bicep_curl():
    seq = SkeletonSequence()
    sf1 = SkeletonFrame(timestamp=0.0)
    sf1.joint_angles["left_elbow"] = 160.0

    sf2 = SkeletonFrame(timestamp=0.1)
    sf2.joint_angles["left_elbow"] = 50.0

    seq.push(sf1)
    seq.push(sf2)

    cat = categorize_exercise_from_sequence(seq, [PoseFrame(timestamp=0.0)])
    assert cat == "bicep_curl"
