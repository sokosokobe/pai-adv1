"""smoke_test.py の引数解析の単体テスト（Isaac Sim 不要）。"""

import argparse

import pytest

from smoke_test import parse_args, parse_resolution


def test_parse_resolution_ok():
    assert parse_resolution("640x480") == (640, 480)
    assert parse_resolution("1280X720") == (1280, 720)


@pytest.mark.parametrize("text", ["640", "640x", "ax480", "0x480", "-1x480"])
def test_parse_resolution_invalid(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_resolution(text)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.steps == 1000
    assert args.no_camera is False
    assert args.resolution == (640, 480)


def test_parse_args_rejects_non_positive_steps():
    with pytest.raises(SystemExit):
        parse_args(["--steps", "0"])
