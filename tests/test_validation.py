"""Tests for parsing Godot validation output into pass/fail results."""
import json

from gamedevbench.src.utils.validation import ValidationParser


def test_parse_passed_with_message():
    r = ValidationParser.parse_output("noise\nVALIDATION_PASSED: all good\nmore")
    assert r.success is True
    assert r.message == "all good"


def test_parse_passed_without_message():
    r = ValidationParser.parse_output("VALIDATION_PASSED")
    assert r.success is True
    assert r.message == "Validation passed"


def test_parse_failed_with_message():
    r = ValidationParser.parse_output("VALIDATION_FAILED: bad thing happened")
    assert r.success is False
    assert r.message == "bad thing happened"


def test_no_marker_is_treated_as_failure():
    r = ValidationParser.parse_output("just some godot logs, nothing relevant")
    assert r.success is False
    assert "No validation result" in r.message


def test_first_marker_wins():
    r = ValidationParser.parse_output("VALIDATION_PASSED: ok\nVALIDATION_FAILED: later")
    assert r.success is True


def test_save_result_to_json(tmp_path):
    r = ValidationParser.parse_output("VALIDATION_PASSED: ok")
    ValidationParser.save_result_to_json("task_9999", r, results_dir=tmp_path)
    out = tmp_path / "task_task_9999.json"  # filename is "task_<task_name>.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["success"] is True


def test_save_result_to_json_creates_nested_run_dir(tmp_path):
    r = ValidationParser.parse_output("VALIDATION_FAILED: nope")
    results_dir = tmp_path / "results" / "named-run"
    ValidationParser.save_result_to_json("task_0002", r, results_dir=results_dir)

    out = results_dir / "task_task_0002.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["message"] == "nope"
