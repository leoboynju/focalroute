import json
import os
import pathlib
import stat
import subprocess

import numpy as np
import pytest

from openpi.training import token_budget

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATED_SCRIPT = REPO_ROOT / "scripts" / "train_pi05_libero_calibrated_pytorch.sh"
VISION_ROUTER_SCRIPT = REPO_ROOT / "scripts" / "train_pi05_libero_vision_router_pytorch.sh"
H70_S85_SCRIPT = REPO_ROOT / "scripts" / "train_pi05_libero_vit_h70_s85_4gpu.sh"


def _make_fake_python(tmp_path: pathlib.Path, *, fail_calibration: bool = False) -> pathlib.Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "invocations.jsonl"
    fake_py = tmp_path / "fake_python.py"
    code = f"""#!{REPO_ROOT / '.venv/bin/python'}
import json
import pathlib
import sys

import numpy as np

log_path = pathlib.Path({str(log_file)!r})
args = sys.argv[1:]
with log_path.open("a") as f:
    f.write(json.dumps({{"args": args}}) + "\\n")

if any("calibrate_token_budget.py" in a for a in args):
    if {fail_calibration!r}:
        sys.exit(1)
    out_idx = args.index("--output-dir") + 1
    src_idx = args.index("--attention-source") + 1
    out_dir = pathlib.Path(args[out_idx])
    source = args[src_idx]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"token_budget_{{source}}.json"
    npz_path = out_dir / f"token_budget_{{source}}.npz"
    json_path.write_text(json.dumps({{"format_version": 1, "attention_source": source}}))
    np.savez(npz_path, sample_indices=np.array([0]), relevance=np.ones((1, 4), dtype=np.float32), valid_mask=np.ones((1, 4), dtype=bool), hard_tokens_per_sample=np.array([2]))
    sys.exit(0)

sys.exit(0)
"""
    fake_py.write_text(code)
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC)
    return fake_py


def test_calibrated_launcher_dry_run_and_rejects_positional_args(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "EXP_NAME": "test_dry_run",
        "OUTPUTS_BASE_DIR": str(tmp_path / "outputs"),
        "CHECKPOINT_BASE_DIR": str(tmp_path / "ckpts"),
        "CALIBRATION_CHECKPOINT_DIR": str(tmp_path / "teacher"),
        "PYTORCH_WEIGHT_PATH": str(tmp_path / "teacher"),
    }
    res = subprocess.run(["bash", str(CALIBRATED_SCRIPT)], env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "[DRY_RUN]" in res.stdout
    assert not (tmp_path / "outputs").exists()

    # Positional arguments must be rejected.
    bad_res = subprocess.run(
        ["bash", str(CALIBRATED_SCRIPT), "--extra-arg"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_res.returncode != 0


def test_calibrated_launcher_runs_calibration_before_training_and_stops_on_failure(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "model.safetensors").write_bytes(b"dummy")

    fake_py_ok = _make_fake_python(tmp_path / "ok", fail_calibration=False)
    env_ok = {
        **os.environ,
        "EXP_NAME": "run_ok",
        "NPROC_PER_NODE": "1",
        "GPU_IDS": "0",
        "OUTPUTS_BASE_DIR": str(tmp_path / "outputs"),
        "CHECKPOINT_BASE_DIR": str(tmp_path / "ckpts"),
        "CALIBRATION_CHECKPOINT_DIR": str(teacher_dir),
        "PYTORCH_WEIGHT_PATH": str(teacher_dir),
        "PYTHON_BIN": str(fake_py_ok),
    }
    res_ok = subprocess.run(["bash", str(CALIBRATED_SCRIPT)], env=env_ok, capture_output=True, text=True, check=False)
    assert res_ok.returncode == 0
    calls = [json.loads(line)["args"] for line in (tmp_path / "ok" / "invocations.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert any("calibrate_token_budget.py" in a for a in calls[0])
    assert any("train_pytorch.py" in a for a in calls[1])

    # Existing output dir must be rejected on second run.
    res_dup = subprocess.run(["bash", str(CALIBRATED_SCRIPT)], env=env_ok, capture_output=True, text=True, check=False)
    assert res_dup.returncode != 0

    # Calibration failure must stop before launching train_pytorch.py.
    (tmp_path / "fail").mkdir()
    fake_py_fail = _make_fake_python(tmp_path / "fail", fail_calibration=True)
    env_fail = {
        **env_ok,
        "EXP_NAME": "run_fail",
        "PYTHON_BIN": str(fake_py_fail),
    }
    res_fail = subprocess.run(["bash", str(CALIBRATED_SCRIPT)], env=env_fail, capture_output=True, text=True, check=False)
    assert res_fail.returncode != 0
    fail_calls = [
        json.loads(line)["args"] for line in (tmp_path / "fail" / "invocations.jsonl").read_text().splitlines()
    ]
    assert len(fail_calls) == 1
    assert any("calibrate_token_budget.py" in a for a in fail_calls[0])


def test_vision_router_launcher_and_h70_s85_guards(tmp_path):
    # Disabling both branches in common launcher must fail.
    env_both_off = {
        **os.environ,
        "HARD_ENABLED": "false",
        "SOFT_ENABLED": "false",
    }
    res = subprocess.run(
        ["bash", str(VISION_ROUTER_SCRIPT)],
        env=env_both_off,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0

    # Resume with budget path must fail.
    env_resume_budget = {
        **os.environ,
        "RUN_MODE": "resume",
        "VISION_ROUTER_BUDGET_PATH": "/tmp/some_budget.json",
    }
    res_rb = subprocess.run(
        ["bash", str(VISION_ROUTER_SCRIPT)],
        env=env_resume_budget,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_rb.returncode != 0

    # H70/S85 script dry run succeeds and checks output collision.
    env_h70 = {
        **os.environ,
        "DRY_RUN": "1",
        "EXP_NAME": "test_h70_s85",
        "OUTPUTS_BASE_DIR": str(tmp_path / "outputs"),
        "CHECKPOINT_BASE_DIR": str(tmp_path / "ckpts"),
    }
    res_h70 = subprocess.run(["bash", str(H70_S85_SCRIPT)], env=env_h70, capture_output=True, text=True, check=False)
    assert res_h70.returncode == 0
    assert "COVERAGE=0.70" in res_h70.stdout


@pytest.mark.parametrize(
    ("script_name", "expected_hard_flag", "expected_soft_flag", "expected_h", "expected_g"),
    [
        ("train_pi05_libero_hard_router_pytorch.sh", "--model.vision-router-hard-enabled", "--model.no-vision-router-soft-enabled", "128", None),
        ("train_pi05_libero_hard_router_h256_pytorch.sh", "--model.vision-router-hard-enabled", "--model.no-vision-router-soft-enabled", "256", None),
        ("train_pi05_libero_soft_router_pytorch.sh", "--model.no-vision-router-hard-enabled", "--model.vision-router-soft-enabled", None, "128"),
        ("train_pi05_libero_soft_router_s256_pytorch.sh", "--model.no-vision-router-hard-enabled", "--model.vision-router-soft-enabled", None, "256"),
        ("train_pi05_libero_dual_router_pytorch.sh", "--model.vision-router-hard-enabled", "--model.vision-router-soft-enabled", "128", "128"),
        ("train_pi05_libero_dual_router_h64_s128_pytorch.sh", "--model.vision-router-hard-enabled", "--model.vision-router-soft-enabled", "64", "128"),
        ("train_pi05_libero_dual_router_h128_s64_pytorch.sh", "--model.vision-router-hard-enabled", "--model.vision-router-soft-enabled", "128", "64"),
        ("train_pi05_libero_dual_router_h128_s256_pytorch.sh", "--model.vision-router-hard-enabled", "--model.vision-router-soft-enabled", "128", "256"),
        ("train_pi05_libero_dual_router_h256_s128_pytorch.sh", "--model.vision-router-hard-enabled", "--model.vision-router-soft-enabled", "256", "128"),
    ],
)
def test_ablation_variant_launcher_scripts(
    tmp_path,
    script_name,
    expected_hard_flag,
    expected_soft_flag,
    expected_h,
    expected_g,
):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir(exist_ok=True)
    (teacher_dir / "model.safetensors").write_bytes(b"dummy")
    fake_py = _make_fake_python(tmp_path / script_name.replace(".sh", ""), fail_calibration=False)
    env = {
        **os.environ,
        "NPROC_PER_NODE": "1",
        "GPU_IDS": "0",
        "PYTORCH_WEIGHT_PATH": str(teacher_dir),
        "CHECKPOINT_BASE_DIR": str(tmp_path / "ckpts"),
        "PYTHON_BIN": str(fake_py),
        "VISION_ROUTER_BUDGET_PATH": "",
    }
    script_path = REPO_ROOT / "scripts" / script_name
    res = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    log_path = tmp_path / script_name.replace(".sh", "") / "invocations.jsonl"
    args = json.loads(log_path.read_text().splitlines()[0])["args"]
    assert expected_hard_flag in args
    assert expected_soft_flag in args
    if expected_h is not None:
        idx = args.index("--model.vision-router-hard-tokens")
        assert args[idx + 1] == expected_h
    if expected_g is not None:
        idx = args.index("--model.vision-router-soft-tokens")
        assert args[idx + 1] == expected_g
