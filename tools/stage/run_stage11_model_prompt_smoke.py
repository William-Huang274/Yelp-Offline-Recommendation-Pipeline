#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = REPO_ROOT / "config" / "demo" / "stage11_model_prompt_smoke_case.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_path(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} missing: {path.relative_to(REPO_ROOT)}")


def require_markers(text: str, markers: list[str], label: str, errors: list[str]) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        errors.append(f"{label} missing markers: {', '.join(missing)}")


def main() -> int:
    errors: list[str] = []
    require_path(CASE_PATH, "smoke case", errors)
    if errors:
        print("FAIL stage11_model_prompt_smoke")
        for item in errors:
            print(f"- {item}")
        return 1

    case = json.loads(read_text(CASE_PATH))
    reward = case["reward_model_mainline"]

    req_path = REPO_ROOT / str(reward["requirements_file"])
    train_script = REPO_ROOT / str(reward["train_script"])
    eval_script = REPO_ROOT / str(reward["eval_script"])
    trainer_source = REPO_ROOT / str(reward["trainer_source"])

    for path, label in [
        (req_path, "stage11 qlora requirements"),
        (train_script, "stage11 train launcher"),
        (eval_script, "stage11 eval launcher"),
        (trainer_source, "stage11 trainer source"),
    ]:
        require_path(path, label, errors)

    if errors:
        print("FAIL stage11_model_prompt_smoke")
        for item in errors:
            print(f"- {item}")
        return 1

    req_text = read_text(req_path)
    train_text = read_text(train_script)
    eval_text = read_text(eval_script)
    trainer_text = read_text(trainer_source)

    require_markers(
        req_text,
        list(reward["requirements_markers"]),
        "requirements-stage11-qlora.txt",
        errors,
    )

    reward_model = str(reward["base_model"])
    for text, label in [
        (train_text, "stage11 train launcher"),
        (eval_text, "stage11 eval launcher"),
        (trainer_text, "stage11 trainer source"),
    ]:
        require_markers(text, [reward_model], label, errors)

    if errors:
        print("FAIL stage11_model_prompt_smoke")
        for item in errors:
            print(f"- {item}")
        return 1

    print("PASS stage11_model_prompt_smoke")
    print(f"- reward_model_mainline={reward_model}")
    print(f"- requirements={reward['requirements_file']}")
    print(f"- train_launcher={reward['train_script']}")
    print(f"- eval_launcher={reward['eval_script']}")
    print(f"- trainer_source={reward['trainer_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
