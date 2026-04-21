#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    prompt_only = case["prompt_only_probe_surface"]

    req_path = REPO_ROOT / str(reward["requirements_file"])
    train_script = REPO_ROOT / str(reward["train_script"])
    eval_script = REPO_ROOT / str(reward["eval_script"])
    trainer_source = REPO_ROOT / str(reward["trainer_source"])
    audit_script = REPO_ROOT / str(prompt_only["audit_script"])
    queue_script = REPO_ROOT / str(prompt_only["queue_infer_script"])
    probe_launchers = [
        (REPO_ROOT / str(item["path"]), str(item["model_marker"]))
        for item in prompt_only["probe_launchers"]
    ]

    for path, label in [
        (req_path, "stage11 qlora requirements"),
        (train_script, "stage11 train launcher"),
        (eval_script, "stage11 eval launcher"),
        (trainer_source, "stage11 trainer source"),
        (audit_script, "stage11 prompt-only audit script"),
        (queue_script, "stage11 prompt-only queue script"),
    ]:
        require_path(path, label, errors)
    for path, _marker in probe_launchers:
        require_path(path, f"prompt-only probe launcher {path.name}", errors)

    if errors:
        print("FAIL stage11_model_prompt_smoke")
        for item in errors:
            print(f"- {item}")
        return 1

    req_text = read_text(req_path)
    train_text = read_text(train_script)
    eval_text = read_text(eval_script)
    trainer_text = read_text(trainer_source)
    audit_text = read_text(audit_script)
    queue_text = read_text(queue_script)

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

    require_markers(
        audit_text,
        ["Qwen3.5-35B-A3B-Base", "PROMPT_ONLY_BASE_MODEL"],
        str(prompt_only["audit_script"]),
        errors,
    )
    require_markers(
        queue_text,
        ["AutoTokenizer", "load tokenizer/model dependencies"],
        str(prompt_only["queue_infer_script"]),
        errors,
    )
    for path, marker in probe_launchers:
        require_markers(read_text(path), [marker], path.name, errors)

    for template in prompt_only["prompt_templates"]:
        template_path = REPO_ROOT / str(template["path"])
        require_path(template_path, f"prompt template {template['path']}", errors)
        if template_path.exists():
            require_markers(
                read_text(template_path),
                list(template["markers"]),
                str(template["path"]),
                errors,
            )

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
    print(f"- prompt_only_audit={prompt_only['audit_script']}")
    print(f"- prompt_only_queue={prompt_only['queue_infer_script']}")
    for probe in prompt_only["probe_launchers"]:
        print(f"- prompt_probe={probe['path']} [{probe['model_marker']}]")
    for template in prompt_only["prompt_templates"]:
        print(f"- prompt_template={template['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
