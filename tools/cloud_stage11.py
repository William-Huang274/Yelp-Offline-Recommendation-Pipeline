#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import posixpath
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HOST = "connect.westb.seetacloud.com"
DEFAULT_PORT = 20804
DEFAULT_USER = "root"


@dataclass(frozen=True)
class CloudItem:
    key: str
    remote_path: str
    local_path: Path
    purpose: str
    recommended: bool = False
    large: bool = False


@dataclass(frozen=True)
class LocalCheck:
    label: str
    path: Path
    cloud_item: str | None = None


CLOUD_ITEMS: dict[str, CloudItem] = {
    "stage11_freeze_pack": CloudItem(
        key="stage11_freeze_pack",
        remote_path="/root/autodl-tmp/5006_BDA_fast/_github_freeze_pack_20260409/stage11_freeze_pack_20260409",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "stage11_freeze_pack_20260409",
        purpose="small Stage11 freeze evidence pack for local review",
        recommended=True,
    ),
    "stage11_v120_eval_full": CloudItem(
        key="stage11_v120_eval_full",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket5_top250_semantic_compact_boundary_reason_rm_eval_westc_v120_segmented31_60_joint12/output/11_qlora_sidecar_eval",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "stage11_v120_eval_full",
        purpose="full v120 two-band eval score dump for detailed local inspection",
    ),
    "stage11_v124_eval_full": CloudItem(
        key="stage11_v124_eval_full",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket5_top250_semantic_compact_boundary_reason_rm_eval_westc_v124_triband_joint12_gate_top100/output/11_qlora_sidecar_eval",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "stage11_v124_eval_full",
        purpose="full v124 tri-band eval score dump for detailed local inspection",
    ),
    "stage10_profile_sourceparity": CloudItem(
        key="stage10_profile_sourceparity",
        remote_path="/root/autodl-tmp/5006_BDA_project/data/output/09_user_intent_profile_v2_bucket5_sourceparity",
        local_path=REPO_ROOT / "data" / "output" / "09_user_intent_profile_v2_bucket5_sourceparity",
        purpose="Stage10 bucket5 source-parity user intent profile prerequisite",
        recommended=True,
    ),
    "stage10_match_channels_sourceparity": CloudItem(
        key="stage10_match_channels_sourceparity",
        remote_path="/root/autodl-tmp/5006_BDA_project/data/output/09_user_business_match_channels_v2_bucket5_sourceparity",
        local_path=REPO_ROOT / "data" / "output" / "09_user_business_match_channels_v2_bucket5_sourceparity",
        purpose="Stage10 bucket5 source-parity match channel prerequisite",
        recommended=True,
    ),
    "stage09_bucket2_sourceparity": CloudItem(
        key="stage09_bucket2_sourceparity",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket2_baseline_fixed/output/09_candidate_fusion_bucket2_baseline_fixed_sourceparity",
        local_path=REPO_ROOT / "data" / "output" / "09_candidate_fusion_bucket2_baseline_fixed_sourceparity",
        purpose="Stage10 bucket2 local replay prerequisite: Stage09 cold-start source-parity candidate run",
        large=True,
    ),
    "stage10_bucket2_model_v4max": CloudItem(
        key="stage10_bucket2_model_v4max",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket2_stage10_fixedcohort_for_stage11_v4max/output/10_rank_models_bucket2_fixedcohort_v4max",
        local_path=REPO_ROOT / "data" / "output" / "10_rank_models_bucket2_fixedcohort_v4max",
        purpose="Optional Stage10 bucket2 trained-model snapshot for cold-start replay inspection",
    ),
    "stage11_v101_11_30_adapter": CloudItem(
        key="stage11_v101_11_30_adapter",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket5_top250_semantic_compact_boundary_reason_rm_train_westd_v101_11_30only4pair/output/11_qlora_models",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "models" / "v101_11_30_adapter",
        purpose="large 11-30 reward-model adapter; keep on cloud unless model files are explicitly needed",
        large=True,
    ),
    "stage11_v117_31_60_adapter": CloudItem(
        key="stage11_v117_31_60_adapter",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket5_top250_semantic_compact_boundary_reason_rm_train_westc_v117_31_60only/output/11_qlora_models",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "models" / "v117_31_60_adapter",
        purpose="large 31-60 reward-model adapter; keep on cloud unless model files are explicitly needed",
        large=True,
    ),
    "stage11_v122_61_100_adapter": CloudItem(
        key="stage11_v122_61_100_adapter",
        remote_path="/root/autodl-tmp/5006_BDA_fast/bucket5_top250_semantic_compact_boundary_reason_rm_train_westc_v122_61_100only_globalaware/output/11_qlora_models",
        local_path=REPO_ROOT / "data" / "output" / "cloud_stage11" / "models" / "v122_61_100_adapter",
        purpose="large 61-100 reward-model adapter; keep on cloud unless model files are explicitly needed",
        large=True,
    ),
}


DEMO_REQUIRED_FILES = [
    LocalCheck(
        "Stage09 current-release summary",
        REPO_ROOT / "data" / "output" / "current_release" / "stage09" / "bucket5_route_aware_sourceparity" / "summary.json",
    ),
    LocalCheck(
        "Stage09 recall audit",
        REPO_ROOT / "data" / "output" / "current_release" / "stage09" / "bucket5_route_aware_sourceparity" / "stage09_recall_audit.json",
    ),
    LocalCheck(
        "Stage10 current-release summary",
        REPO_ROOT / "data" / "output" / "current_release" / "stage10" / "stage10_current_mainline_summary.json",
    ),
    LocalCheck(
        "Stage11 v120 best-known summary",
        REPO_ROOT / "data" / "output" / "current_release" / "stage11" / "eval" / "bucket5_two_band_best_known_v120_alpha080.json",
    ),
    LocalCheck(
        "Stage11 v124 freeze summary",
        REPO_ROOT / "data" / "output" / "current_release" / "stage11" / "eval" / "bucket5_tri_band_freeze_v124_alpha036" / "summary.json",
    ),
    LocalCheck(
        "Stage11 expert training summary",
        REPO_ROOT / "data" / "output" / "current_release" / "stage11" / "experts" / "expert_training_summary.json",
    ),
]


STAGE10_LOCAL_CHECKS = [
    LocalCheck(
        "Stage09 bucket5 source-parity candidate run",
        REPO_ROOT
        / "data"
        / "output"
        / "09_candidate_fusion_structural_v5_sourceparity"
        / "20260324_030511_full_stage09_candidate_fusion",
    ),
    LocalCheck(
        "Stage09 text-match features",
        REPO_ROOT
        / "data"
        / "output"
        / "09_candidate_wise_text_match_features_v1"
        / "20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build",
    ),
    LocalCheck(
        "Stage09 group-gap features",
        REPO_ROOT
        / "data"
        / "output"
        / "09_stage10_group_gap_features_v1"
        / "20260323_174757_full_stage09_stage10_group_gap_features_v1_build",
    ),
    LocalCheck(
        "Stage10 source-parity user intent profile root",
        CLOUD_ITEMS["stage10_profile_sourceparity"].local_path,
        "stage10_profile_sourceparity",
    ),
    LocalCheck(
        "Stage10 source-parity match channel root",
        CLOUD_ITEMS["stage10_match_channels_sourceparity"].local_path,
        "stage10_match_channels_sourceparity",
    ),
    LocalCheck(
        "Stage10 fixed eval cohort",
        REPO_ROOT / "data" / "output" / "fixed_eval_cohorts" / "bucket5_accepted_test_users_1935_userid.csv",
    ),
]


STAGE10_BUCKET2_LOCAL_CHECKS = [
    LocalCheck(
        "Stage09 bucket2 source-parity candidate root",
        CLOUD_ITEMS["stage09_bucket2_sourceparity"].local_path,
        "stage09_bucket2_sourceparity",
    ),
    LocalCheck(
        "Stage09 text-match features",
        REPO_ROOT
        / "data"
        / "output"
        / "09_candidate_wise_text_match_features_v1"
        / "20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build",
    ),
    LocalCheck(
        "Stage09 group-gap features",
        REPO_ROOT
        / "data"
        / "output"
        / "09_stage10_group_gap_features_v1"
        / "20260323_174757_full_stage09_stage10_group_gap_features_v1_build",
    ),
    LocalCheck(
        "Stage10 bucket2 fixed eval cohort",
        REPO_ROOT / "data" / "output" / "fixed_eval_cohorts" / "bucket2_gate_eval_users_5344_useridx.csv",
    ),
    LocalCheck(
        "Optional Stage10 bucket2 trained-model snapshot",
        CLOUD_ITEMS["stage10_bucket2_model_v4max"].local_path,
        "stage10_bucket2_model_v4max",
    ),
]


def import_paramiko():
    try:
        import paramiko  # type: ignore
    except ImportError as exc:
        raise SystemExit("paramiko is required for cloud access. Install it with: python -m pip install paramiko") from exc
    return paramiko


def cloud_password() -> str:
    password = os.environ.get("BDA_CLOUD_PASSWORD")
    if password:
        return password
    return getpass.getpass("BDA_CLOUD_PASSWORD: ")


def connect(args: argparse.Namespace):
    paramiko = import_paramiko()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=cloud_password(),
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run_remote(client, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def local_status(path: Path) -> str:
    if path.exists():
        return "present"
    return "missing"


def print_local_checks() -> bool:
    ok = True
    print("Local demo files")
    for check in DEMO_REQUIRED_FILES:
        status = local_status(check.path)
        ok &= status == "present"
        print(f"- {status}: {check.label} -> {check.path.relative_to(REPO_ROOT)}")

    print("")
    print("Local Stage10 bucket5 prerequisites")
    for check in STAGE10_LOCAL_CHECKS:
        status = local_status(check.path)
        ok &= status == "present"
        suffix = f" | pull item: {check.cloud_item}" if check.cloud_item and status == "missing" else ""
        print(f"- {status}: {check.label} -> {check.path.relative_to(REPO_ROOT)}{suffix}")

    print("")
    print("Optional local Stage10 bucket2 prerequisites")
    for check in STAGE10_BUCKET2_LOCAL_CHECKS:
        status = local_status(check.path)
        suffix = f" | pull item: {check.cloud_item}" if check.cloud_item and status == "missing" else ""
        print(f"- {status}: {check.label} -> {check.path.relative_to(REPO_ROOT)}{suffix}")

    print("")
    print("Optional cloud-backed Stage11 artifacts")
    for item in CLOUD_ITEMS.values():
        if item.key.startswith("stage11_"):
            status = local_status(item.local_path)
            marker = "large, keep on cloud" if item.large else "optional local copy"
            print(f"- {status}: {item.key} ({marker}) -> {item.local_path.relative_to(REPO_ROOT)}")

    return ok


def remote_item_summary(client, item: CloudItem) -> str:
    path = shlex.quote(item.remote_path)
    command = (
        f"if [ -e {path} ]; then "
        f"echo exists; du -sh {path} 2>/dev/null | awk '{{print $1}}'; "
        f"find {path} -type f 2>/dev/null | wc -l; "
        f"find {path} -maxdepth 2 -type f 2>/dev/null | sort | head -5; "
        f"else echo missing; fi"
    )
    code, out, err = run_remote(client, command)
    if code != 0:
        return f"error: {err.strip() or out.strip()}"
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines or lines[0] != "exists":
        return "missing on cloud"
    size = lines[1] if len(lines) > 1 else "unknown-size"
    file_count = lines[2] if len(lines) > 2 else "unknown-file-count"
    samples = ", ".join(posixpath.basename(line) for line in lines[3:]) or "no sample files"
    return f"cloud present, size={size}, files={file_count}, samples={samples}"


def command_inventory(args: argparse.Namespace) -> int:
    print_local_checks()
    print("")
    print(f"Cloud inventory: {args.user}@{args.host}:{args.port}")
    client = connect(args)
    try:
        for item in CLOUD_ITEMS.values():
            summary = remote_item_summary(client, item)
            local = local_status(item.local_path)
            print(f"- {item.key}: {summary}; local={local}; purpose={item.purpose}")
    finally:
        client.close()
    return 0


def download_dir(sftp, remote_dir: str, local_dir: Path, overwrite: bool) -> tuple[int, int]:
    downloaded = 0
    skipped = 0
    local_dir.mkdir(parents=True, exist_ok=True)
    for attr in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, attr.filename)
        local_path = local_dir / attr.filename
        if stat.S_ISDIR(attr.st_mode):
            d_count, s_count = download_dir(sftp, remote_path, local_path, overwrite)
            downloaded += d_count
            skipped += s_count
            continue
        if local_path.exists() and not overwrite:
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local_path))
        downloaded += 1
    return downloaded, skipped


def command_pull(args: argparse.Namespace) -> int:
    if args.recommended:
        selected = [item for item in CLOUD_ITEMS.values() if item.recommended]
    else:
        selected = [CLOUD_ITEMS[args.item]]

    client = connect(args)
    try:
        sftp = client.open_sftp()
        try:
            for item in selected:
                if item.large and not args.allow_large:
                    print(f"SKIP {item.key}: large artifact. Re-run with --allow-large if you really need it.")
                    continue
                print(f"PULL {item.key}")
                print(f"- remote: {item.remote_path}")
                print(f"- local: {item.local_path}")
                downloaded, skipped = download_dir(sftp, item.remote_path, item.local_path, args.overwrite)
                print(f"- downloaded={downloaded}, skipped={skipped}")
        finally:
            sftp.close()
    finally:
        client.close()
    return 0


def command_print_ssh(args: argparse.Namespace) -> int:
    print("SSH command")
    print(f"ssh -p {args.port} {args.user}@{args.host}")
    print("")
    print("Recommended cloud paths")
    for item in CLOUD_ITEMS.values():
        print(f"- {item.key}: {item.remote_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud Stage11 inventory and artifact pull helper.")
    parser.add_argument("--host", default=os.environ.get("BDA_CLOUD_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BDA_CLOUD_PORT", DEFAULT_PORT)))
    parser.add_argument("--user", default=os.environ.get("BDA_CLOUD_USER", DEFAULT_USER))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("local-check", help="check local demo and Stage10 prerequisite files")
    sub.add_parser("inventory", help="check local files and remote Stage11/Stage10 artifact availability")
    pull = sub.add_parser("pull", help="pull one cloud artifact or all recommended small artifacts")
    pull.add_argument("--item", choices=sorted(CLOUD_ITEMS), default="")
    pull.add_argument("--recommended", action="store_true", help="pull recommended small missing inputs")
    pull.add_argument("--overwrite", action="store_true", help="overwrite existing local files")
    pull.add_argument("--allow-large", action="store_true", help="allow pulling large Stage11 adapter directories")
    sub.add_parser("print-ssh", help="print the SSH command and known cloud paths")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "local-check":
        return 0 if print_local_checks() else 1
    if args.command == "inventory":
        return command_inventory(args)
    if args.command == "pull":
        if not args.recommended and not args.item:
            parser.error("pull requires --item or --recommended")
        return command_pull(args)
    if args.command == "print-ssh":
        return command_print_ssh(args)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
