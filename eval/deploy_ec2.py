#!/usr/bin/env python3
"""
EC2 Parallel Deployment Orchestrator

--mode c1  : Run Bedrock Sonnet-only baseline on a single t3.xlarge (no GPU required)
--mode c2  : Run Layer 1 + Layer 3 C2 (hybrid) on g6e.xlarge (default: 6 previously failed models)
--mode all : Run Layer 1 + Layer 3 C1+C2 on g6e.xlarge (default behaviour)

Usage:
  # C1 baseline (run once, low cost)
  python eval/deploy_ec2.py --mode c1

  # C2 re-run (6 previously failed models)
  python eval/deploy_ec2.py --mode c2 --models exaone3.5:7.8b,gemma3:4b,llama3.1:8b,phi4-mini:latest,qwen3:14b,gemma3:12b

  # Full run for a specific model
  python eval/deploy_ec2.py --mode all --models qwen3:8b

  # Dry run
  python eval/deploy_ec2.py --mode c2 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).parent.parent

# ── Config ─────────────────────────────────────────────────────────────────────
AWS_REGION           = os.getenv("AWS_REGION",            "us-east-1")
S3_BUCKET            = os.getenv("S3_BUCKET",             "")
S3_PREFIX            = os.getenv("S3_PREFIX",             "benchmark-results")
AMI_ID               = os.getenv("AMI_ID",                "")   # Deep Learning AMI Ubuntu 22.04
INSTANCE_TYPE        = os.getenv("INSTANCE_TYPE",         "g6e.xlarge")
SUBNET_ID            = os.getenv("SUBNET_ID",             "")
SECURITY_GROUP_ID    = os.getenv("SECURITY_GROUP_ID",     "")
KEY_NAME             = os.getenv("KEY_NAME",              "")    # optional
IAM_INSTANCE_PROFILE = os.getenv("IAM_INSTANCE_PROFILE", "")    # needs S3+Bedrock access
SONNET_MODEL         = os.getenv("SONNET_MODEL",          "us.anthropic.claude-sonnet-4-6")

POLL_INTERVAL_SEC    = 60     # S3 result polling interval
TIMEOUT_MIN          = 180    # C2/all: maximum instance wait time (14b models may take up to 2.5 hours)
TIMEOUT_MIN_C1       = 30     # C1: Bedrock only, short timeout

INSTANCE_TYPE_C1     = os.getenv("INSTANCE_TYPE_C1", "t3.xlarge")   # C1: no GPU required
AMI_ID_C1            = os.getenv("AMI_ID_C1", AMI_ID)               # same AMI is fine

MODELS = [
    {"name": "qwen3:4b",         "tier": "A", "num_parallel": 12},
    {"name": "gemma3:4b",        "tier": "A", "num_parallel": 12},
    {"name": "phi4-mini:latest", "tier": "A", "num_parallel": 12},
    {"name": "qwen3:8b",         "tier": "B", "num_parallel": 8},
    {"name": "exaone3.5:7.8b",   "tier": "B", "num_parallel": 8},
    {"name": "gemma3:9b",        "tier": "B", "num_parallel": 8},
    {"name": "llama3.1:8b",      "tier": "B", "num_parallel": 8},
    {"name": "qwen3:14b",        "tier": "C", "num_parallel": 5},
    {"name": "gemma3:12b",       "tier": "C", "num_parallel": 6},
]

# C2 re-run targets (models that failed in the previous run)
C2_RERUN_MODELS = [
    {"name": "exaone3.5:7.8b",   "tier": "B", "num_parallel": 8},
    {"name": "gemma3:4b",        "tier": "A", "num_parallel": 12},
    {"name": "llama3.1:8b",      "tier": "B", "num_parallel": 8},
    {"name": "phi4-mini:latest", "tier": "A", "num_parallel": 12},
    {"name": "qwen3:14b",        "tier": "C", "num_parallel": 5},
    {"name": "gemma3:12b",       "tier": "C", "num_parallel": 6},
]


def _safe(name: str) -> str:
    return name.replace(":", "_").replace(".", "_")


def _make_user_data_c2(model: dict, run_id: str, lang: str = "en") -> str:
    """C2/all mode: install Ollama, pull model, and run benchmark."""
    model_name   = model["name"]
    num_parallel = model["num_parallel"]
    bench_mode   = "c2"   # Layer 1 + Layer 3 C2
    return f"""#!/bin/bash
set -eo pipefail
exec >> /home/ubuntu/init.log 2>&1
echo "=== START $(date) ==="
export HOME=/home/ubuntu
export MODEL="{model_name}"
export BENCH_MODE="{bench_mode}"
export LANG_MODE="{lang}"
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_NUM_PARALLEL={num_parallel}
export S3_BUCKET="{S3_BUCKET}"
export S3_PREFIX="{S3_PREFIX}"
export RUN_ID="{run_id}"
export AWS_REGION="{AWS_REGION}"
export SONNET_MODEL="{SONNET_MODEL}"
export PYTHONUNBUFFERED=1

# ── Install and start Ollama ──────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable ollama 2>/dev/null || true
systemctl start  ollama 2>/dev/null || true
sleep 5

for i in $(seq 1 12); do
    curl -sf http://localhost:11434/api/version && break
    echo "Waiting for Ollama... ($i/12)"
    sleep 5
done

# ── Pull model ────────────────────────────────────────────────────────────────
echo "Pulling model: $MODEL"
OLLAMA_NUM_PARALLEL={num_parallel} ollama pull "$MODEL"
echo "Model pull complete"

# ── Python dependencies ───────────────────────────────────────────────────────
pip install httpx anthropic boto3 --quiet 2>&1 | tail -5

# ── Download benchmark script ─────────────────────────────────────────────────
aws s3 cp "s3://{S3_BUCKET}/benchmark-scripts/standalone_benchmark.py" \\
    /home/ubuntu/standalone_benchmark.py

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== BENCHMARK START $(date) ==="
cd /home/ubuntu
python3 standalone_benchmark.py --mode {bench_mode} --lang {lang} 2>&1 | tee /home/ubuntu/benchmark.log

# ── Upload log + completion signal ────────────────────────────────────────────
aws s3 cp /home/ubuntu/benchmark.log \\
    "s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/{_safe(model_name)}/benchmark.log"
echo "DONE" | aws s3 cp - \\
    "s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/{_safe(model_name)}/DONE"
echo "=== DONE $(date) ==="
shutdown -h now
"""


def _make_user_data_c1(run_id: str, lang: str = "en") -> str:
    """C1 mode: run Bedrock Sonnet baseline only, no Ollama (t3.xlarge)."""
    model_key = "c1_baseline"
    return f"""#!/bin/bash
set -eo pipefail
exec >> /home/ubuntu/init.log 2>&1
echo "=== START $(date) ==="
export HOME=/home/ubuntu
export MODEL="{model_key}"
export BENCH_MODE="c1"
export LANG_MODE="{lang}"
export S3_BUCKET="{S3_BUCKET}"
export S3_PREFIX="{S3_PREFIX}"
export RUN_ID="{run_id}"
export AWS_REGION="{AWS_REGION}"
export SONNET_MODEL="{SONNET_MODEL}"
export PYTHONUNBUFFERED=1

# ── Python dependencies ───────────────────────────────────────────────────────
pip install httpx anthropic boto3 --quiet 2>&1 | tail -5

# ── Download benchmark script ─────────────────────────────────────────────────
aws s3 cp "s3://{S3_BUCKET}/benchmark-scripts/standalone_benchmark.py" \\
    /home/ubuntu/standalone_benchmark.py

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== BENCHMARK START $(date) ==="
cd /home/ubuntu
python3 standalone_benchmark.py --mode c1 --lang {lang} 2>&1 | tee /home/ubuntu/benchmark.log

# ── Upload log + completion signal ────────────────────────────────────────────
aws s3 cp /home/ubuntu/benchmark.log \\
    "s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/{model_key}/benchmark.log"
echo "DONE" | aws s3 cp - \\
    "s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/{model_key}/DONE"
echo "=== DONE $(date) ==="
shutdown -h now
"""


def upload_benchmark_script(s3_client) -> None:
    """Upload standalone_benchmark.py to S3."""
    script_path = ROOT / "eval" / "standalone_benchmark.py"
    if not script_path.exists():
        raise FileNotFoundError(f"standalone_benchmark.py not found: {script_path}")
    key = "benchmark-scripts/standalone_benchmark.py"
    s3_client.upload_file(str(script_path), S3_BUCKET, key)
    print(f"  Uploaded script → s3://{S3_BUCKET}/{key}")


def launch_instance(ec2_client, model_name: str, user_data: str,
                    instance_type: str, ami: str, run_id: str, dry_run: bool) -> dict | None:
    """Launch a single EC2 instance."""
    launch_kwargs: dict = dict(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",        "Value": f"bench-{_safe(model_name)}-{run_id}"},
                {"Key": "BenchRunId",  "Value": run_id},
                {"Key": "BenchModel",  "Value": model_name},
                {"Key": "Project",     "Value": "deep-research-agent-benchmark"},
            ],
        }],
    )
    if SECURITY_GROUP_ID:
        launch_kwargs["SecurityGroupIds"] = [SECURITY_GROUP_ID]
    if KEY_NAME:
        launch_kwargs["KeyName"] = KEY_NAME
    if IAM_INSTANCE_PROFILE:
        launch_kwargs["IamInstanceProfile"] = {"Name": IAM_INSTANCE_PROFILE}
    launch_kwargs["BlockDeviceMappings"] = [{
        "DeviceName": "/dev/sda1",
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True},
    }]

    if dry_run:
        print(f"    [DRY RUN] Would launch {instance_type} for {model_name}")
        return {"InstanceId": f"i-dryrun-{_safe(model_name)}", "dry_run": True}

    # Fall back through AZs in order when capacity is insufficient (1b → 1c → 1d → 1a)
    # Set SUBNET_IDS env var as comma-separated list, or replace placeholders below
    # e.g. export SUBNET_IDS="subnet-aaa,subnet-bbb,subnet-ccc,subnet-ddd"
    _subnet_env = os.getenv("SUBNET_IDS", "")
    SUBNET_FALLBACKS = (
        [s.strip() for s in _subnet_env.split(",") if s.strip()]
        if _subnet_env else [
            "subnet-XXXXXXXXXXXX1",  # us-east-1b  ← replace with your subnet ID
            "subnet-XXXXXXXXXXXX2",  # us-east-1c  ← replace with your subnet ID
            "subnet-XXXXXXXXXXXX3",  # us-east-1d  ← replace with your subnet ID
            "subnet-XXXXXXXXXXXX4",  # us-east-1a  ← replace with your subnet ID
        ]
    )
    last_error = None
    for subnet_id in SUBNET_FALLBACKS:
        try:
            resp = ec2_client.run_instances(**{**launch_kwargs, "SubnetId": subnet_id})
            az = resp["Instances"][0].get("Placement", {}).get("AvailabilityZone", "?")
            print(f"      (AZ: {az})")
            return resp["Instances"][0]
        except Exception as e:
            if "InsufficientInstanceCapacity" in str(e):
                last_error = e
                continue
            raise
    raise last_error


def wait_for_result(s3_client, model_name: str, run_id: str, timeout_min: int) -> dict | None:
    """Poll until the DONE file appears in S3, then download results.json."""
    safe   = _safe(model_name)
    done_key   = f"{S3_PREFIX}/{run_id}/{safe}/DONE"
    result_key = f"{S3_PREFIX}/{run_id}/{safe}/results.json"
    deadline   = time.time() + timeout_min * 60

    print(f"    Waiting for {model_name} (timeout={timeout_min}min) ...", flush=True)
    while time.time() < deadline:
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=done_key)
            # DONE file confirmed → download results.json
            try:
                r = s3_client.get_object(Bucket=S3_BUCKET, Key=result_key)
                return json.loads(r["Body"].read())
            except Exception as e:
                print(f"    WARNING: DONE exists but results.json missing: {e}")
                return {"model": model_name, "error": "results.json missing after DONE"}
        except s3_client.exceptions.ClientError:
            pass  # not yet
        time.sleep(POLL_INTERVAL_SEC)

    return {"model": model_name, "error": f"timeout after {timeout_min}min"}


def print_leaderboard(all_results: list[dict]) -> None:
    print(f"\n\n{'═'*72}")
    print("  LEADERBOARD — Layer 1 + Layer 3")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'L1 Score':>9} {'C1 Score':>9} {'C2 Score':>9} {'Gap':>7}  Status")
    print(f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*7}  {'─'*8}")

    for r in sorted(all_results, key=lambda x: x.get("layer3", {}).get("c2_hybrid", {}).get("avg_overall_score", 0), reverse=True):
        model = r.get("model", "?")
        if "error" in r and "layer1" not in r:
            print(f"  {model:<22}  ERROR: {str(r.get('error',''))[:40]}")
            continue

        l1  = r.get("layer1", {}).get("weighted_score", 0)
        l3  = r.get("layer3", {})
        c1  = l3.get("c1_sonnet_only", {}).get("avg_overall_score", 0)
        c2  = l3.get("c2_hybrid",      {}).get("avg_overall_score", 0)
        gap = l3.get("quality_gap", c1 - c2)

        status = "✅" if c2 >= 0.6 and abs(gap) <= 0.10 else "⚠️ " if c2 >= 0.5 else "❌"
        print(f"  {model:<22}  {l1:>7.3f}    {c1:>7.3f}    {c2:>7.3f}    {gap:>+6.3f}  {status}")

    print(f"\n  Criteria: C2 score ≥ 0.6 and quality gap ≤ 0.10 → ✅ recommended")
    print(f"{'═'*72}")


async def main(args: argparse.Namespace) -> None:
    mode   = args.mode
    lang   = args.lang
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'═'*60}")
    print(f"  EC2 BENCHMARK DEPLOYMENT  [mode={mode}  lang={lang}]")
    print(f"  Run ID: {run_id}")
    print(f"  Region: {AWS_REGION}")
    print(f"  S3: s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/")
    print(f"{'═'*60}\n")

    if not args.dry_run:
        for var, val in [("S3_BUCKET", S3_BUCKET), ("AMI_ID", AMI_ID),
                         ("SECURITY_GROUP_ID", SECURITY_GROUP_ID),
                         ("IAM_INSTANCE_PROFILE", IAM_INSTANCE_PROFILE)]:
            if not val:
                print(f"  ERROR: {var} environment variable is required")
                sys.exit(1)

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    s3  = boto3.client("s3",  region_name=AWS_REGION)

    # 1. Upload script
    if not args.dry_run:
        print("  Uploading benchmark script to S3 ...")
        upload_benchmark_script(s3)

    # 2. Determine run list + launch instances
    instances: list[dict] = []   # {"model": str, "InstanceId": str, ...}
    timeout_min = TIMEOUT_MIN

    if mode == "c1":
        # ── C1: single t3.xlarge, no Ollama required ──────────────────────────
        timeout_min = TIMEOUT_MIN_C1
        print(f"  [C1] Launching 1× {INSTANCE_TYPE_C1} for Sonnet baseline ...\n")
        ud = _make_user_data_c1(run_id, lang=lang)
        try:
            inst = launch_instance(ec2, "c1_baseline", ud,
                                   INSTANCE_TYPE_C1, AMI_ID_C1 or AMI_ID,
                                   run_id, dry_run=args.dry_run)
            if inst:
                inst["model"] = "c1_baseline"
                instances.append(inst)
                print(f"    ✓ c1_baseline → {inst['InstanceId']}")
        except Exception as e:
            print(f"    ✗ c1_baseline → ERROR: {e}")

    else:
        # ── C2 / all: g6e.xlarge, one instance per model ─────────────────────
        if args.models:
            names = [m.strip() for m in args.models.split(",")]
            models_to_run = [m for m in MODELS if m["name"] in names]
            if not models_to_run:
                print(f"No matching models: {names}")
                sys.exit(1)
        elif mode == "c2":
            models_to_run = C2_RERUN_MODELS   # 6 previously failed models
        else:
            models_to_run = MODELS             # all: full 9-model set

        inst_type = INSTANCE_TYPE
        print(f"  Launching {len(models_to_run)}× {inst_type} [{mode} mode] ...\n")
        print(f"  Models: {[m['name'] for m in models_to_run]}\n")

        bench_mode = "c2" if mode == "c2" else "all"
        for model in models_to_run:
            ud = _make_user_data_c2(model, run_id, lang=lang)
            # In all mode, run both C1+C2
            if mode == "all":
                # Override BENCH_MODE to "all" in user_data
                ud = ud.replace('export BENCH_MODE="c2"', 'export BENCH_MODE="all"')
                ud = ud.replace("--mode c2", "--mode all")
            try:
                inst = launch_instance(ec2, model["name"], ud,
                                       inst_type, AMI_ID, run_id, dry_run=args.dry_run)
                if inst:
                    inst["model"] = model["name"]
                    instances.append(inst)
                    print(f"    ✓ {model['name']:<22} → {inst['InstanceId']}")
            except Exception as e:
                print(f"    ✗ {model['name']:<22} → ERROR: {e}")

    if args.dry_run:
        print("\n  [DRY RUN] Complete — no instances launched")
        return

    # 3. Wait for results
    print(f"\n  Waiting for results (poll={POLL_INTERVAL_SEC}s, timeout={timeout_min}min) ...\n")

    async def wait_one(model_name: str) -> dict:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, wait_for_result, s3, model_name, run_id, timeout_min
        )
        verdict = "✅ done" if "error" not in result else f"❌ {result.get('error','')[:40]}"
        print(f"    {model_name:<22} {verdict}")
        return result

    all_results = list(await asyncio.gather(*[wait_one(inst["model"]) for inst in instances]))

    # 4. Save results
    summary_path = ROOT / "eval" / "results" / "standalone" / f"_{run_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "run_id": run_id, "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": all_results,
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2))
    print(f"\n  Summary saved → {summary_path}")

    try:
        key = f"{S3_PREFIX}/{run_id}/_summary.json"
        s3.put_object(Bucket=S3_BUCKET, Key=key,
                      Body=json.dumps(summary_data, ensure_ascii=False, indent=2).encode(),
                      ContentType="application/json")
        print(f"  Summary → s3://{S3_BUCKET}/{key}")
    except Exception as e:
        print(f"  S3 summary upload failed: {e}")

    # 5. Leaderboard (all / c2 mode only)
    if mode != "c1":
        print_leaderboard(all_results)
    else:
        r = all_results[0] if all_results else {}
        c1 = r.get("layer3", {}).get("c1_sonnet_only", {})
        if "avg_overall_score" in c1:
            print(f"\n  C1 (Sonnet baseline): avg_overall={c1['avg_overall_score']:.3f}  "
                  f"revisions={c1['avg_revision_count']:.1f}  latency={c1['avg_latency_sec']:.0f}s")

    instance_ids = [i["InstanceId"] for i in instances if not i.get("dry_run")]
    if instance_ids:
        print(f"\n  Instances auto-terminate via 'shutdown -h now'.")
        print(f"  Manual termination if needed:")
        print(f"  aws ec2 terminate-instances --instance-ids {' '.join(instance_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC2 benchmark deployment")
    parser.add_argument("--mode",    default="c2", choices=["c1", "c2", "all"],
                        help="c1=Sonnet baseline (1 instance) | c2=hybrid re-run | all=full run")
    parser.add_argument("--models",  default=None, help="comma-separated model names (c2/all mode)")
    parser.add_argument("--lang",    default="en", choices=["en", "ko", "ja"],
                        help="Layer 3 query language: en=English (default) ko=Korean ja=Japanese")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
