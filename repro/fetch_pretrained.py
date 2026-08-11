"""Download the pre-trained Kronos checkpoints this reproduction fine-tunes.

Both repositories are MIT-licensed and ungated. The revisions pinned below are
upstream's own: they are the exact commits `tests/test_kronos_regression.py`
pins (TOKENIZER_REVISION / MODEL_REVISION), so the weights this row fine-tunes
are the weights upstream tests against.

Weights land in `pretrained/<name>/`, which is the path
`finetune_csv/configs/config_repro_kronos-small_1epoch.yaml` reads.

This file is not part of upstream Kronos; it exists only so the download is a
separate, traceable pipeline stage rather than a side effect of training.
"""

import os

from huggingface_hub import snapshot_download

REPOS = [
    # (repo_id, revision, local dir)
    ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426",
     "pretrained/Kronos-Tokenizer-base"),
    ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320",
     "pretrained/Kronos-small"),
]


def main() -> None:
    for repo_id, revision, local_dir in REPOS:
        print(f"downloading {repo_id}@{revision[:8]} -> {local_dir}", flush=True)
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=["config.json", "model.safetensors"],
        )
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                print(f"  {name}  {os.path.getsize(full)} bytes", flush=True)


if __name__ == "__main__":
    main()
