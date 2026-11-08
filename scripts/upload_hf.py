"""Upload only the prepared neural-model files to your own private Hub repo."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
ALLOW_FILES = [
    "README.md", "config.json", "model.safetensors", "model.py", "metrics.json",
    "sample_input.json", "prediction_example.json", "requirements.txt", "inference_example.py",
    "assets/red_neuronal.svg", "assets/validacion_temporal.svg",
]


def check_artifacts(folder):
    folder = Path(folder)
    for name in ALLOW_FILES:
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing regular model file: " + name)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
    if digest != config["weights_sha256"]:
        raise ValueError("The model weights do not match config.json")
    return [{"file": name, "bytes": (folder / name).stat().st_size} for name in ALLOW_FILES]


def upload_model(api, folder, name, downloader=hf_hub_download):
    folder = Path(folder)
    check_artifacts(folder)
    identity = api.whoami()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("Use a model name, without an account or path")
    repo_id = identity["name"] + "/" + name
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if not info.private:
            raise ValueError("Use a different name: this repository already exists with public access")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", visibility="private", exist_ok=False)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=ALLOW_FILES, commit_message="Upload trained Mexican yield neural network")
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit.oid)
    if not set(ALLOW_FILES).issubset(files):
        raise ValueError("Some model files are missing from the remote commit")
    downloaded = Path(downloader(repo_id=repo_id, filename="model.safetensors", revision=commit.oid))
    expected = hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
        raise ValueError("Uploaded weights failed the readback checksum")
    return {"repo_id": repo_id, "revision": commit.oid, "visibility": "private",
            "weights_sha256": expected, "model_url": "https://huggingface.co/" + repo_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mexico-yield-neural")
    parser.add_argument("--dry-run", action="store_true", help="List files without using the network")
    args = parser.parse_args()
    folder = ROOT / "hf_model"
    files = check_artifacts(folder)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_upload", "files": files}, indent=2))
        return
    receipt = upload_model(HfApi(), folder, args.name)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/hf_upload_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Keep credentials and HTTP request internals out of terminal output.
        print("Upload not completed (" + type(exc).__name__ +
              "). Check hf auth login, write permission, repository name and model files.", file=sys.stderr)
        sys.exit(1)
