"""Fetch the fixed COCO val2017 subset for the vision benchmark: the first 500 val2017 image names in sorted order,
with YOLO-format box labels from Ultralytics' coco2017labels.zip. Stdlib only.

Every image is checked against coco_val2017_first500.txt (name, sha256) and the labels zip against LABELS_SHA256.
The script also checks that the manifest really is the first 500 sorted names of the zip's val2017.txt.

python benchmark/vision/fetch_coco_subset.py --out /tmp/coco500 [--limit 5]

Writes <out>/images/val2017/*.jpg, <out>/labels/val2017/*.txt (Ultralytics layout; images without objects have no
label file) and <out>/val.txt (image list for an Ultralytics dataset YAML). Downloads are cached in <out>.
COCO annotations are CC BY 4.0; images keep their Flickr licenses. Nothing from COCO is committed to the repo.
"""

import argparse
import hashlib
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "coco_val2017_first500.txt"
LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
)
LABELS_SHA256 = "51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234"
IMAGE_URL = "http://images.cocodataset.org/val2017/{}"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2**i)
    raise AssertionError("unreachable")


def cached(path, url, sha):
    """Bytes of path if it matches sha, else download url, check sha, write path."""
    if path.exists() and sha256(path.read_bytes()) == sha:
        return path.read_bytes()
    data = download(url)
    if sha256(data) != sha:
        raise ValueError(f"{url}: sha256 {sha256(data)} != pinned {sha}")
    path.write_bytes(data)
    return data


def manifest():
    return [
        ln.split()
        for ln in MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="first N manifest images only (0 = all 500)",
    )
    args = p.parse_args()
    out = Path(args.out)
    img_dir, lbl_dir = out / "images" / "val2017", out / "labels" / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    full = manifest()
    entries = full[: args.limit or None]

    cached(out / "coco2017labels.zip", LABELS_URL, LABELS_SHA256)
    with zipfile.ZipFile(out / "coco2017labels.zip") as z:
        val = sorted(Path(x).name for x in z.read("coco/val2017.txt").decode().split())
        if val[: len(full)] != [n for n, _ in full]:
            raise ValueError(
                "manifest is not the first N sorted names of coco/val2017.txt"
            )
        names = set(z.namelist())
        instances = 0
        for name, _ in entries:
            member = f"coco/labels/val2017/{Path(name).stem}.txt"
            if member in names:
                text = z.read(member)
                instances += len(text.splitlines())
                (lbl_dir / f"{Path(name).stem}.txt").write_bytes(text)

    with ThreadPoolExecutor(8) as ex:
        list(
            ex.map(
                lambda e: cached(img_dir / e[0], IMAGE_URL.format(e[0]), e[1]), entries
            )
        )
    (out / "val.txt").write_text(
        "".join(f"./images/val2017/{n}\n" for n, _ in entries), encoding="utf-8"
    )
    labelled = sum(1 for n, _ in entries if (lbl_dir / f"{Path(n).stem}.txt").exists())
    print(
        f"{len(entries)} images (sha256 ok), {labelled} with labels, {instances} boxes -> {out}"
    )


if __name__ == "__main__":
    main()
