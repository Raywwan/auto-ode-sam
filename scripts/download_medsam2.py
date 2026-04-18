"""Download MedSAM2 Hiera-Tiny checkpoint to checkpoints/medsam2/."""
from pathlib import Path
from urllib.request import urlretrieve

MEDSAM2_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "medsam2"
OUT_PATH = OUT_DIR / "MedSAM2_hiera_tiny.pt"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading MedSAM2 to {OUT_PATH} ...")
    urlretrieve(MEDSAM2_URL, OUT_PATH)
    print(f"Done: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
