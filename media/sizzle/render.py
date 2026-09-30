"""Render sizzle.html frame by frame in headless Chromium and encode an MP4.

    python render.py                 # full 30 s video -> autodrive_sizzle.mp4
    python render.py --stills 45 150 # PNG stills of given frames, for review
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def chromium() -> str | None:
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return found[-1] if found else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=int)
    ap.add_argument("--out", default=str(HERE / "autodrive_sizzle.mp4"))
    ap.add_argument("--still-dir", default=str(HERE / "stills"))
    args = ap.parse_args()

    data = json.loads((HERE / "data.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium())
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto((HERE / "sizzle.html").as_uri())
        loaded = page.evaluate("""async () => {
            for (const f of ["780 100px Archivo", "400 20px 'IBM Plex Mono'", "500 20px 'IBM Plex Mono'",
                             "600 20px 'IBM Plex Mono'"]) await document.fonts.load(f);
            return [...document.fonts].filter(f => f.status === "loaded").map(f => f.family);
        }""")
        # document.fonts.check() is true even when a font is missing, so count real loads.
        assert {"Archivo", "IBM Plex Mono"} <= set(loaded), f"web fonts failed to load: {loaded}"
        page.evaluate("d => window.init(d)", data)
        total = page.evaluate("window.TOTAL")

        if args.stills is not None:
            Path(args.still_dir).mkdir(exist_ok=True)
            for f in args.stills:
                page.evaluate(f"window.renderFrame({f})")
                page.screenshot(path=f"{args.still_dir}/f{f:04d}.png")
            print(f"wrote {len(args.stills)} stills to {args.still_dir}")
            return 0

        ffmpeg = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "image2pipe", "-framerate", "30", "-c:v", "png", "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", args.out],
            stdin=subprocess.PIPE)
        for f in range(total):
            page.evaluate(f"window.renderFrame({f})")
            ffmpeg.stdin.write(page.screenshot(type="png"))
            if f % 90 == 0:
                print(f"  frame {f}/{total}", flush=True)
        ffmpeg.stdin.close()
        ffmpeg.wait()
        browser.close()
    print(f"wrote {args.out}")
    return ffmpeg.returncode


if __name__ == "__main__":
    sys.exit(main())
