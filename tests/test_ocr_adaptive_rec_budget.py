"""Exercise one OCR handle while the active REC line-worker count changes."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("benchmark", "det", "cls", "rec", "dictionary", "sample"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    # The build's 500x500 P6 fixture comes from the bundled sample.jpg. Keep
    # its first text line and blank the rest without requiring Pillow/OpenCV.
    sample = args.sample.resolve()
    ppm = sample.read_bytes()
    header = b"P6\n500 500\n255\n"
    width, height, retained_rows = 500, 500, 82
    if not ppm.startswith(header) or len(ppm) != len(header) + width * height * 3:
        raise AssertionError("unexpected bundled sample PPM format")

    with tempfile.TemporaryDirectory(prefix="lw-ocr-rec-budget-") as temporary:
        root = Path(temporary)
        sparse = root / "one-line.ppm"
        first_line = ppm[len(header) : len(header) + width * retained_rows * 3]
        sparse.write_bytes(
            header
            + first_line
            + bytes([255]) * (width * (height - retained_rows) * 3)
        )
        two_lines = root / "two-lines.ppm"
        pixels = bytearray(bytes([255]) * (width * height * 3))
        pixels[: len(first_line)] = first_line
        second_row = 150
        offset = second_row * width * 3
        pixels[offset : offset + len(first_line)] = first_line
        two_lines.write_bytes(header + pixels)
        image_list = root / "images.txt"
        image_list.write_text(
            f"{sparse}\n{two_lines}\n{sample}\n{sparse}\n", encoding="utf-8"
        )
        command = [
            str(args.benchmark.resolve()),
            str(args.det.resolve()),
            str(args.cls.resolve()),
            str(args.rec.resolve()),
            str(args.dictionary.resolve()),
            str(image_list),
            "1", "2", "4", "960",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            raise AssertionError(completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        assert report["images"] == 4, report
        assert report["lines"] == 20, report
        assert report["workers"] == 4, report
        assert report["iterations"] == 2, report
        assert len(report["output_checksum"]) == 16, report
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
