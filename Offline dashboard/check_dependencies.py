"""Check the exact interpreter used by run.bat; no runtime downloads."""
import importlib
import sys
import os
from pathlib import Path
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))

PACKAGES = [('torch', 'CPU PyTorch'), ('torchvision', 'torchvision'),
            ('cv2', 'opencv-python'), ('ultralytics', 'ultralytics'),
            ('lap', 'lap'), ('tap', 'typed-argument-parser'),
            ('pandas', 'pandas'), ('PIL', 'Pillow'), ('tkinter', 'Python Tkinter')]


def main():
    failures = []
    print('Python:', sys.executable)
    for module, package in PACKAGES:
        try:
            imported = importlib.import_module(module)
            if module == 'tap' and not hasattr(imported, 'Tap'):
                raise ImportError('Wrong tap package: install typed-argument-parser')
        except Exception as exc:
            failures.append((package, str(exc)))
    if failures:
        for package, error in failures:
            print(f'MISSING OR BROKEN: {package}: {error}')
        print('Close the app and run setup_windows.bat in this folder.')
        return 1
    print('All application dependencies are available, including lap and tap.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
