"""Run explicitly with desktop access: python test_desktop_smoke.py."""
import time
import tkinter as tk
from unittest.mock import patch
from desktop_app import AquaVisionDesktop, dependency_error


def pump(root, predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while not predicate():
        root.update()
        time.sleep(.01)
        if time.monotonic() > deadline:
            raise TimeoutError('Desktop task did not complete')
    root.update()


def main():
    root = tk.Tk()
    root.withdraw()
    app = AquaVisionDesktop(root)
    errors = []
    try:
        with patch('desktop_app.messagebox.showerror', side_effect=lambda *a: errors.append(a)):
            app.open_sample()
            pump(root, lambda: app.future is None)
            assert app.video_path and app.photo
            app.play_original()
            pump(root, lambda: app.video_progress.get() > 0)
            app.toggle_pause()
            progress = app.video_progress.get()
            for _ in range(20):
                root.update()
                time.sleep(.01)
            assert app.video_progress.get() == progress
            app.toggle_pause()
            pump(root, lambda: app.future is None)
            assert app.video_progress.get() == 100
            app.play_original()
            pump(root, lambda: app.video_progress.get() > 0)
            app.stop()
            pump(root, lambda: app.future is None)
            app.use_demo_model()
            app.max_frames.set(8)
            app.start_video()
            pump(root, lambda: app.future is None, timeout=90)
            assert app.table is not None and app.video_progress.get() == 100
            app.load_video('missing_video.mp4')
            pump(root, lambda: app.future is None)
            assert len(errors) == 1 and app.video_path is None, errors
            assert 'typed-argument-parser' in dependency_error(ModuleNotFoundError(name='tap'))
            assert 'install lap' in dependency_error(ModuleNotFoundError(name='lap'))
            print('PASS: upload/preview, original playback, progress, pause/resume, replay, stop, real video analysis, unreadable video, dependency hints')
    finally:
        app.close()


if __name__ == '__main__':
    main()
