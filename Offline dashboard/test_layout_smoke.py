"""Explicit desktop check for aspect-ratio stability and scrolling."""
import tkinter as tk
import numpy as np
from desktop_app import AquaVisionDesktop


def main():
    root = tk.Tk()
    app = AquaVisionDesktop(root)
    try:
        root.geometry('900x700')
        root.update()
        for h, w in [(1080, 1920), (1920, 1080), (1200, 1200), (400, 3000)]:
            for _ in range(4):
                app.display_image(np.zeros((h, w, 3), dtype=np.uint8))
                root.update()
                assert root.winfo_width() == 900 and root.winfo_height() == 700
                pw, ph = app.photo.width(), app.photo.height()
                assert pw <= app.video_viewport.winfo_width()
                assert ph <= app.video_viewport.winfo_height()
                assert abs(pw / ph - w / h) < .06
        root.geometry('600x420')
        root.update()
        panel = app.panels[0]
        assert panel.body.winfo_height() > panel.canvas.winfo_height()
        panel.canvas.yview_moveto(1)
        root.update()
        assert panel.canvas.yview()[0] > 0
        assert panel.canvas.yview()[1] > .99
        print('PASS: landscape, portrait, square, ultrawide, stable window size, small-window scrolling')
    finally:
        app.close()


if __name__ == '__main__':
    main()
