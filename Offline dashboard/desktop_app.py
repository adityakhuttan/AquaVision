"""Offline AquaVision desktop; all Tk calls stay on the main thread."""
import pipeline  # PyTorch must load before OpenCV on Windows.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from queue import Queue, Empty, Full
import time
import cv2
from PIL import Image, ImageTk
import dataset_check


def dependency_error(exc):
    if isinstance(exc, ModuleNotFoundError) and exc.name in {'tap', 'lap'}:
        package = 'typed-argument-parser' if exc.name == 'tap' else 'lap'
        return f"Missing {exc.name}. Close AquaVision and run setup_windows.bat to repair dependencies.\nManual repair in this folder:\n.venv\\Scripts\\python -m pip install {package}"
    return str(exc)


class ScrollablePanel(ttk.Frame):
    """Keep controls reachable on small screens and Windows display scaling."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vertical = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        horizontal = ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.body = ttk.Frame(self.canvas, padding=12)
        self.window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', self.update_region)
        self.canvas.bind('<Configure>', self.resize_body)
        # Route wheel events to this panel, including events over its children.
        parent.winfo_toplevel().bind('<MouseWheel>', self.wheel, add='+')
        parent.winfo_toplevel().bind('<Button-4>', self.wheel, add='+')
        parent.winfo_toplevel().bind('<Button-5>', self.wheel, add='+')

    def update_region(self, event=None):
        self.canvas.itemconfigure(self.window, height=max(self.canvas.winfo_height(), 620, self.body.winfo_reqheight()))
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def resize_body(self, event):
        self.canvas.itemconfigure(self.window, width=max(event.width, 780), height=max(event.height, 620, self.body.winfo_reqheight()))

    def wheel(self, event):
        widget = event.widget
        while widget is not None and widget is not self:
            widget = getattr(widget, 'master', None)
        if widget is self:
            delta = -1 if getattr(event, 'num', None) == 4 else 1 if getattr(event, 'num', None) == 5 else (-1 if event.delta > 0 else 1)
            self.canvas.yview_scroll(delta, 'units')
            return 'break'


class AquaVisionDesktop:
    def __init__(self, root):
        self.root = root
        root.title('AquaVision · Offline Desktop')
        root.geometry('1080x760')
        root.minsize(600, 420)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.stop_event = Event()
        self.frames = Queue(maxsize=2)
        self.paused = False
        self.closed = False
        self.current_image_rgb = None
        self.video_path = None
        self.table = None
        self.gallery = []
        self.gallery_index = 0
        self.last_frame_time = 0
        self.pending_frame = None
        self.status = tk.StringVar(value='Select local model weights, then open an image or video.')
        self.model_banner = tk.StringVar(value='Model setup required: select your trained detector in Models & settings.')
        ttk.Label(root, textvariable=self.model_banner, padding=8, wraplength=1100).pack(fill='x')
        tabs = ttk.Notebook(root)
        self.tabs = tabs
        tabs.pack(fill='both', expand=True)
        self.panels = [ScrollablePanel(tabs) for _ in range(3)]
        media, datasets, models = [panel.body for panel in self.panels]
        for title, panel in zip(['Home', 'Datasets', 'Models & settings'], self.panels):
            tabs.add(panel, text=title)
        self.video_name = tk.StringVar(value='No video selected — browse or use the sample')
        upload = ttk.LabelFrame(media, text='1. Choose a video', padding=8)
        upload.pack(fill='x')
        ttk.Entry(upload, textvariable=self.video_name, state='readonly').pack(side='left', fill='x', expand=True, padx=(0, 8))
        ttk.Button(upload, text='Upload video…', command=self.open_video).pack(side='left')
        ttk.Button(upload, text='Use sample', command=self.open_sample).pack(side='left', padx=(8, 0))
        actions = ttk.Frame(media)
        actions.pack(fill='x', pady=8)
        for label, command in [('Play original', self.play_original), ('Analyse video', self.start_video)]:
            ttk.Button(actions, text=label, command=command).pack(side='left', padx=(0, 6))
        self.pause_button = ttk.Button(actions, text='Pause', command=self.toggle_pause)
        self.pause_button.pack(side='left', padx=(0, 6))
        ttk.Button(actions, text='Stop', command=self.stop).pack(side='left')
        ttk.Button(actions, text='Save CSV', command=self.export_table).pack(side='right')
        self.display_source = None
        self.video_viewport = tk.Frame(media, bg='#101923', height=280, width=400)
        self.video_viewport.pack(fill='both', expand=True)
        self.video_viewport.pack_propagate(False)
        self.video_viewport.bind('<Configure>', self.resize_display)
        self.image_label = tk.Label(self.video_viewport, bg='#101923', fg='white', text='Upload a video or click Use sample', font=('Arial', 15))
        self.image_label.pack(fill='both', expand=True)
        self.video_progress = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(media, variable=self.video_progress, maximum=100)
        self.progress_bar.pack(fill='x', pady=(8, 0))
        self.progress_text = tk.StringVar(value='Ready')
        ttk.Label(media, textvariable=self.progress_text).pack(anchor='w')
        bottom = ttk.Frame(media)
        bottom.pack(fill='x', pady=6)
        ttk.Label(bottom, text='Image tools:').pack(side='left')
        ttk.Button(bottom, text='Open image', command=self.upload_image).pack(side='left', padx=6)
        ttk.Button(bottom, text='Analyse image', command=self.process_image).pack(side='left')
        self.results_text = tk.Text(media, height=4, wrap='word')
        self.results_text.pack(fill='x')
        advanced = ttk.LabelFrame(models, text='Analysis options', padding=8)
        advanced.pack(fill='x')
        self.conf = tk.DoubleVar(value=.25)
        self.iou = tk.DoubleVar(value=.5)
        for label, var in [('Detection confidence', self.conf), ('NMS IoU', self.iou)]:
            row = ttk.Frame(advanced)
            row.pack(fill='x')
            ttk.Label(row, text=label, width=28).pack(side='left')
            tk.Scale(row, variable=var, from_=.05, to=.95, resolution=.05, orient='horizontal', length=260).pack(side='left')
        self.species = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced, text='Classify species', variable=self.species).pack(anchor='w')
        self.max_frames = tk.IntVar(value=0)
        self.calibration = tk.DoubleVar(value=0)
        for label, var in [('Maximum frames (0 = all)', self.max_frames), ('Pixels per cm (0 = uncalibrated)', self.calibration)]:
            row = ttk.Frame(advanced)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, width=38).pack(side='left')
            ttk.Entry(row, textvariable=var, width=14).pack(side='left')
        self.det_path = tk.StringVar(value=pipeline.CFG['det'])
        self.cls_path = tk.StringVar(value=pipeline.CFG['cls'])
        for label, var in [('Custom detector (.pt)', self.det_path), ('Species classifier (.pt)', self.cls_path)]:
            ttk.Label(models, text=label).pack(anchor='w', pady=(12, 0))
            ttk.Entry(models, textvariable=var, width=100).pack(fill='x')
            ttk.Button(models, text='Browse…', command=lambda v=var: self.pick_model(v)).pack(anchor='w')
        ttk.Button(models, text='Apply models', command=self.apply_models).pack(anchor='w', pady=16)
        ttk.Button(models, text='Use included stock model (demo only)', command=self.use_demo_model).pack(anchor='w')
        self.model_text = tk.StringVar(value=pipeline.model_status())
        ttk.Label(models, textvariable=self.model_text, wraplength=950).pack(anchor='w')
        ttk.Label(models, text='The supplied yolo11s-seg.pt is a generic COCO model, not a trained fish detector.\nSpecies remain Unknown until your team supplies a compatible classification model.\nLength and size are estimates; centimetres require calibration at the fish depth.', wraplength=950).pack(anchor='w', pady=20)
        self.paths = {}
        for name in dataset_check.NAMES:
            row = ttk.Frame(datasets)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=name, width=18).pack(side='left')
            var = tk.StringVar(value=dataset_check.DEFAULT_PATHS[name])
            self.paths[name] = var
            ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(row, text='Browse…', command=lambda v=var: self.pick_folder(v)).pack(side='left')
        controls = ttk.Frame(datasets)
        controls.pack(fill='x', pady=8)
        self.dataset_name = tk.StringVar(value=dataset_check.NAMES[0])
        ttk.Combobox(controls, textvariable=self.dataset_name, values=dataset_check.NAMES, state='readonly', width=18).pack(side='left')
        self.samples = tk.IntVar(value=12)
        self.per_class = tk.IntVar(value=5)
        self.seed = tk.IntVar(value=42)
        for label, var in [('Samples', self.samples), ('Per species', self.per_class), ('Seed', self.seed)]:
            ttk.Label(controls, text=label).pack(side='left', padx=5)
            ttk.Entry(controls, textvariable=var, width=5).pack(side='left')
        actions = ttk.Frame(datasets)
        actions.pack(fill='x')
        for label, command in [('Scan folder', self.scan_dataset), ('Test selected', self.test_dataset), ('Test all', self.test_all), ('Export CSV', self.export_table), ('Previous preview', lambda: self.preview(-1)), ('Next preview', lambda: self.preview(1))]:
            ttk.Button(actions, text=label, command=command).pack(side='left', padx=2)
        self.dataset_text = tk.Text(datasets, height=10, wrap='none')
        self.dataset_text.pack(fill='both', expand=True, pady=8)
        self.video_choice = tk.StringVar()
        self.video_list = ttk.Combobox(datasets, textvariable=self.video_choice, state='readonly')
        self.video_list.pack(fill='x')
        ttk.Button(datasets, text='Open dataset video in media tab', command=self.open_dataset_video).pack(anchor='w')
        self.preview_label = ttk.Label(datasets)
        self.preview_label.pack()
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1100).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(30, self.poll)

    def settings(self):
        st = pipeline.Settings(conf=self.conf.get(), iou=self.iou.get(), do_species=self.species.get(), max_frames=self.max_frames.get(), px_per_cm=self.calibration.get())
        if st.max_frames < 0 or st.px_per_cm < 0:
            raise ValueError('Frame limit and calibration must be nonnegative.')
        return st

    def busy(self):
        if self.future is not None:
            self.status.set('A task is running. Stop video or wait for it to finish.')
            return True
        return False

    def submit(self, kind, func):
        if self.busy():
            return
        self.stop_event.clear()
        self.paused = False
        self.pause_button.configure(text='Pause')
        if kind in {'image', 'video'} and not Path(pipeline.CFG['det']).is_file():
            self.future = None
            self.stop_event.set()
            self.tabs.select(2)
            self.status.set('Select a trained detector and Apply models, or choose the included demo model.')
            return
        self.future = (kind, self.pool.submit(func))
        self.status.set('Loading video…' if kind == 'loadvideo' else 'Processing…')

    def pick_model(self, var):
        path = filedialog.askopenfilename(filetypes=[('Model weights', '*.pt')])
        if path:
            var.set(path)

    def pick_folder(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def apply_models(self):
        if self.busy():
            return
        det = self.det_path.get().strip()
        cls = self.cls_path.get().strip()
        if not Path(det).is_file():
            messagebox.showerror('Detector required', 'Choose an existing detector .pt file with Browse.\nYour custom detector was not included in the original archive.\nFor a plumbing test, choose Use included stock model (demo only).')
            return
        pipeline.set_weights(det, cls)
        self.model_text.set(pipeline.model_status())
        demo = Path(det).name == 'yolo11s-seg.pt'
        self.model_banner.set('DEMO ONLY — generic COCO model selected. Outputs are not validated fish detections.' if demo else 'Local detector selected. Species require a separate classification model.')
        self.status.set('Local model paths applied. Models load when processing starts.')

    def use_demo_model(self):
        if self.busy():
            return
        demo = pipeline.BASE_DIR / 'yolo11s-seg.pt'
        if not demo.is_file():
            messagebox.showerror('Demo model missing', 'Extract the complete AquaVision-Windows-fixed.zip package, including yolo11s-seg.pt.')
            return
        self.det_path.set(str(demo))
        self.apply_models()

    def upload_image(self):
        if self.busy():
            return
        path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if path:
            bgr = cv2.imread(path)
            if bgr is None:
                messagebox.showerror('Image', 'Could not decode image.')
                return
            self.current_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_image_rgb)
            self.status.set('Image ready.')

    def process_image(self):
        if self.current_image_rgb is None:
            self.status.set('Open an image first.')
            return
        try:
            st, image = self.settings(), self.current_image_rgb.copy()
            self.submit('image', lambda: pipeline.process_image(image, st))
        except Exception as exc:
            messagebox.showerror('Settings', str(exc))

    def open_video(self):
        if self.busy():
            return
        path = filedialog.askopenfilename(filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.m4v'), ('All files', '*.*')])
        if path:
            self.load_video(path)

    def open_sample(self):
        if not self.busy():
            self.load_video(str(pipeline.BASE_DIR / 'samples' / 'fish_test.mp4'))

    def load_video(self, path):
        if self.busy():
            return
        self.video_path = None
        self.table = None
        self.video_name.set(str(path))
        self.video_progress.set(0)
        self.progress_text.set('Opening video…')
        self.submit('loadvideo', lambda: pipeline.video_preview(path))

    def play_original(self):
        if self.busy():
            return
        if not self.video_path:
            self.status.set('Upload a video first, or click Use sample.')
            return
        path = self.video_path
        self.video_progress.set(0)
        self.submit('playback', lambda: self.run_video(path, None))

    def start_video(self):
        if self.busy():
            return
        if not self.video_path:
            self.status.set('Upload a video first, or click Use sample.')
            return
        try:
            st, path = self.settings(), self.video_path
            self.table = None
            self.video_progress.set(0)
            self.submit('video', lambda: self.run_video(path, st))
        except Exception as exc:
            messagebox.showerror('Settings', str(exc))

    def run_video(self, path, st):
        stream = pipeline.iter_video(path, st, stop_event=self.stop_event) if st is not None else pipeline.iter_original_video(path, self.stop_event)
        try:
            while not self.stop_event.is_set():
                while self.paused and not self.stop_event.wait(.05):
                    pass
                if self.stop_event.is_set():
                    break
                try:
                    packet = next(stream)
                except StopIteration as done:
                    return done.value
                while not self.stop_event.is_set():
                    try:
                        self.frames.put(packet, timeout=.05)
                        break
                    except Full:
                        continue
            # Resume once to finalize partial track statistics after cancellation.
            try:
                next(stream)
            except StopIteration as done:
                return done.value
        finally:
            stream.close()

    def toggle_pause(self):
        if not self.future or self.future[0] not in {'video', 'playback'}:
            return
        self.paused = not self.paused
        self.pause_button.configure(text='Resume' if self.paused else 'Pause')
        self.status.set('Paused' if self.paused else 'Resumed')

    def stop(self):
        self.stop_event.set()
        self.paused = False
        self.pause_button.configure(text='Pause')

    def scan_dataset(self):
        name = self.dataset_name.get()
        path = self.paths[name].get()
        self.submit('scan', lambda: dataset_check.scan_summary(path, refresh=True))

    def test_dataset(self):
        try:
            name = self.dataset_name.get()
            path, n, per, seed, st = self.paths[name].get(), self.samples.get(), self.per_class.get(), self.seed.get(), self.settings()
            if n < 1 or per < 1:
                raise ValueError('Sample counts must be positive.')
            self.submit('dataset', lambda: dataset_check.sample_dataset(name, path, n, per, seed, st))
        except Exception as exc:
            messagebox.showerror('Dataset settings', str(exc))

    def test_all(self):
        try:
            paths = {k: v.get() for k, v in self.paths.items()}
            n, per, seed, st = self.samples.get(), self.per_class.get(), self.seed.get(), self.settings()
            if n < 1 or per < 1:
                raise ValueError('Sample counts must be positive.')
            self.submit('all', lambda: dataset_check.check_all(paths, n, per, seed, st))
        except Exception as exc:
            messagebox.showerror('Dataset settings', str(exc))

    def open_dataset_video(self):
        try:
            if self.busy():
                return
            path = dataset_check.resolve_video(self.paths[self.dataset_name.get()].get(), self.video_choice.get())
            self.tabs.select(0)
            self.load_video(path)
        except Exception as exc:
            messagebox.showerror('Video', str(exc))

    def export_table(self):
        if self.table is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try:
                self.table.to_csv(path, index=False)
                self.status.set(f'Saved {path}')
            except Exception as exc:
                messagebox.showerror('Export', str(exc))

    def display_image(self, rgb):
        self.display_source = Image.fromarray(rgb)
        self.resize_display()

    def resize_display(self, event=None):
        if self.closed or self.display_source is None:
            return
        # A fixed viewport prevents the image's requested size from growing the window.
        width = max(1, self.video_viewport.winfo_width())
        height = max(1, self.video_viewport.winfo_height())
        image = self.display_source.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text='')

    def preview(self, delta=0):
        if self.gallery:
            self.gallery_index = (self.gallery_index + delta) % len(self.gallery)
            rgb, caption = self.gallery[self.gallery_index]
            image = Image.fromarray(rgb)
            image.thumbnail((600, 230))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text=caption, compound='top')

    def show_text(self, widget, text):
        widget.delete('1.0', 'end')
        widget.insert('end', text)

    def poll(self):
        if self.closed:
            return
        failed = self.future and self.future[1].done() and self.future[1].exception() is not None
        if failed:
            self.pending_frame = None
            while not self.frames.empty():
                try:
                    self.frames.get_nowait()
                except Empty:
                    break
        if not self.paused or self.stop_event.is_set():
            try:
                if self.pending_frame is None:
                    try:
                        self.pending_frame = self.frames.get_nowait()
                    except Empty:
                        pass
                packet = self.pending_frame
                if packet and (self.stop_event.is_set() or time.monotonic() - self.last_frame_time >= 1 / packet['fps']):
                    self.pending_frame = None
                    self.display_image(packet['image'])
                    self.last_frame_time = time.monotonic()
                    if packet['total']:
                        self.video_progress.set(min(100, 100 * packet['frame'] / packet['total']))
                    self.progress_text.set(f"{packet['frame'] / packet['fps']:.1f}s · Frame {packet['frame']}/{packet['total'] or '?'}")
                    self.status.set(f"Playing · Tracks: {packet['count']}" if self.future and self.future[0] == 'video' else 'Playing original video')
            except Empty:
                pass
        if self.future and self.future[1].done() and self.frames.empty() and self.pending_frame is None:
            kind, future = self.future
            self.future = None
            try:
                result = future.result()
                if kind == 'loadvideo':
                    self.video_path = result['path']
                    self.display_image(result['image'])
                    self.progress_text.set(f"Ready · {result['total'] / result['fps']:.1f}s · {result['fps']:.1f} FPS")
                elif kind == 'image':
                    image, dets, self.table = result
                    self.display_image(image)
                    self.show_text(self.results_text, f'Detections: {len(dets)}\n' + self.table.to_string(index=False))
                elif kind == 'video' and result:
                    self.table = result['tracks']
                    self.show_text(self.results_text, pipeline.summarize_tracks(self.table, result['n_frames'], result['fps'], self.calibration.get() > 0).replace('\\n', '\n'))
                elif kind == 'scan':
                    self.show_text(self.dataset_text, str(result))
                    self.video_list['values'] = result['videos']
                    self.video_choice.set(result['videos'][0] if result['videos'] else '')
                elif kind == 'dataset':
                    self.table = result['table']
                    self.gallery = result['gallery']
                    self.gallery_index = 0
                    self.show_text(self.dataset_text, result['summary'].replace('\\n', '\n') + '\n' + self.table.to_string(index=False) + ('\n' + result['per_class'].to_string(index=False) if result['per_class'] is not None else ''))
                    self.preview()
                elif kind == 'all':
                    self.table = result
                    self.show_text(self.dataset_text, result.to_string(index=False))
                self.status.set('Video ready. Choose Play original or Analyse video.' if kind == 'loadvideo' else ('Stopped.' if self.stop_event.is_set() else 'Complete.'))
                self.pause_button.configure(text='Pause')
            except Exception as exc:
                self.status.set(str(exc))
                self.progress_text.set('Could not complete — see the error message')
                messagebox.showerror('Processing error', dependency_error(exc))
        self.root.after(20, self.poll)

    def close(self):
        self.closed = True
        self.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    AquaVisionDesktop(root)
    root.mainloop()
