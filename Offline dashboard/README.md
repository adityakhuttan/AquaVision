# AquaVision — simple offline desktop

## Windows: start here

1. Extract the complete ZIP. Open the **AquaVision** folder inside it.
2. Install 64-bit Python 3.11 or 3.12, with Tkinter and the Python launcher.
3. Double-click **setup_windows.bat**. This installs or repairs dependencies in this folder's `.venv`. An internet connection is required for setup. It installs CPU PyTorch when PyTorch is absent.
4. When setup succeeds, double-click **run.bat**.
5. Click **Use sample**, then **Play original**. No detector is needed for playback.
6. For a processing demonstration, open **Models & settings**, click **Use included stock model (demo only)**, return Home, then click **Analyse video**.

Updating an existing installation: close the app, copy the new package contents into the existing AquaVision folder, replacing matching files while keeping `.venv`, datasets, and custom weights. Run **setup_windows.bat** once to repair dependencies, then **run.bat**.

After setup, image processing and video playback/analysis run offline. This ZIP contains source, a stock demo checkpoint, sample video, and setup scripts. Python packages must be installed; it is not a standalone Windows executable or an air-gapped installer.

## Home screen

- **Upload video…** opens a local file chooser. The file bar shows the selected path, and the first frame appears after decoding. This is local file loading; nothing is uploaded to a server.
- **Use sample** loads `samples/fish_test.mp4`.
- **Play original** plays the video without inference or model weights.
- **Analyse video** processes frames with the selected detector and ByteTrack, showing annotated playback.
- **Pause / Resume**, **Stop**, elapsed video time, frame count, and a progress bar control playback and analysis. Stop takes effect after the current inference call and retains available partial track statistics.
- **Save CSV** exports the current result table. A track needs at least three frames to appear in the video summary.
- **Open image / Analyse image** support static images.

CPU analysis can run slower than the original video's frame rate. Playback is silent. The first frame stays visible while a model loads. Videos with an unreadable codec produce an error; the included H.264 MP4 provides a known decoding test. The frame limit defaults to 0 (all frames). The complete selected file is used; only the preview is resized. Analysis reads until the end of the file unless you set a frame limit or click Stop.

## Fixing “No module named tap” or “No module named lap”

Run **setup_windows.bat** in the same folder as run.bat. It repairs both dependencies using the same Python environment as the launcher:

```bat
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python check_dependencies.py
```

`tap` is supplied by **typed-argument-parser**, not the unrelated package named tap. `lap` is a separate tracker dependency. Both are included in requirements.txt and checked before launch. The reported error was provided as text without its traceback; these fixes supply both missing modules, while other checkpoint-specific dependencies may need separate attention.

Tap package reference: https://github.com/swansonk14/typed-argument-parser

If a previous manual install put the unrelated `tap` package in this environment and the dependency check says “Wrong tap package”, run:

```bat
.venv\Scripts\python -m pip uninstall tap
.venv\Scripts\python -m pip install --force-reinstall typed-argument-parser==1.12.0
```

For Windows `c10.dll` / WinError 1114, install the Microsoft Visual C++ Redistributable and use CPU PyTorch. The application imports PyTorch before OpenCV and explicitly uses CPU inference.

## Models and accuracy

Your custom `weights/detector_seg.pt` and `weights/species_cls.pt` were not supplied. To use them, Browse to the team's Ultralytics detection/segmentation model and classification model in **Models & settings**, then Apply models. You can also place them in `weights/` with these exact names. No missing checkpoint is downloaded automatically.

The included `yolo11s-seg.pt` is a generic COCO segmentation model. Select it explicitly for a demonstration; the app displays a demo banner. **It may detect no fish in the sample.** Successful playback or analysis is not evidence of fish accuracy. Species remain Unknown without the custom classifier. Do not rename the stock model to masquerade as trained fish or species weights.

Calibrated centimetres require pixels-per-cm at the fish depth. Otherwise, sizes are relative estimates. Tracking IDs can fragment after occlusion, affecting unique counts.

## Datasets

Choose dataset folders in the **Datasets** tab. Scan folders, test sampled images, inspect previews, test all four datasets, or open a discovered video in Home. Empty data folders are placeholders; full datasets are not included.

Detection tests compare predicted counts with YOLO labels in a parallel labels folder or adjacent text files. They do not compute mAP or read COCO/AAU annotations. Fish4Knowledge tests require species subfolders whose names match the classifier classes. Seed and sample counts make sampling repeatable. Export CSV saves the latest result table.

## Sample video

`samples/fish_test.mp4` is a six-second H.264 MP4 (640 × 360, 15 FPS), derived from the fish-big footage in your original archive. `samples/README.txt` identifies the source clip. It is for playback and pipeline testing; no custom-model accuracy is claimed.

## Mac / verification

On this prepared Mac, use `run.command`. A Windows `.venv` must be created on Windows; environments are not portable between operating systems.

```sh
.venv/bin/python -m unittest -v test_offline
.venv/bin/python check_dependencies.py
.venv/bin/python test_desktop_smoke.py
```

The last command needs desktop access. See TEST_RESULTS.md for executed checks and limitations. API users can consume `pipeline.iter_video(...)`, close abandoned generators, or call `pipeline.process_video(..., write_video=True)` to write annotated MP4 and retrieve track and frame tables.

## Aspect ratios and small screens

Video stays inside a fixed preview area with its original aspect ratio; portrait, square, and landscape footage fit without cropping or stretching. Resizing the window redraws the preview. Every tab has vertical/horizontal scrollbars and mouse-wheel scrolling so controls remain accessible on small screens or at high Windows display scaling. Choosing a video references the complete original file, without trimming or uploading to a server. The first frame is only a preview; Play original and Analyse video read the full file by default.
