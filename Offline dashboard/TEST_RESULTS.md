# AquaVision simple GUI — verification, 13 November 2026

Executed on macOS ARM64 with Python 3.13, torch 2.12.0, ultralytics 8.4.212, OpenCV 5.0.0, lap 0.5.14, and typed-argument-parser 1.12.0.

- Eleven pipeline regression tests pass: local-only weight loading, frame limits, streaming, stop and partial summaries, MP4 export, nonempty track summaries, capture release, dataset checks, path boundaries, name normalization, preview, original playback, and invalid-video errors.
- Desktop integration check passed with native desktop access: sample file loading, first-frame preview, playback without models, percentage progress reaching 100%, pause/resume, replay, stop, real stock-model analysis, unreadable-video handling, and missing tap/lap repair messages.
- Real inference used the included stock YOLO11 segmentation model with ByteTrack on eight sample frames. This is a runtime check, not a fish-accuracy benchmark.
- Dependency check imports torch, torchvision, cv2, ultralytics, lap, tap, pandas, PIL, and tkinter successfully; verifies tap exposes the expected Tap class.
- Sample video is a decodable six-second, 90-frame, 640 × 360 H.264 MP4 derived from the supplied fish-big clip.
- Source compilation and ZIP integrity checked before delivery.

Windows setup/run scripts are supplied but have not been executed on a Windows host. No custom fish detector/species classifier or full datasets were supplied. Windows-specific codec/DLL behavior and custom model accuracy still need validation on the target computer. The ZIP includes source and sample assets; Python packages are installed by setup_windows.bat and are not bundled as a wheelhouse or executable.

## Aspect ratio / full-video update

- Explicit layout test passed with 1920x1080 landscape, 1080x1920 portrait, square, and ultrawide frames. Repeated rendering kept the window size fixed and each image inside the preview bounds.
- At a 600x420 window, the home panel scrolled to its bottom.
- Existing native desktop playback/analysis integration check passed after the layout change.
- Frame-limit regression verifies that full-video mode reads to EOF even when metadata underreports frame count. Default max_frames is now 0 (all frames).
- All 545 frames of the attached WhatsApp recording were decoded using the original-playback iterator.
