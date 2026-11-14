"""Regression checks; fake inference isolates video lifecycle from model accuracy."""
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
from types import SimpleNamespace
import pipeline
import cv2
import numpy as np
import dataset_check


class EmptyDetector:
    def track(self, frame, **kwargs):
        return [SimpleNamespace(boxes=None)]


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / 'sample.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(5):
            writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
        writer.release()

    def test_missing_weights_never_call_yolo(self):
        with patch.object(pipeline, 'YOLO') as model:
            with self.assertRaises(FileNotFoundError):
                pipeline._load(str(self.root / 'missing.pt'), 'remote.pt')
            model.assert_not_called()

    def test_stream_and_frame_limit(self):
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()):
            stream = pipeline.iter_video(str(self.video), pipeline.Settings(max_frames=3))
            packets = [next(stream) for _ in range(3)]
            self.assertEqual(packets[-1]['frame'], 3)
            self.assertEqual(packets[0]['image'].shape, (60, 80, 3))
            with self.assertRaises(StopIteration) as end:
                next(stream)
            self.assertEqual(end.exception.value['n_frames'], 3)

    def test_stop_returns_partial_summary(self):
        stop = Event()
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()):
            stream = pipeline.iter_video(str(self.video), pipeline.Settings(), stop_event=stop)
            next(stream)
            stop.set()
            with self.assertRaises(StopIteration) as end:
                next(stream)
            self.assertEqual(end.exception.value['n_frames'], 1)

    def test_legacy_video_export(self):
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()), patch.object(pipeline, '_reencode_h264', side_effect=lambda p: p):
            result = pipeline.process_video(str(self.video), pipeline.Settings(max_frames=2))
            self.assertEqual(result['n_frames'], 2)
            self.assertTrue(Path(result['out_video']).is_file())

    def test_scan_and_path_boundary(self):
        summary = dataset_check.scan_summary(self.root, refresh=True)
        self.assertEqual(summary['n_videos'], 1)
        with self.assertRaises(ValueError):
            dataset_check.scan('')
        with self.assertRaises(ValueError):
            dataset_check.resolve_video(self.root, '../outside.avi')

    def test_check_all_runs_tests(self):
        cv2.imwrite(str(self.root / 'image.png'), np.zeros((60, 80, 3), np.uint8))
        table = pipeline.pd.DataFrame({'fish detected': [2], 'count match': [True]})
        with patch.object(dataset_check, 'sample_dataset', return_value={'table': table, 'summary': 'tested'}) as sample:
            result = dataset_check.check_all({n: str(self.root) for n in dataset_check.NAMES}, 1, 1, 42, pipeline.Settings())
            self.assertEqual(sample.call_count, 4)
            self.assertEqual(result.iloc[0]['Tested'], 1)
            self.assertEqual(result.iloc[0]['GT count match (%)'], 100)

    def test_nonempty_tracks_and_resource_cleanup(self):
        cap = cv2.VideoCapture(str(self.video))
        detection = dict(id=1, box=(10, 10, 30, 30), poly=None, conf=.9,
                         species='Unknown', sconf=0., length_px=20., rel=.2,
                         length_cm=None, size='Large', centre=(20., 20.))
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()), patch.object(pipeline.cv2, 'VideoCapture', return_value=cap), patch.object(pipeline, 'extract_detections', side_effect=lambda *a, **k: [detection.copy()]):
            result = pipeline.process_video(str(self.video), pipeline.Settings(), write_video=False)
            self.assertFalse(cap.isOpened())
            self.assertEqual(result['count'], 1)
            self.assertEqual(result['tracks'].iloc[0]['Frames seen'], 5)
        cap = cv2.VideoCapture(str(self.video))
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()), patch.object(pipeline.cv2, 'VideoCapture', return_value=cap):
            stream = pipeline.iter_video(str(self.video), pipeline.Settings())
            next(stream)
            stream.close()
            self.assertFalse(cap.isOpened())

    def test_video_preview_and_original_playback(self):
        preview = pipeline.video_preview(self.video)
        self.assertEqual(preview['total'], 5)
        self.assertEqual(preview['image'].shape, (60, 80, 3))
        packets = list(pipeline.iter_original_video(self.video))
        self.assertEqual(len(packets), 5)
        self.assertEqual(packets[-1]['frame'], 5)
        stop = Event()
        stream = pipeline.iter_original_video(self.video, stop)
        next(stream)
        stop.set()
        self.assertEqual(list(stream), [])

    def test_invalid_video(self):
        with self.assertRaises(ValueError):
            pipeline.video_preview(self.root / 'missing.mp4')

    def test_no_implicit_frame_limit(self):
        self.assertEqual(pipeline.Settings().max_frames, 0)
        real_cap = cv2.VideoCapture(str(self.video))
        class UnderreportedCapture:
            def get(self, prop):
                return 1 if prop == cv2.CAP_PROP_FRAME_COUNT else real_cap.get(prop)
            def __getattr__(self, name):
                return getattr(real_cap, name)
        with patch.object(pipeline, 'get_detector', return_value=EmptyDetector()), patch.object(pipeline.cv2, 'VideoCapture', return_value=UnderreportedCapture()):
            result = pipeline.process_video(str(self.video), pipeline.Settings(), write_video=False)
            self.assertEqual(result['n_frames'], 5)

    def test_species_normalization(self):
        self.assertEqual(dataset_check._norm('Fish_species-A'), 'fishspeciesa')


if __name__ == '__main__':
    unittest.main()
