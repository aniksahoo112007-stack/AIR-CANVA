import tempfile, time, unittest
from pathlib import Path
import numpy as np

from air_canvas.annotation_renderer import AnnotationRenderer
from air_canvas.desktop_mapper import DesktopMapper
from air_canvas.monitor_manager import MonitorBounds
from air_canvas.pointer_renderer import PointerRenderer

class DesktopAnnotationTests(unittest.TestCase):
    def test_mapper_clamps_and_persists_per_camera_monitor(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json'; mapper=DesktopMapper(path); monitor=MonitorBounds(1,100,50,1000,500)
            mapper.set_context(2,1); mapper.calibrate([(0.1,0.2),(0.9,0.2),(0.9,0.8),(0.1,0.8)])
            self.assertEqual(mapper.camera_to_desktop((0.9,0.2),monitor),(100,50))
            again=DesktopMapper(path);again.set_context(2,1);self.assertEqual(again.camera_to_desktop((0.1,0.8),monitor),(1099,549))

    def test_vector_history_groups_shapes_and_clear_is_undoable(self):
        renderer=AnnotationRenderer(320,200)
        renderer.begin('rectangle',(10,10),(0,0,255));renderer.append((100,80));self.assertTrue(renderer.finish())
        renderer.begin('pen',(20,100),(255,0,0));renderer.append((120,100));renderer.finish()
        self.assertEqual(len(renderer.actions),2);renderer.undo();self.assertEqual(len(renderer.actions),1);renderer.redo();self.assertEqual(len(renderer.actions),2)
        renderer.clear();self.assertEqual(len(renderer.actions),0);renderer.undo();self.assertEqual(len(renderer.actions),2)
        self.assertGreater(np.count_nonzero(renderer.render()[:,:,3]),0)

    def test_laser_expires_without_modifying_annotation(self):
        layer=np.zeros((200,320,4),np.uint8);pointer=PointerRenderer();now=time.monotonic();pointer.update((50,60),now);pointer.render(layer,(0,0),now)
        self.assertGreater(np.count_nonzero(layer[:,:,3]),0);pointer.update(None,now+1);fresh=np.zeros_like(layer);pointer.render(fresh,(0,0),now+1);self.assertEqual(np.count_nonzero(fresh),0)

if __name__=='__main__':unittest.main()
