from abc import abstractmethod
from threading import Thread
from typing import Dict

import dearpygui.dearpygui as dpg
import numpy as np


class VisualizationWindow:

    def __init__(self, title: str = 'Visualization Window', width: int = 800, height: int = 600):
        """
        Creates a new visualization window that runs in the background.
        Subclasses should overwrite the _populate() method to add dearpygui components to the window
        NB: the super().__init__() call should be AT THE END of the subclass constructor because the superclass __init__() will already start the worker thread.
        To close the window, call close()
        """

        self._title = title
        self._width = width
        self._height = height

        self._stop = False

        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    @abstractmethod
    def _populate(self):
        pass

    def _worker(self):
        dpg.create_context()

        self._populate()

        dpg.create_viewport(title=self._title, width=self._width, height=self._height)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self._stop:
                break
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def close(self):
        self._stop = True


class ImageWindow(VisualizationWindow):

    def __init__(self, image_buffer: np.ndarray, image_name: str = 'Image', title: str = 'Visualization Window'):
        """
        Creates a visualization window that shows a numpy array.
        The size of the window is inferred from the image buffer.
        To update the image, just change th image buffer in place.

        Parameters
        ----------
            image_buffer: [H, W, 3] numpy array (float32) that holds the image data
            image_name: Image name to display
            title: Title of the window
        """

        assert image_buffer.dtype == np.float32

        self._image_buffer = image_buffer
        self._image_name = image_name

        super(ImageWindow, self).__init__(title=title, width=image_buffer.shape[1], height=image_buffer.shape[0])

    def _populate(self):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self._image_buffer.shape[1], height=self._image_buffer.shape[0], default_value=self._image_buffer,
                                format=dpg.mvFormat_Float_rgb,
                                tag="image_buffer")

        with dpg.window(label=self._image_name, no_scrollbar=True, no_resize=True, no_close=True, no_collapse=True, no_move=True):
            dpg.add_image("image_buffer")

class MultiImageWindow(VisualizationWindow):

    def __init__(self, image_buffer_dict: Dict[str, np.ndarray], title: str = 'Visualization Window'):
        """
        Creates a visualization window that shows a numpy array.
        The size of the window is inferred from the image buffer.
        To update the image, just change th image buffer in place.

        Parameters
        ----------
            image_buffer_dict: [H, W, 3] numpy arrays (float32) that holds the image data. Keys will be used as image name
            title: Title of the window
        """

        for image_buffer in image_buffer_dict.values():
            assert image_buffer.dtype == np.float32

        self._image_buffer_dict = image_buffer_dict

        sum_width = sum([image_buffer.shape[1] for image_buffer in image_buffer_dict.values()])
        max_height = max([image_buffer.shape[0] for image_buffer in image_buffer_dict.values()])

        super(MultiImageWindow, self).__init__(title=title, width=sum_width, height=max_height)

    def _populate(self):
        for image_name, image_buffer in self._image_buffer_dict.items():
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(width=image_buffer.shape[1], height=image_buffer.shape[0], default_value=image_buffer,
                                    format=dpg.mvFormat_Float_rgb,
                                    tag=image_name)

            with dpg.window(label=image_name, no_scrollbar=True, no_resize=True, no_close=True, no_collapse=True, no_move=True):
                dpg.add_image(image_name)