# dreifus

OpenCV, OpenGL, Pytorch3D, ... you never know which camera coordinate convention you are currently using?  
You always have to try `invert()` on your "pose" matrices because you never know whether they are cam2world or
world2cam?  
The dreifus library is what you need!   
dreifus (German for tripod) assists you in dealing with 3D cameras in Python.

## 1. Installation

```shell
pip install dreifus
```

## 2. Usage

### 2.1. Extrinsic (Pose) matrices
Translating between coordinate conventions made easy:

```python
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType

# wrap some 4x4 extrinsic matrix
# Default assumed coordinate convention: OPEN_CV
# Default assumed pose type: WORLD_2_CAM 
pose = Pose(some_extrinsic_matrix, pose_type=..., camera_coordinate_convention=...)

# Translate between coordinate conventions
pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)

# Ensure your pose transforms into the direction you expect
pose.change_pose_type(PoseType.CAM_2_WORLD)
```

### 2.2. Intrinsics

```python
from dreifus.matrix import Intrinsics

intrinsics = Intrinsics(fx, fy, cx, cy)

# Adapt your intrinsics to images downscaled by a factor of 2x
intrinsics.rescale(0.5)

# Adapt your intrinsics to an image cropped at (50, 50) left-top
intrinsics.crop(50, 50)
```

## 3. Visualization

The visualization tools will automatically interpret your camera poses correctly, as long as you specified `camera_coordinate_convention` and `pose_type` correctly.
```python

import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum

pose = Pose(...)  # Some extrinsics
intrinsics = Intrinsics(...)  # Some intrinsics
image = ...  # Some images taken from that view

p = pv.Plotter()

add_coordinate_axes(p)
add_camera_frustum(p, pose, intrinsics, image=image)

p.show()
```

Render a pyvista scene from a specific camera:
```python
import pyvista as pv
from dreifus.pyvista import render_from_camera

p = pv.Plotter(window_size=[IMG_W, IMG_H], off_screen=True)
p.background_color = (0, 0, 0, 0)

image = render_from_camera(p, pose, intrinsics)
```