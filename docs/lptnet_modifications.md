# LPTNet-related Modifications to OpenLPT

This document summarizes the source-code modifications made to the original OpenLPT codebase in this repository.

The main purpose of these modifications is to connect LPTNet-generated 2D particle detections with the original OpenLPT C++ stereo matching, Shake refinement, and STB tracking pipeline.

Compared with the original OpenLPT workflow, the modified version can directly import neural-network-predicted particle centers, diameters, and intensities from Python, preserve this information in `Tracer2D`, use it during Shake-based residual image computation, and optionally perform segmented / polyline-based stereo matching for nonlinear projection models.

The modifications are grouped by functionality rather than by file location.

---

## Modified Files Overview

| Category | Modified files | Main purpose |
|---|---|---|
| Configuration and data structures | `inc/libSTB/Config.h`, `src/pybind_OpenLPT/pyConfig.cpp`, `inc/libObject/ObjectInfo.h` | Add new parameters, update camera containers, and preserve neural-network-predicted tracer intensity |
| Object detection | `inc/libObject/ObjectFinder.h`, `src/srcObject/ObjectFinder.cpp`, `src/pybind_OpenLPT/pyObjectFinder.cpp` | Import LPTNet / neural-network 2D detections into OpenLPT as `Tracer2D` objects |
| Stereo matching | `inc/libSTB/StereoMatch.h`, `src/srcSTB/StereoMatch.cpp` | Add segmented / polyline-based stereo matching support |
| Shake / STB workflow | `src/srcSTB/Shake.cpp`, `src/srcSTB/STB.cpp`, `src/pybind_OpenLPT/pyShake.cpp` | Use neural-network intensity during Shake and expose a combined Python workflow |
| Matrix binding | `src/pybind_OpenLPT/pyMatrix.cpp` | Improve Python / NumPy data exchange |
| TIFF I/O compatibility | `inc/libtiff/tif_unix.c` | Improve Unix-like build compatibility |

---

## Configuration and Data Structure Changes

### Modified files

- `inc/libSTB/Config.h`
- `src/pybind_OpenLPT/pyConfig.cpp`
- `inc/libObject/ObjectInfo.h`

### Summary

This group of modifications updates the configuration system, Python bindings, and object data structures used by the modified LPTNet/OpenLPT workflow.

The main changes are:

1. The camera list representation was changed from `std::vector<std::shared_ptr<Camera>>` to `std::vector<Camera>`.
2. A new stereo-matching parameter, `SMParam::n_segments`, was added and exposed to Python.
3. `Tracer2D` was extended with an `_intensity` field to store the peak intensity predicted by the neural network.
4. Object projection and bubble-radius related interfaces were updated to use the new camera container.
5. `CreateArgs` was updated so that camera information can be passed into object-creation logic when needed.

### Detailed changes

#### Camera container update

In the original OpenLPT implementation, camera models were stored and passed as:

```cpp
std::vector<std::shared_ptr<Camera>>
```

In this modified version, the camera list is changed to:

```cpp
std::vector<Camera>
```

This change appears in configuration-related structures and object projection interfaces, including:

```cpp
Config::_cam_list
CreateArgs::_cams
Object3D::projectObject2D(...)
Object3D::isReconstructable(...)
Object3D::additional2DProjection(...)
Bubble::calRadiusFromCams(...)
```

This modification keeps the camera representation consistent across the modified C++ code and the Python binding layer.

#### New stereo-matching parameter

A new parameter was added to `SMParam`:

```cpp
int n_segments = 1;
```

This parameter is exposed to Python through `pyConfig.cpp`:

```cpp
.def_readwrite("n_segments", &SMParam::n_segments)
```

The default value is `1`, which keeps the conventional straight-line matching behavior. Larger values enable the segmented / polyline-based stereo matching mode introduced in this repository.

#### Neural-network intensity field

The `Tracer2D` class was extended with a new field:

```cpp
double _intensity = 0.0;
```

This field stores the peak intensity predicted by the neural network for each detected 2D tracer candidate.

The copy constructor of `Tracer2D` was also updated so that `_intensity` is preserved when `Tracer2D` objects are copied:

```cpp
Tracer2D(const Tracer2D &tracer)
    : Object2D(tracer),
      _r_px(tracer._r_px),
      _intensity(tracer._intensity) {}
```

This is important because 2D tracer objects may be copied or moved during object detection, stereo matching, and tracking. Without copying `_intensity`, the neural-network-derived information could be lost.

#### Object creation arguments

The `CreateArgs` structure was updated from:

```cpp
const std::vector<std::shared_ptr<Camera>> *_cam_list = nullptr;
```

to:

```cpp
const std::vector<Camera> *_cams = nullptr;
```

The corresponding Python binding was extended with:

```cpp
set_cams(...)
clear_cams(...)
```

This allows camera information to be attached to object-creation arguments from Python when needed, for example during bubble-radius computation or projection-related operations.

### Motivation

These changes provide the data-structure and configuration-level support required by the LPTNet-based workflow.

In the original OpenLPT pipeline, 2D tracer objects are mainly generated by OpenLPT's internal image-based object detection routines. In this modified workflow, tracer candidates can instead be produced by a neural network and then passed into the OpenLPT backend.

Therefore, the object representation needs to preserve additional neural-network output, such as the predicted peak intensity. At the same time, the stereo matching stage needs additional parameters, such as `n_segments`, to control the modified matching behavior.

### Impact

The modified configuration and object structures affect downstream modules including object detection, stereo matching, object creation, bubble-radius estimation, and the Shake/STB tracking pipeline.

### Notes

The change from `std::vector<std::shared_ptr<Camera>>` to `std::vector<Camera>` is a structural change that propagates through multiple modules. It should be kept consistent across all functions that access camera models.

The `CreateArgs::_cams` member stores a non-owning pointer to an external camera list. Therefore, the camera list must remain valid while `CreateArgs` is being used.

---

## Object Detection Modifications

### Modified files

- `inc/libObject/ObjectFinder.h`
- `src/srcObject/ObjectFinder.cpp`
- `src/pybind_OpenLPT/pyObjectFinder.cpp`

### Summary

The object-finding module was extended to support 2D tracer candidates generated outside the original OpenLPT image-based detection routine.

This modification is mainly used to import LPTNet / neural-network detection results into the OpenLPT C++ pipeline. Instead of relying only on OpenLPT's built-in object detection from images, the modified pipeline can directly construct `Tracer2D` objects from particle centers, diameters, and intensities predicted on the Python side.

### Main changes

A new C++ interface was added to `ObjectFinder2D`:

```cpp
std::vector<std::unique_ptr<Object2D>>
createTracersFromRawData(
    const double* centers_ptr,
    const double* diameters_ptr,
    const double* intensities_ptr,
    size_t num_particles
);
```

This function creates `Tracer2D` objects directly from raw array data.

For each detected particle candidate, it assigns:

```cpp
tr->_pt_center[0] = centers_ptr[2 * i];
tr->_pt_center[1] = centers_ptr[2 * i + 1];
tr->_r_px = diameters_ptr[i] / 2.0;
tr->_intensity = intensities_ptr[i];
```

The input convention is:

- `centers_ptr`: flattened 2D center coordinates in the order `[x0, y0, x1, y1, ...]`;
- `diameters_ptr`: predicted particle diameters in pixels;
- `intensities_ptr`: predicted peak intensities;
- `num_particles`: number of particle candidates.

Since OpenLPT stores the tracer size as radius, the predicted diameter is converted internally by:

```cpp
radius = diameter / 2.0
```

### Python binding

The Python binding was extended by including NumPy support:

```cpp
#include <pybind11/numpy.h>
```

A new Python-facing method was added:

```python
ObjectFinder2D.findTracer2D_fromNN(centers, diameters, intensities)
```

This method accepts three NumPy arrays:

- `centers`: an `N x 2` array containing particle center coordinates;
- `diameters`: a 1D array of length `N`;
- `intensities`: a 1D array of length `N`.

The binding checks the input dimensions before passing the raw pointers to the C++ backend. This prevents shape mismatches between the neural-network output and the OpenLPT object representation.

### Motivation

The original OpenLPT workflow detects 2D tracer particles from images using its built-in object-finding procedure. In the modified LPTNet/OpenLPT workflow, the 2D particle candidates can instead be produced by a neural network.

This modification provides a bridge between Python-side neural-network inference and the C++ OpenLPT tracking backend.

The modified data flow is:

```text
LPTNet output
    -> centers / diameters / intensities
    -> ObjectFinder2D.findTracer2D_fromNN(...)
    -> ObjectFinder2D::createTracersFromRawData(...)
    -> std::vector<std::unique_ptr<Object2D>>
    -> OpenLPT stereo matching / STB pipeline
```

### Impact

This change allows the modified pipeline to bypass or supplement the original image-based particle detection step. The generated `Tracer2D` objects can then be passed to downstream OpenLPT components such as stereo matching, 3D reconstruction, Shake refinement, and STB tracking.

The added `_intensity` field in `Tracer2D` ensures that neural-network-predicted intensity information is preserved after the Python detection results are converted into C++ objects.

### Notes

The Python method name `findTracer2D_fromNN` indicates that this interface is intended for neural-network-generated detections.

The function expects diameters rather than radii from Python. The conversion to radius is performed internally in C++ to match OpenLPT's `Tracer2D::_r_px` convention.

---

## Stereo Matching Modifications

### Modified files

- `inc/libSTB/StereoMatch.h`
- `src/srcSTB/StereoMatch.cpp`

### Summary

The stereo matching module was modified to support both the original straight-line epipolar search and a new segmented / polyline-based search mode.

In the original OpenLPT implementation, a 3D line-of-sight is projected onto another camera as a 2D line, and candidate particles are searched near this line. In the modified version, when `SMParam::n_segments > 1`, the 3D line-of-sight is sampled within the configured 3D search volume and projected as a 2D polyline. Candidate enumeration and back-projection validation are then performed using point-to-polyline distances.

### Main changes

The main changes include:

1. Updating the stereo matching module to use `std::vector<Camera>` instead of `std::vector<std::shared_ptr<Camera>>`.
2. Adding polyline-generation helpers for projected line-of-sight curves.
3. Extending candidate enumeration to switch between line-based and polyline-based matching according to `SMParam::n_segments`.
4. Extending IDMap row-span computation and point visitation to support projected polylines.
5. Updating back-projection checks to use point-to-polyline distances when segmented matching is enabled.
6. Updating triangulation and object creation to pass the modified camera list through `CreateArgs`.

### Segmented / polyline epipolar search

The modified stereo matching logic uses `SMParam::n_segments` to select the matching mode:

```text
n_segments <= 1:
    use the original straight-line epipolar search

n_segments > 1:
    use segmented / polyline-based epipolar search
```

When segmented matching is enabled, each 3D line-of-sight is sampled within the configured 3D search volume. The sampled 3D points are then projected to the target camera to form a 2D polyline.

This gives the following workflow:

```text
3D line-of-sight
    -> sample points within the 3D search volume
    -> project sampled points to the target camera
    -> construct a 2D polyline
    -> search candidate particles near the polyline
```

### IDMap support for polylines

The IDMap candidate-search logic was extended to support polyline constraints.

For each projected polyline, row spans are computed based on the tolerance band around its segments. For multiple line-of-sight constraints, the candidate regions are intersected so that a candidate point must satisfy all active geometric constraints.

The point visitation step was also extended to check point-to-polyline distances. For two-point polylines, a fast straight-line path is used. For multi-segment polylines, the minimum distance between the candidate point and all polyline segments is computed.

### Back-projection validation

The back-projection consistency check was updated to match the selected stereo matching mode.

When `n_segments <= 1`, the original point-to-line distance check is used.

When `n_segments > 1`, the check uses point-to-polyline distance. This keeps candidate enumeration and geometric validation consistent.

### Motivation

This modification is intended to address cases where the projection of a 3D line-of-sight onto an image plane cannot be accurately represented by a single straight 2D line.

Such cases may occur when using nonlinear camera models, strong image distortion, polynomial calibration models, or other imaging geometries where the projected epipolar trace is curved. The segmented / polyline mode approximates this curved trace by sampling the 3D line-of-sight and projecting the sampled points to the target camera.

### Impact

When `n_segments <= 1`, the stereo matching behavior remains close to the original straight-line mode.

When `n_segments > 1`, the stereo matching module uses polyline-based candidate search and validation. This can improve geometric consistency for nonlinear projection models, at the cost of additional computation.

This modification affects candidate enumeration, back-projection checking, IDMap search, triangulation preparation, and object creation.

---

## Shake, STB, and Residual-image Workflow Modifications

### Modified files

- `src/srcSTB/Shake.cpp`
- `src/srcSTB/STB.cpp`
- `src/pybind_OpenLPT/pyShake.cpp`

### Summary

The Shake and STB modules were modified to make the refinement and residual-image computation stages compatible with the LPTNet-based detection workflow.

The main changes include updating the camera container type, adding neural-network-aware tracer rendering during Shake, exporting additional intermediate tracking results, and adding a Python-facing macro function that runs stereo matching, shaking, filtering, and residual-image calculation in one C++ call.

### Main changes

1. The Shake module was updated to use `std::vector<Camera>` consistently with the modified configuration and stereo-matching modules.
2. `TracerShakeStrategy::project2DInt()` was modified to use `Tracer2D::_intensity` and `Tracer2D::_r_px` when neural-network-derived tracer information is available.
3. The original OTF-based tracer rendering path is preserved as a fallback when `_intensity <= 0`.
4. The STB initialization and convergence phases were modified to save additional intermediate track outputs.
5. The VSC skipping logic was changed from an unconditional skip to a calibration-state-dependent condition.
6. A new Python-facing function, `match_shake_and_calc_residual`, was added to run stereo matching, Shake refinement, ghost/repeated filtering, and residual image calculation in a single C++ call.

### Neural-network-aware tracer rendering

For tracer particles created from LPTNet outputs, the Shake module can use the neural-network-predicted peak intensity and particle radius directly when rendering the projected 2D particle image.

When `Tracer2D::_intensity > 0`, the Gaussian rendering parameters are constructed from:

- the predicted peak intensity;
- the predicted radius in pixels;
- a circular Gaussian assumption.

The corresponding rendering parameters are constructed using the neural-network-predicted intensity as the Gaussian amplitude and the predicted radius as the Gaussian width scale.

When `Tracer2D::_intensity <= 0`, the original OpenLPT OTF-based rendering model is used.

This keeps the modified Shake stage compatible with both neural-network-generated tracer detections and the original OpenLPT object representation.

### Python macro workflow

The new Python function:

```python
match_shake_and_calc_residual(cams, obj2d_lists, obj_cfg, current_image_list)
```

wraps the following C++ workflow:

```text
Python-side 2D detections
    -> clone Object2D objects into C++ containers
    -> StereoMatch::match()
    -> Shake::runShake()
    -> remove Ghost / Repeated objects
    -> calculate residual images
    -> return valid Object3D objects and residual images to Python
```

The function first clones Python-side `Object2D` objects into C++-owned `std::unique_ptr<Object2D>` containers. This avoids ownership and lifetime issues when passing objects from Python into the C++ stereo matching pipeline.

After stereo matching and Shake refinement, objects marked as `Ghost` or `Repeated` are removed before residual-image calculation and before returning the final 3D objects to Python.

This reduces Python/C++ data transfer overhead and provides a compact interface for the LPTNet/OpenLPT workflow.

### STB output changes

The STB workflow was modified to save additional intermediate tracking results.

In the initialization phase, initial tracks are saved to:

```text
InitialTrack/
```

In the convergence phase, tracks are saved to:

```text
ConvergeTrack/
```

This is useful for inspecting intermediate tracking results during initialization and convergence.

### VSC behavior change

The VSC skipping logic was changed from an unconditional skip to a calibration-state-dependent condition.

This means the VSC step is skipped only when the relevant camera calibration and OTF calibration conditions are already satisfied.

### Impact

These modifications allow neural-network-derived tracer candidates to be refined using OpenLPT's Shake procedure while preserving their predicted size and intensity information.

They also make it easier to call the core stereo matching and shaking pipeline from Python, which is important for integrating OpenLPT with a Python-based LPTNet inference workflow.

### Notes

The Python binding for `Shake` keeps internal references to the camera list and object configuration alive for the lifetime of the `Shake` instance. This is necessary because the C++ `Shake` object stores references to these objects.

---

## Matrix Binding Modifications

### Modified file

- `src/pybind_OpenLPT/pyMatrix.cpp`

### Summary

The Python binding for `Matrix<T>` was extended to support the Python buffer protocol.

### Main changes

The `Matrix<T>` pybind11 class binding was updated with:

```cpp
py::buffer_protocol()
```

A `def_buffer()` implementation was added so that the underlying matrix memory can be exposed as a two-dimensional Python buffer.

The buffer metadata includes:

- the raw data pointer;
- element size;
- Python format descriptor;
- matrix shape;
- row and column strides.

### Motivation

This modification makes it easier to exchange matrix-like data between the C++ OpenLPT backend and Python-side code.

In the LPTNet/OpenLPT workflow, Python needs to access image, matrix, and residual data produced by the C++ pipeline. Supporting the buffer protocol allows these objects to be viewed or converted more naturally as NumPy-compatible arrays.

### Impact

This improves Python/C++ interoperability and reduces the need for manual element-wise copying when matrix data is accessed from Python.

### Notes

This binding assumes that `Matrix<T>` stores its elements in contiguous row-major memory.

---

## TIFF I/O Compatibility Modification

### Modified file

- `inc/libtiff/tif_unix.c`

### Summary

A POSIX header was added to improve TIFF I/O compilation compatibility on Unix-like systems.

### Main change

The following header was added:

```c
#include <unistd.h>
```

### Motivation

Some Unix/POSIX file I/O functions used by `tif_unix.c` require declarations from `unistd.h`. Adding this include improves compatibility with compilers or build environments that require explicit function declarations.

### Impact

This change does not modify the TIFF I/O algorithm. It is a build-compatibility fix for Unix-like environments.

---

## Summary of the Modified Pipeline

The overall modified workflow can be summarized as:

```text
LPTNet / Python inference
    -> predicted centers, diameters, and intensities
    -> ObjectFinder2D.findTracer2D_fromNN(...)
    -> Tracer2D objects with radius and intensity
    -> StereoMatch with optional segmented / polyline epipolar search
    -> Shake refinement using neural-network-aware tracer rendering
    -> Ghost / repeated-object filtering
    -> residual image calculation
    -> Python-side output
```

In short, this modified version connects LPTNet-generated 2D detections with the original OpenLPT C++ tracking backend, while adding support for neural-network-derived tracer intensity, Python-side workflow integration, and optional polyline-based stereo matching.


### Practical observation

Although the modified Shake module can use neural-network-predicted particle diameter and intensity during residual image computation, our preliminary tests did not show a clear difference in the final 3D reconstruction results compared with the original OTF-based parameter estimation. Therefore, this modification is documented as a supported implementation option rather than as a validated accuracy improvement.