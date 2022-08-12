#include <pybind11/pybind11.h>

#include "voxelization/vox.hpp"
#include "voxelize_float/voxf.hpp"

PYBIND11_MODULE(_backend4, m) {
  m.def("avg_voxelize_forward", &avg_voxelize_forward,
        "Voxelization forward with average pooling (CUDA)");
  m.def("avg_voxelize_backward", &avg_voxelize_backward,
        "Voxelization backward (CUDA)");
  m.def("favg_voxelize_forward", &favg_voxelize_forward,
        "fVoxelization forward with average pooling (CUDA)");
  m.def("favg_voxelize_backward", &favg_voxelize_backward,
        "fVoxelization backward (CUDA)");

}
