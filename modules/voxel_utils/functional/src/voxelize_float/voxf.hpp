#ifndef _VOXF_HPP
#define _VOXF_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> favg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int x,
                                             const int y,
                                             const int z);

at::Tensor favg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt);

#endif
