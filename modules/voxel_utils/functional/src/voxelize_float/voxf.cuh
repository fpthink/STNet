#ifndef _VOXF_CUH
#define _VOXF_CUH

// CUDA function declarations
void favg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out);
void favg_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const int *cnt, const float *grad_y, float *grad_x);

#endif
