#ifndef _NEWJOY_CULIBLP_HPP_
#define _NEWJOY_CULIBLP_HPP_

extern struct timespec ev_start, ev_end, lv_start, lv_end, b_start, b_end, alloc_start,
  alloc_end, dealloc_start, dealloc_end, init_start, init_end;
extern struct timespec blas_end;

// Not sure about this
#define BS 32
#define MAX_ITER 1000
#define EPS 1e-8
#define R2C(i,j,s) (((j)*(s))+(i))
float lpsolve(float* A, float* b, float* c, float* xb, int* bi, int m, int n);
int entering_index(float* e, int n);
void extract_column(float* M, float* v, int start_i, int stride, int size);
int leaving_index(float* t, int* flag, int size);
int compute_E(float* E, float* alpha, int m, int li);
int get_min_idx(float* a, int n, float* val);
__global__ void zeros(float* a, int m, int n);
__global__ void reduce_min(float* f, int n, float* min);
__global__ void get_val(float* f, int index, float* val);
__global__ void get_idx(float* f, int* index, float* val, int n);
__global__ void init_yb(float* yb);
__global__ void init_cInD(float* c, float* D, int m, int n);
__global__ void init_AInD(float* A, float* D, int m, int n);
__global__ void init_I(float* I, int m);
__global__ void init_bi(int* bi, int m, int n);
__global__ void compute_theta(float* xb, float* alpha, float* theta,
                              int* theta_flag, int m, int* num_max);
__global__ void compute_new_E(float* E, float* alpha, int m, int li, float qth);
__global__ void update_bi_cb(int* bi, float* cb, float* c, int li, int ei);

#endif /* _NEWJOY_CULIBLP_HPP_ */
