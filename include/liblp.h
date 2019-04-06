#ifndef _NEWJOY_LIBLP_HPP_
#define _NEWJOY_LIBLP_HPP_

int entering_index(float* v, int size);
int leaving_index(float* t, int* flag, int size);
void compute_theta(float* x, float* a, float* t, int* flag, int size);
int compute_E(float* E, float* a, float* I, int size, int li);
void extract_column(float* M, float* v, int start_i, int stride, int size);
void create_identity_matrix(float** m, int size);

#endif /* _NEWJOY_LIBLP_HPP_ */
