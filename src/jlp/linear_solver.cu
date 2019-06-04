#include <tuple>
#include <vector>
#include <thrust/device_vector>

#include "jlp/linear_solver.cuh"
#include "jlp/status.h"

namespace compute_tools {
// This code snippet is from
// https://github.com/thrust/thrust/blob/master/Examples/strided_range.cu
// strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6]
// strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
// strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
// ...
// strided_range(first, last, stride)
template <typename Iterator>
class strided_range {
 public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor
      : public thrust::unary_function<difference_type, difference_type> {
    difference_type stride;

    stride_functor(difference_type stride) : stride(stride) {}

    __host__ __device__ difference_type
    operator()(const difference_type& i) const {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator>
      TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
      PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}

  iterator begin(void) const {
    return PermutationIterator(
        first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

 protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
    const std::vector<std::vector<float>>& A,
    const std::vector<float>& b, std::vector<float>& c,
    int num_iterations = 1000) {
  int m = host_A.size();
  DCHECK(m > 0) << "m == 0";
  int n = host_A.front().size();
  DCHECK(n > 0 && n >= m) << "m == 0 or n == 0 or n < m";
  DCHECK(n == host_c.size()) << "Incompatibility of size between A and c";
  DCHECK(m == host_b.size()) << "Incompatibility of size between A and b";
  thrust::host_vector<float> host_A(m * n);
  thrust::host_vector<float> host_b(m);
  thrust::host_vector<float> host_c(n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      host_A[i][j] = A[i][j];
  for (int i = 0; i < m; ++i)
    host_b[i] = b[i];
  for (int i = 0; i < n; ++i)
    host_c[i] = c[i];

  // In this implementation, try less data transfer between CPU and GPU
  thrust::device_vector<float> device_A = host_A;
  thrust::device_vector<float> device_b = host_b;
  thrust::device_vector<float> device_c = host_c;

  thrust::device_vector<float> device_inv_B(m * m);
  thrust::device_vector<int> device_basic_indices(m);
  // replace this with nonbasic columns of A
  // thrust::device_vector<int> device_nonbasic_indices(n - m);
  thrust::device_vector<float> device_nonbasic_A(m * (n - m));
  thrust::device_vector<float> device_nonbasic_c(n - m);
  thrust::device_vector<int> device_basic_coefficients(m);
  thrust::device_vector<int> device_basic_solutions(m);

  thrust::device_vector<float> device_simplex_multiplier(m);
  thrust::device_vector<float> device_exchange_reduction(m);
  thrust::device_vector<float> device_tmp_column(m);
  thrust::device_vector<float> device_eta(m);
  float objective_value = 0.0;

  for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
    thrust::fill(device_simplex_multiplier.begin(),
                 device_simplex_multiplier.end(), 0.0f);
    float alpha = 1.0f, beta = 0.0f;
    // Compute simplex multiplier
    cublasSgemv(handle, CUBLAS_OP_T, m, m, &alpha, device_inv_B.data(), m,
                device_basic_coefficients.data(), &beta,
                device_simplex_multiplier.data(), 1);

    alpha = -1.0f;
    beta = 1.0f;
    // Need to copy the origin device_nonbasic_c
    // device_nonbasic_c
    cublasSgemv(handle, CUBLAS_OP_T, m, n - m, &alpha, device_nonbasic_A.data(),
                n - m, device_simplex_multiplier.data(), &beta, device_nonbasic_c.data(), 1);
    int entering_index = -1;
    cublasIsamin(handle, n - m, device_nonbasic_c.data(), 1, &entering_index);

    // Optimal?
  }
}
} // namespace compute_tools