/* Copyright (C) 2018 New Joy - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3
 *
 *
 * You should have received a copy of the GPLv3 license with
 * this file. If not, please visit https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: yangqp5@outlook.com (New Joy)
 *
 */
#ifndef _JLP_CUDA_LINEAR_SOLVER_HPP_
#define _JLP_CUDA_LINEAR_SOLVER_HPP_
#include "status.h"

namespace compute_tools {
__global__ void InitializeVar(float* device_inv_B, float* device_b,
                              float* device_c, float* device_basic_coefficients,
                              float* device_basic_indices, float* device_nonbasic_indices,
                              float* basic_solutions, int m, int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int offset = n - m;
  for (int i = index; i < m; i += stride)
    device_inv_B[index + index * m] = 1.0f;

  for (int i = index; i < m; i += stride) {
    device_basic_coefficients[i] = device_c[i + offset];
    device_basic_indices[i] = i + offset;
    device_basic_solutions[i] = device_b[i];
  }

  for (int i = index; i < n - m; i += stride)
    device_nonbasic_indices[i] = i;
}

__global__ void ComputeSimplexMultiplier(float* device_simplex_multiplier,
                                         float* device_basic_coefficients,
                                         float* device_inv_B, int m) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < m; i += stride) {
    simplex_multiplier[i] = 0.0f;
    for (int j = 0; j < m; ++ j) {
      simplex_multiplier[i] += basic_coefficients[j] * device_inv_B[j][i];
    }
  }
}

// Takes in a linear program instance and perfrom Revised Simplex Method.
// Return the basic solution and their indices. A is a dense matrix of size
// m x n. b is a vector of size m, c is a vector of size n.
std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
    const std::vector<std::vector<float>>& host_A,
    const std::vector<float>& host_b, std::vector<float>& host_c,
    int num_iterations = 1000) {
  int m = A.size();
  DCHECK(m > 0) << "m == 0";
  int n = A.front().size();
  DCHECK(n > 0 && n >= m) << "m == 0 or n == 0 or n < m";
  DCHECK(n == host_c.size()) << "Incompatibility of size between A and c";
  DCHECK(m == host_b.size()) << "Incompatibility of size between A and b";

  // In this implementation, try less data transfer between CPU and GPU
  float* device_A;
  float* device_b;
  float* device_c;
  cudaMalloc(&device_A, (m * n) * sizeof(float));
  cudaMalloc(&device_b, m * sizeof(float));
  cudaMalloc(&device_c, n * sizeof(float));
  cudaMemcpy(device_A, host_A.data(), (m * n) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_c, host_c.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  float* device_inv_B;
  int* device_basic_indices;
  int* device_nonbasic_indices;
  float* device_basic_coefficients;
  float* device_basic_solutions;
  cudaMalloc(&device_inv_B, (m * m) * sizeof(float));
  cudaMalloc(&device_basic_indices, m * sizeof(int));
  cudaMalloc(&device_nonbasic_indices, (n - m) * sizeof(int));
  cudaMalloc(&device_basic_coefficients, m * sizeof(float));
  cudaMalloc(&device_basic_solutions, m * sizeof(float));

  // Initialize these vectors with kernel
  InitializeVar(device_inv_B, device_b, device_c, device_basic_coefficients,
                device_basic_indices, device_nonbasic_indices, basic_solutions,
                m, n);

  // Intermediate variables to aid computation
  float* device_simplex_multiplier;
  float* device_exchange_reduction;
  // For the update of inverse B
  float* device_tmp_column;
  float* device_eta;
  // initial objective value
  float* device_objective_value;
  cudaMalloc(&device_simplex_multiplier, m * sizeof(float));
  cudaMalloc(&device_exchange_reduction, m * sizeof(float));
  cudaMalloc(&device_tmp_column, m * sizeof(float));
  cudaMalloc(&device_eta, m * sizeof(float));
  cudaMalloc(&device_objective_value, sizeof(float));

  for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
    ComputeSimplexMultiplier(device_simplex_multiplier,
                             device_basic_coefficients, device_inv_B, m);

    int entering_index = -1;
    int nonbasic_picked = -1;
    float best_entering_cost = 1.0;
  }

  // Freeing cuda memory
  cudaFree(device_A);
  cudaFree(device_b);
  cudaFree(device_c);

  cudaFree(device_inv_B);
  cudaFree(device_basic_indices);
  cudaFree(device_nonbasic_indices);
  cudaFree(device_basic_coefficients);
  cudaFree(device_basic_solutions);

  cudaFree(simplex_multiplier);
  cudaFree(exchange_reduction);
  cudaFree(tmp_column);
  cudaFree(eta);
  cudaFree(objective_value);
}
} // namespace compute_tools

#endif /* _JLP_CUDA_LINEAR_SOLVER_HPP_ */
