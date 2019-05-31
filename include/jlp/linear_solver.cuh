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

// #define BLOCK_SIZE 16
// #define GRID_SIZE 16

namespace compute_tools {
std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
    const std::vector<std::vector<float>>& host_A,
    const std::vector<float>& host_b, std::vector<float>& host_c,
    int num_iterations = 1000);
// __global__ void InitializeVar(const float* device_b, const float* device_c,
//                               float* device_inv_B,
//                               float* device_basic_coefficients,
//                               int* device_basic_indices,
//                               int* device_nonbasic_indices,
//                               float* device_basic_solutions, int m, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   int offset = n - m;
//   for (int i = index; i < m; i += stride)
//     device_inv_B[index + index * m] = 1.0f;

//   for (int i = index; i < m; i += stride) {
//     device_basic_coefficients[i] = device_c[i + offset];
//     device_basic_indices[i] = i + offset;
//     device_basic_solutions[i] = device_b[i];
//   }

//   for (int i = index; i < n - m; i += stride)
//     device_nonbasic_indices[i] = i;
// }

// __global__ void ComputeSimplexMultiplier(float* device_simplex_multiplier,
//                                          float* device_basic_coefficients,
//                                          float* device_inv_B, int m) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = index; i < m; i += stride) {
//     device_simplex_multiplier[i] = 0.0f;
//     for (int j = 0; j < m; ++ j) {
//       device_simplex_multiplier[i] +=
//           device_basic_coefficients[j] * device_inv_B[j * m + i];
//     }
//   }
// }

// __global__ void PricingOut(const float* device_simplex_multipiler,
//                            const float* device_A, const float* device_c,
//                            const int* device_nonbasic_indices,
//                            // output variables
//                            int* device_entering_index,
//                            int* device_nonbasic_picked,
//                            float* device_best_entering_cost,
//                            // sizes
//                            int m, int n) {
//   // __shared__ float best_entering_costs[BLOCK_SIZE];
//   // __shared__ int best_entering_index[BLOCK_SIZE];
//   // __shared__ int best_nonbasic_picked[BLOCK_SIZE];

//   // int best_entering_cost_self;
//   // int best_entering_index_self;
//   // int best_nonbasic_picked
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = 0; i < n - m; i += stride) {
//     int index = device_nonbasic_indices[i];
//     float entering_cost = device_c[index];
//     for (int j = 0; j < m; ++j) {
//       entering_cost -= device_simplex_multipiler[j] * device_A[j][index];
//     }
//     if (entering_cost < best_entering_costs[])
//   }
// }

// // Takes in a linear program instance and perfrom Revised Simplex Method.
// // Return the basic solution and their indices. A is a dense matrix of size
// // m x n. b is a vector of size m, c is a vector of size n.
// // std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
// std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
//     const std::vector<std::vector<float>>& host_A,
//     const std::vector<float>& host_b, std::vector<float>& host_c,
//     int num_iterations = 1000) {
//   int m = host_A.size();
//   DCHECK(m > 0) << "m == 0";
//   int n = host_A.front().size();
//   DCHECK(n > 0 && n >= m) << "m == 0 or n == 0 or n < m";
//   DCHECK(n == host_c.size()) << "Incompatibility of size between A and c";
//   DCHECK(m == host_b.size()) << "Incompatibility of size between A and b";

//   // In this implementation, try less data transfer between CPU and GPU
//   float* device_A;
//   float* device_b;
//   float* device_c;
//   cudaMalloc(&device_A, (m * n) * sizeof(float));
//   cudaMalloc(&device_b, m * sizeof(float));
//   cudaMalloc(&device_c, n * sizeof(float));
//   cudaMemcpy(device_A, host_A.data(), (m * n) * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(device_b, host_b.data(), m * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(device_c, host_c.data(), n * sizeof(float), cudaMemcpyHostToDevice);

//   float* device_inv_B;
//   int* device_basic_indices;
//   int* device_nonbasic_indices;
//   float* device_basic_coefficients;
//   float* device_basic_solutions;
//   cudaMalloc(&device_inv_B, (m * m) * sizeof(float));
//   cudaMalloc(&device_basic_indices, m * sizeof(int));
//   cudaMalloc(&device_nonbasic_indices, (n - m) * sizeof(int));
//   cudaMalloc(&device_basic_coefficients, m * sizeof(float));
//   cudaMalloc(&device_basic_solutions, m * sizeof(float));

//   // Initialize these vectors with kernel
//   InitializeVar<<<64, 64>>>(device_b, device_c, device_inv_B,
//                             device_basic_coefficients, device_basic_indices,
//                             device_nonbasic_indices, device_basic_solutions, m,
//                             n);

//   // Intermediate variables to aid computation
//   float* device_simplex_multiplier;
//   float* device_exchange_reduction;
//   // For the update of inverse B
//   float* device_tmp_column;
//   float* device_eta;
//   // initial objective value
//   float* device_objective_value;
//   cudaMalloc(&device_simplex_multiplier, m * sizeof(float));
//   cudaMalloc(&device_exchange_reduction, m * sizeof(float));
//   cudaMalloc(&device_tmp_column, m * sizeof(float));
//   cudaMalloc(&device_eta, m * sizeof(float));
//   cudaMalloc(&device_objective_value, sizeof(float));

//   int* device_entering_index;
//   int* device_nonbasic_picked;
//   float* device_best_entering_cost;
//   cudaMalloc(&device_entering_index, sizeof(int));
//   cudaMalloc(&device_nonbasic_picked, sizeof(int));
//   cudaMalloc(&device_best_entering_cost, sizeof(float));

//   for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
//     ComputeSimplexMultiplier<<<64, 64>>>(
//         device_simplex_multiplier, device_basic_coefficients, device_inv_B, m);

//     // __device__ int entering_index = -1;
//     // __device__ int nonbasic_picked = -1;
//     // __device__ float best_entering_cost = 1.0;
//   }

//   // Freeing cuda memory
//   cudaFree(device_A);
//   cudaFree(device_b);
//   cudaFree(device_c);

//   cudaFree(device_inv_B);
//   cudaFree(device_basic_indices);
//   cudaFree(device_nonbasic_indices);
//   cudaFree(device_basic_coefficients);
//   cudaFree(device_basic_solutions);

//   cudaFree(device_simplex_multiplier);
//   cudaFree(device_exchange_reduction);
//   cudaFree(device_tmp_column);
//   cudaFree(device_eta);
//   cudaFree(device_objective_value);

//   cudaFree(device_entering_index);
//   cudaFree(device_nonbasic_picked);
//   cudaFree(device_best_entering_cost);

//   // return std::tuple<ProblemStatus, std::vector<float>, std::vector<int>>(
//   //     {ProblemStatus::INIT, std::vector<float>(1), std::vector<int>(1)});
//   std::vector<float> a(1);
//   std::vector<int> b(1);
//   return {INIT, {}, {}};
// }
} // namespace compute_tools

#endif /* _JLP_CUDA_LINEAR_SOLVER_HPP_ */
