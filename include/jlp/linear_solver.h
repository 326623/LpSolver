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
#ifndef _JLP_LINEAR_SOLVER_H_
#define _JLP_LINEAR_SOLVER_H_
#include <vector>
#include <tuple>

#include "status.h"

namespace compute_tools {
// There are plenty of space for optimization, like:
// 1. changing the memory layout of the matries to column major.
// 2. change the implementation to adapt sparse matrix-vector multiplication
// 3. auto vectorization.

// A is a dense matrix of size m x n. b is a vector of size m, c is a vector of
// size n.
std::tuple<ProblemStatus, std::vector<double>, std::vector<int>> Solve(
    const std::vector<std::vector<double>>& A, const std::vector<double>& b,
    std::vector<double>& c, int num_iterations = 1000);
}  // namespace compute_tools

#endif /* _JLP_LINEAR_SOLVER_H_ */
