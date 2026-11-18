#pragma once

#include <memory>
#include <string>
#include <vector>

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
    // many structures as one graph: the edges of atom i are first[i] to first[i+1] - 1, with r = x[j] - x[i] + shift;
    // energy[num_systems] and virial[num_systems][6] are sums over the atoms i of each system[i]
    virtual void compute_batch(int n, const double* x, const int* numbers, const int* first, const int* j,
                               const double* shift, int num_systems, const int* system, double* energy,
                               double* forces, double* virial, double* energies) = 0;
    virtual double r_cut() const = 0;
};

std::unique_ptr<Evaluator> mace_foundation_cpu(const std::string& file, const std::string& dtype, double skin);
std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin);

namespace mace_foundation {

struct NeighborList { std::vector<int> start, j, S; };
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);

}  // namespace mace_foundation
