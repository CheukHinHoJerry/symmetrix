
#include "evaluator.hpp"
#include "mace_foundation_cuda.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mace_foundation;

namespace {

constexpr unsigned ERR_SHORT = 4;

struct Atoms { const double *x, *cell; const int *first, *j, *S; double* f; const double* shift; };

__device__ __forceinline__ void displacement(const Atoms& A, int i, int c, double& dx, double& dy, double& dz) {
    const int j = A.j[c];
    if (A.shift) {
        dx = A.x[3*j] - A.x[3*i] + A.shift[3*c];
        dy = A.x[3*j+1] - A.x[3*i+1] + A.shift[3*c+1];
        dz = A.x[3*j+2] - A.x[3*i+2] + A.shift[3*c+2];
        return;
    }
    const int* s = A.S + 3*c;
    dx = A.x[3*j] - A.x[3*i] + s[0]*A.cell[0] + s[1]*A.cell[3] + s[2]*A.cell[6];
    dy = A.x[3*j+1] - A.x[3*i+1] + s[0]*A.cell[1] + s[1]*A.cell[4] + s[2]*A.cell[7];
    dz = A.x[3*j+2] - A.x[3*i+2] + s[0]*A.cell[2] + s[1]*A.cell[5] + s[2]*A.cell[8];
}

template <class real> __global__ void k_count(Model<real> M, Graph<real> G, Atoms A) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* cnt = shared + (threadIdx.x >> 5)*M.T;
    if (n >= G.N) return;
    for (int b = lane; b < M.T; b += 32) cnt[b] = 0;
    __syncwarp();
    const int i = G.atom[n];
    for (int c = A.first[i] + lane; c < A.first[i+1]; c += 32) {
        double dx, dy, dz;
        displacement(A, i, c, dx, dy, dz);
        if (dx*dx + dy*dy + dz*dz < M.r_cut2) atomicAdd(&cnt[G.elem_all[A.j[c]]], 1);
    }
    __syncwarp();
    int total = 0;
    for (int b = lane; b < M.T; b += 32) {
        G.seg[n*(M.T + 1) + 1 + b] = cnt[b];
        total += cnt[b];
    }
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(~0u, total, o);
    if (lane == 0) G.count[n] = total;
}

template <class real> __global__ void k_fill(Model<real> M, Graph<real> G, Atoms A) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* off = shared + (threadIdx.x >> 5)*M.T;
    if (n >= G.N) return;
    const int T = M.T;
    const int a = G.elem[n];
    const int i = G.atom[n];
    int* seg = G.seg + n*(T + 1);
    if (lane == 0) {
        int o = G.start[n];
        for (int b = 0; b < T; ++b) {
            const int c = seg[b + 1];
            seg[b] = off[b] = o;
            o += c;
        }
        seg[T] = o;
    }
    __syncwarp();
    real s0 = 0;
    for (int jj0 = A.first[i]; jj0 < A.first[i+1]; jj0 += 32) {
        const int jj = jj0 + lane;
        int j = 0;
        int b = -1;
        double dx = 0;
        double dy = 0;
        double dz = 0;
        double r2 = 0;
        if (jj < A.first[i+1]) {
            j = A.j[jj];
            displacement(A, i, jj, dx, dy, dz);
            r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < M.r_cut2) b = G.elem_all[j];
        }
        const unsigned peers = match_any(b);
        if (b >= 0) {
            const int e = off[b] + __popc(peers & ((1u << lane) - 1));
            const double r = sqrt(r2);
            const double t = (r - M.r_min)*M.inv_h;
            const int knot = min(max((int)t, 0), M.K - 1);
            const int p = M.ordered ? a*T + b : pair_index(a, b, T);
            const real x = real(t - knot - 0.5);
            *reinterpret_cast<Vec4<real>*>(G.geometry + 4*e) = {real(dz/r), real(dx/r), real(dy/r), real(r)};
            G.neigh_indices[e] = j;
            G.row_pair[e] = make_int2(G.row_of_atom[j], p);
            if (r < M.r_min) atomicOr(G.err, ERR_SHORT);
            const long blk = ((long)p*M.K + knot)*4;
            if (M.A0_splines) s0 += cubic(M.A0_splines + blk, 1, 0, x);
        }
        __syncwarp();
        if (b >= 0 and lane == __ffs(peers) - 1) off[b] += __popc(peers);
        __syncwarp();
    }
    s0 = warp_sum(s0);
    if (lane == 0) G.s0[n] = 1 + s0;
}

template <class real> __global__ void k_forces(Model<real> M, Graph<real> G, Atoms A) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    const real q0 = M.A0_splines ? G.dot0[n] : 0;
    const real q1 = M.A1_splines ? G.dot1[n] : 0;
    double fx = 0;
    double fy = 0;
    double fz = 0;
    for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
        real* fe = G.edge_forces + 3*e;
        if (q0 != 0 or q1 != 0) {
            const Edge<real> q = edge(M, G, e);
            const real* blk = nullptr;
            real v, d0 = 0, d1 = 0;
            const long b = ((long)q.pair*M.K + q.knot)*4;
            if (M.A0_splines) {
                blk = M.A0_splines + b;
                cubic(blk, 1, 0, q.t, v, d0);
            }
            if (M.A1_splines) {
                blk = M.A1_splines + b;
                cubic(blk, 1, 0, q.t, v, d1);
            }
            const real s = (q0*d0 + q1*d1)*M.inv_h;
            fe[0] += s*q.y;
            fe[1] += s*q.z;
            fe[2] += s*q.x;
        }
        const int j = G.neigh_indices[e];
        atomicAdd(A.f + 3*j, (double)fe[0]); atomicAdd(A.f + 3*j + 1, (double)fe[1]); atomicAdd(A.f + 3*j + 2,
                (double)fe[2]);
        fx += fe[0];
        fy += fe[1];
        fz += fe[2];
    }
    fx = warp_sum(fx);
    fy = warp_sum(fy);
    fz = warp_sum(fz);
    if (lane == 0) {
        const int i = G.atom[n];
        atomicAdd(A.f + 3*i, -fx);
        atomicAdd(A.f + 3*i + 1, -fy);
        atomicAdd(A.f + 3*i + 2, -fz);
    }
}

template <class real> __global__ void k_system_outputs(Graph<real> G, const int* system, double* eatom, double* out) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    double v[7] = {};
    for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
        const real* g = G.geometry + 4*e;
        const double x = g[1]*g[3];
        const double y = g[2]*g[3];
        const double z = g[0]*g[3];
        const double fx = G.edge_forces[3*e];
        const double fy = G.edge_forces[3*e+1];
        const double fz = G.edge_forces[3*e+2];
        v[1] += x*fx;
        v[2] += y*fy;
        v[3] += z*fz;
        v[4] += 0.5*(x*fy + y*fx);
        v[5] += 0.5*(x*fz + z*fx);
        v[6] += 0.5*(y*fz + z*fy);
    }
    for (int c = 1; c < 7; ++c) v[c] = warp_sum(v[c]);
    if (lane == 0) {
        const int i = G.atom[n];
        v[0] = G.node_energies[n];
        if (eatom) eatom[i] = v[0];
        for (int c = 0; c < 7; ++c) atomicAdd(out + 7*system[i] + c, v[c]);
    }
}

template <class Precision, bool mixed>
struct MACEFoundationCUDA : Evaluator {
    MACEFoundationCUDA(const std::string& file, double skin);
    ~MACEFoundationCUDA() override;
    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double* forces, double* virial, double* energies) override;
    void compute_batch(int n, const double* x, const int* numbers, const int* first, const int* neigh,
                       const double* shift, int num_systems, const int* system, double* energy, double* forces,
                       double* virial, double* energies) override;
    double r_cut() const override { return mace.r_cut; }
    void set_atoms(int n, const int* numbers);
    template <class Sh> void launch(const Atoms&);
    template <class Sh> double compute_shape(int, const double*, const double*, double*, double*, double*);
    template <class Sh>
    void compute_node_energies_forces(const Model<Precision>&, const Graph<Precision>&, const Atoms&, bool);
    void compute_edges(const Model<Precision>&, const Graph<Precision>&, const Atoms&);
    void compute_forces(const Model<Precision>&, const Graph<Precision>&, const Atoms&);

    MACEFoundationCuda<Precision, mixed> mace;
    double skin;
    int natoms = -1;
    std::vector<double> x0, cell0;
    std::vector<int> numbers0;
    std::array<bool, 3> pbc0{};
    NeighborList nl;
    Buffer<double> xd, celld, fd, eatom;
    Buffer<int> nl_first, nl_j, nl_S;
    std::vector<int> numbers_of_nodes, batch_first, batch_j, batch_system;
    std::vector<double> batch_shift;
    Buffer<int> batch_first_d, batch_j_d, batch_system_d;
    Buffer<double> batch_shift_d, batch_out;
    cudaGraphExec_t graph = nullptr;
    std::vector<const void*> captured;
};

template <class Precision, bool mixed>
MACEFoundationCUDA<Precision, mixed>::MACEFoundationCUDA(const std::string& file, double skin) : mace(file,
        "MACEFoundation: "), skin(skin)
{
}

template <class Precision, bool mixed>
MACEFoundationCUDA<Precision, mixed>::~MACEFoundationCUDA()
{
    if (graph) cudaGraphExecDestroy(graph);
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::set_atoms(int n, const int* numbers)
{
    const int T = mace.num_elements;
    numbers_of_nodes.clear();
    std::vector<int> el(n);
    for (int i = 0; i < n; ++i) {
        const auto it = std::find(mace.atomic_numbers.begin(), mace.atomic_numbers.end(), numbers[i]);
        if (it == mace.atomic_numbers.end())
            throw std::runtime_error("MACEFoundation: atomic number " + std::to_string(numbers[i])
                                     + " is not an element of the model");
        el[i] = int(it - mace.atomic_numbers.begin());
    }
    std::vector<int> nodes;
    mace.segments.clear();
    for (int t = 0; t < T; ++t) {
        const int first = int(nodes.size());
        for (int i = 0; i < n; ++i) if (el[i] == t) nodes.push_back(i);
        if (int(nodes.size()) > first) mace.segments.push_back({first, int(nodes.size()), t});
    }
    mace.N = n;
    mace.NR = n;
    std::vector<int> node_elem(mace.N);
    std::vector<int> rows(n);
    for (int k = 0; k < mace.N; ++k) {
        node_elem[k] = el[nodes[k]];
        rows[nodes[k]] = k;
    }
    mace.atom_of_node.upload(nodes, mace.stream);
    mace.elem.upload(node_elem, mace.stream);
    mace.row_of_atom.upload(rows, mace.stream);
    mace.elem_all.upload(el, mace.stream);
    numbers_of_nodes.assign(numbers, numbers + n);
}

template <class Precision, bool mixed>
double MACEFoundationCUDA<Precision, mixed>::compute(int n, const double* x, const int* numbers,
                                                     const double* cell, const bool* pbc,
                                                 double* forces, double* virial, double* energies) {
    std::fill(virial, virial + 6, 0.0);
    if (n == 0) return 0;
    bool rebuild = n != natoms or not std::equal(numbers, numbers + n, numbers0.begin()) or not std::equal(cell,
            cell + 9, cell0.begin())
                   or not std::equal(pbc, pbc + 3, pbc0.begin());
    for (int i = 0; not rebuild and i < n; ++i) {
        const double dx = x[3*i] - x0[3*i];
        const double dy = x[3*i+1] - x0[3*i+1];
        const double dz = x[3*i+2] - x0[3*i+2];
        rebuild = 4*(dx*dx + dy*dy + dz*dz) > skin*skin;
    }
    if (rebuild) {
        natoms = -1;
        set_atoms(n, numbers);
        neighbor_list(n, x, cell, pbc, mace.r_cut + skin, nl);
        nl_first.upload(nl.start, mace.stream);
        nl_j.upload(nl.j, mace.stream);
        nl_S.upload(nl.S, mace.stream);
        xd.reserve(3*(size_t)n);
        fd.reserve(3*(size_t)n);
        eatom.reserve(n);
        celld.reserve(9);

        mace.reserve(std::max<long>(nl.j.size(), 1));
        CHECK(cudaStreamSynchronize(mace.stream));

        natoms = n;
        x0.assign(x, x + 3*(size_t)n);
        cell0.assign(cell, cell + 9);
        numbers0.assign(numbers, numbers + n);
        pbc0 = {pbc[0], pbc[1], pbc[2]};
    }
    double energy = 0;
    with_preset(mace.num_channels, mace.L_max, mace.num_channels_node, [&](auto s) {
        energy = this->template compute_shape<decltype(s)>(n, x, cell, forces, virial, energies);
    });
    return energy;
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::compute_batch(int n, const double* x, const int* numbers,
                                                         const int* first, const int* neigh, const double* shift,
                                                         int num_systems, const int* system, double* energy,
                                                         double* forces, double* virial, double* energies)
{
    std::fill(energy, energy + num_systems, 0.0);
    std::fill(virial, virial + 6*(size_t)num_systems, 0.0);
    std::fill(forces, forces + 3*(size_t)n, 0.0);
    if (n == 0) return;

    // nodes, edges (uploaded again only when they change; compute builds its neighbor list again after this)
    natoms = -1;
    const long E = first[n];
    auto same = [](const auto* p, size_t m, const auto& v) {
        return v.size() == m and std::equal(p, p + m, v.begin());
    };
    if (not same(numbers, n, numbers_of_nodes)) {
        batch_first.clear();
        set_atoms(n, numbers);
    }
    if (not same(first, n + 1, batch_first) or not same(neigh, E, batch_j) or not same(system, n, batch_system)) {
        batch_first.clear();
        std::vector<int> f(first, first + n + 1);
        std::vector<int> j(neigh, neigh + E);
        std::vector<int> s(system, system + n);
        batch_first_d.upload(f, mace.stream);
        batch_j_d.upload(j, mace.stream);
        batch_system_d.upload(s, mace.stream);
        xd.reserve(3*(size_t)n);
        fd.reserve(3*(size_t)n);
        eatom.reserve(n);
        mace.reserve(std::max<long>(E, 1));
        CHECK(cudaStreamSynchronize(mace.stream));
        batch_j = std::move(j);
        batch_system = std::move(s);
        batch_first = std::move(f);
    }
    if (not same(shift, 3*(size_t)E, batch_shift)) {
        batch_shift.clear();
        std::vector<double> v(shift, shift + 3*E);
        batch_shift_d.upload(v, mace.stream);
        batch_shift = std::move(v);
    }
    batch_out.reserve(7*(size_t)num_systems);

    CHECK(cudaMemcpyAsync(xd, x, 3*(size_t)n*sizeof(double), cudaMemcpyHostToDevice, mace.stream));
    const Atoms A{xd, nullptr, batch_first_d, batch_j_d, nullptr, fd, batch_shift_d};
    with_preset(mace.num_channels, mace.L_max, mace.num_channels_node, [&](auto s) {
        this->template launch<decltype(s)>(A);
    });

    // energies and virials of the systems
    const Graph<Precision> G = mace.graph();
    CHECK(cudaMemsetAsync(batch_out, 0, 7*(size_t)num_systems*sizeof(double), mace.stream));
    k_system_outputs<<<blocks(mace.N, WARPS), 32*WARPS, 0, mace.stream>>>(G, batch_system_d,
            energies ? eatom.p : nullptr, batch_out);
    std::vector<double> out(7*(size_t)num_systems);
    CHECK(cudaMemcpyAsync(out.data(), batch_out, out.size()*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    CHECK(cudaMemcpyAsync(forces, A.f, 3*(size_t)n*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    if (energies)
        CHECK(cudaMemcpyAsync(energies, eatom, (size_t)n*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    CHECK(cudaStreamSynchronize(mace.stream));
    if (*mace.err & ERR_SHORT) {
        *mace.err = 0;
        throw std::runtime_error("MACEFoundation: a pair is closer than the model's r_min");
    }
    for (int q = 0; q < num_systems; ++q) {
        energy[q] = out[7*q];
        std::copy(out.data() + 7*q + 1, out.data() + 7*q + 7, virial + 6*q);
    }
}

template <class Precision, bool mixed>
template <class Sh>
double MACEFoundationCUDA<Precision, mixed>::compute_shape(int n, const double* x, const double* cell, double* forces,
                                                           double* virial, double* energies)
{
    CHECK(cudaMemcpyAsync(xd, x, 3*(size_t)n*sizeof(double), cudaMemcpyHostToDevice, mace.stream));
    CHECK(cudaMemcpyAsync(celld, cell, 9*sizeof(double), cudaMemcpyHostToDevice, mace.stream));
    const Atoms A{xd, celld, nl_first, nl_j, nl_S, fd};
    launch<Sh>(A);

    const Graph<Precision> G = mace.graph();
    CHECK(cudaMemsetAsync(mace.outputs, 0, 7*sizeof(double), mace.stream));
    k_outputs<<<blocks(mace.N, WARPS), 32*WARPS, 0, mace.stream>>>(G, energies ? eatom.p : nullptr, mace.outputs);
    double out[7];
    CHECK(cudaMemcpyAsync(out, mace.outputs, sizeof(out), cudaMemcpyDeviceToHost, mace.stream));
    CHECK(cudaMemcpyAsync(forces, A.f, 3*(size_t)n*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    if (energies)
        CHECK(cudaMemcpyAsync(energies, eatom, (size_t)n*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    CHECK(cudaStreamSynchronize(mace.stream));
    if (*mace.err & ERR_SHORT) {
        *mace.err = 0;
        throw std::runtime_error("MACEFoundation: a pair is closer than the model's r_min");
    }
    std::copy(out + 1, out + 7, virial);
    return out[0];
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCUDA<Precision, mixed>::launch(const Atoms& A)
{
    const Model<Precision> M = mace.model();
    const Graph<Precision> G = mace.graph();
    std::vector<const void*> key = {(const void*)(long)mace.N, A.x, A.cell, A.first, A.j, A.S, A.f, A.shift,
        G.atom, G.elem, G.row_of_atom, G.count, G.start, G.seg, G.elem_all, G.geometry, G.neigh_indices, G.row_pair,
        G.edge_forces, G.Phi0, G.A0, G.M0, G.H1,
        G.Phi1, G.A1, G.H1_adj, G.A1_adj, G.dPhi1, G.M0_adj, G.A0_adj, G.Phi0_adj, G.s0, G.s1, G.dot0, G.dot1,
        G.node_energies, mace.scan_storage.p, G.X0, G.X1, G.Tg, G.T_adj, G.X_adj, G.B1, G.B1_adj, G.M0s_adj,
        G.M1, G.M1_adj};
    for (const auto& [b, e, t] : mace.segments)
        key.insert(key.end(), {(const void*)(long)b, (const void*)(long)e, (const void*)(long)t});
    const bool capture = key != captured;
    compute_node_energies_forces<Sh>(M, G, A, capture);
    if (capture) captured = key;
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCUDA<Precision, mixed>::compute_node_energies_forces(
    const Model<Precision>& M, const Graph<Precision>& G, const Atoms& A, bool capture)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    if (capture) {
        if (graph) CHECK(cudaGraphExecDestroy(graph));
        cudaGraph_t g;
        CHECK(cudaStreamBeginCapture(mace.stream, cudaStreamCaptureModeThreadLocal));
        compute_edges(M, G, A);
        mace.template compute_first_layer<Sh>(M, G);
        mace.template compute_second_layer<Sh>(M, G);
        mace.template compute_readouts<Sh>(M, G);
        mace.template reverse_second_layer<Sh>(M, G);
        mace.template reverse_first_layer<Sh>(M, G);
        compute_forces(M, G, A);
        CHECK(cudaStreamEndCapture(mace.stream, &g));
        CHECK(cudaGraphInstantiate(&graph, g, 0));
        CHECK(cudaGraphDestroy(g));
    }
    CHECK(cudaGraphLaunch(graph, mace.stream));
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::compute_edges(const Model<Precision>& M,
                                                         const Graph<Precision>& G, const Atoms& A)
{
    // neighbor list
    // A0 scaling
    // s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(0)}_{i,klm} = A^{(0)}_{i,klm} / s^{(0)}_i
    const size_t shared = WARPS*mace.num_elements*sizeof(int);
    k_count<<<blocks(mace.N, WARPS), 32*WARPS, shared, mace.stream>>>(M, G, A);
    size_t bytes = mace.scan_bytes;
    CHECK(cub::DeviceScan::ExclusiveSum(mace.scan_storage.p, bytes, G.count, G.start, mace.N + 1, mace.stream));
    k_fill<<<blocks(mace.N, WARPS), 32*WARPS, shared, mace.stream>>>(M, G, A);
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::compute_forces(const Model<Precision>& M,
                                                          const Graph<Precision>& G, const Atoms& A)
{
    // A1 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(1)}_i
    // A0 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(0)}_i
    // forces
    CHECK(cudaMemsetAsync(A.f, 0, 3*(size_t)mace.N*sizeof(double), mace.stream));
    k_forces<<<blocks(mace.N, WARPS), 32*WARPS, 0, mace.stream>>>(M, G, A);
}

}  // namespace

std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin)
{
    if (dtype == "float32-mixed") return std::make_unique<MACEFoundationCUDA<float, true>>(file, skin);
    if (dtype == "float32") return std::make_unique<MACEFoundationCUDA<float, false>>(file, skin);
    return std::make_unique<MACEFoundationCUDA<double, false>>(file, skin);
}
