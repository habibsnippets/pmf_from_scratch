// pmf_cpu.cpp
// PMF CPU baseline with percentile-per-cell and auto-tune dh0
// Build: same CMake as before (O3, -fopenmp)

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include <queue>
#include <deque>
#include <iostream>
#include <cmath>
#include <cstring>
#include <sstream>
#include <fstream>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

using PointT = pcl::PointXYZ;
using Clock = std::chrono::high_resolution_clock;

struct Grid {
    int rows, cols;
    double xmin, xmax, ymin, ymax;
    double cell;
    std::vector<float> data; // row-major
    Grid(int r=0, int c=0, double cell_=1.0): rows(r), cols(c), xmin(0), xmax(0), ymin(0), ymax(0), cell(cell_) {
        data.assign(rows*cols, std::numeric_limits<float>::infinity());
    }
    inline float& at(int r, int c) { return data[r*cols + c]; }
    inline const float& at(int r, int c) const { return data[r*cols + c]; }
};

inline int clampi(int v, int a, int b){ return v < a ? a : (v > b ? b : v); }

// rasterize collecting per-cell z-values (for percentile)
void rasterize_collect(const pcl::PointCloud<PointT>::Ptr &pc, Grid &G, std::vector<int> &ix, std::vector<int> &iy,
                       std::vector<std::vector<float>> &cellvals) {
    int n = pc->size();
    ix.resize(n); iy.resize(n);
    cellvals.assign(G.rows * G.cols, std::vector<float>());
    double xmin = G.xmin, cell = G.cell;
    int cols = G.cols, rows = G.rows;
    for (int i=0;i<n;i++){
        const auto &p = pc->points[i];
        int c = int(std::floor((p.x - xmin) / cell));
        int r = int(std::floor((p.y - G.ymin) / cell));
        c = clampi(c, 0, cols-1);
        r = clampi(r, 0, rows-1);
        ix[i] = c; iy[i] = r;
        cellvals[r*cols + c].push_back(p.z);
    }
}

// fill grid with p-th percentile (p in [0,100])
void fill_grid_percentile(Grid &G, const std::vector<std::vector<float>> &cellvals, int p_percentile=5) {
    int R = G.rows, C = G.cols;
    for (int r=0;r<R;++r){
        for (int c=0;c<C;++c){
            auto const &vals = cellvals[r*C + c];
            if (!vals.empty()) {
                if (vals.size() == 1) {
                    G.at(r,c) = vals[0];
                } else {
                    // compute percentile
                    std::vector<float> tmp = vals;
                    std::nth_element(tmp.begin(), tmp.begin() + (tmp.size()*p_percentile)/100, tmp.end());
                    float val = tmp[(tmp.size()*p_percentile)/100];
                    G.at(r,c) = val;
                }
            } else {
                G.at(r,c) = std::numeric_limits<float>::infinity();
            }
        }
    }
}

// nearest neighbor fill using multi-source BFS (Manhattan)
void fill_empty_cells_bfs(Grid &G) {
    int R = G.rows, C = G.cols;
    std::queue<std::pair<int,int>> q;
    std::vector<char> visited(R*C, 0);
    for (int r=0;r<R;r++) for (int c=0;c<C;c++) {
        if (!std::isinf(G.at(r,c))) {
            q.emplace(r,c);
            visited[r*C + c] = 1;
        }
    }
    if (q.empty()) return;
    const int dr[4] = {-1,1,0,0};
    const int dc[4] = {0,0,-1,1};
    while(!q.empty()){
        auto pr = q.front(); q.pop();
        int r=pr.first, c=pr.second;
        float val = G.at(r,c);
        for (int k=0;k<4;k++){
            int rn = r + dr[k], cn = c + dc[k];
            if (rn<0 || rn>=R || cn<0 || cn>=C) continue;
            int idx = rn*C + cn;
            if (!visited[idx]) {
                G.at(rn,cn) = val;
                visited[idx] = 1;
                q.emplace(rn,cn);
            }
        }
    }
}

// sliding window 1D min and max (deque)
void sliding_min_1d(const float *in, float *out, int n, int w) {
    if (w <= 1) { for (int i=0;i<n;i++) out[i]=in[i]; return; }
    std::deque<int> dq;
    int half = (w-1)/2;
    for (int i=0;i<n;i++){
        while(!dq.empty() && in[dq.back()] >= in[i]) dq.pop_back();
        dq.push_back(i);
        int left = i - w + 1;
        while(!dq.empty() && dq.front() < left) dq.pop_front();
        int out_idx = i - half;
        if (out_idx >= 0 && out_idx < n && !dq.empty()) out[out_idx] = in[dq.front()];
    }
    // fill boundary heads/tails if any (fallback exact compute)
    for (int i=0;i< std::min(half, n); ++i) {
        int L = 0, R = std::min(n-1, i + half);
        float mn = std::numeric_limits<float>::infinity();
        for (int j=L;j<=R;j++) mn = std::min(mn, in[j]);
        out[i] = mn;
    }
    for (int i = std::max(0, n-half); i < n; ++i) {
        int L = std::max(0, i-half), R = std::min(n-1, i+half);
        float mn = std::numeric_limits<float>::infinity();
        for (int j=L;j<=R;j++) mn = std::min(mn, in[j]);
        out[i] = mn;
    }
}
void sliding_max_1d(const float *in, float *out, int n, int w) {
    if (w <= 1) { for (int i=0;i<n;i++) out[i]=in[i]; return; }
    std::deque<int> dq;
    int half = (w-1)/2;
    for (int i=0;i<n;i++){
        while(!dq.empty() && in[dq.back()] <= in[i]) dq.pop_back();
        dq.push_back(i);
        int left = i - w + 1;
        while(!dq.empty() && dq.front() < left) dq.pop_front();
        int out_idx = i - half;
        if (out_idx >= 0 && out_idx < n && !dq.empty()) out[out_idx] = in[dq.front()];
    }
    for (int i=0;i< std::min(half, n); ++i) {
        int L = 0, R = std::min(n-1, i + half);
        float mx = -std::numeric_limits<float>::infinity();
        for (int j=L;j<=R;j++) mx = std::max(mx, in[j]);
        out[i] = mx;
    }
    for (int i = std::max(0, n-half); i < n; ++i) {
        int L = std::max(0, i-half), R = std::min(n-1, i+half);
        float mx = -std::numeric_limits<float>::infinity();
        for (int j=L;j<=R;j++) mx = std::max(mx, in[j]);
        out[i] = mx;
    }
}

// separable erosion/dilation
void separable_erosion(const Grid &G_in, Grid &G_out, int w) {
    int R = G_in.rows, C = G_in.cols;
    Grid tmp(R,C,G_in.cell);
    #pragma omp parallel for schedule(static)
    for (int r=0;r<R;++r) {
        sliding_min_1d(&G_in.data[r*C], &tmp.data[r*C], C, w);
    }
    #pragma omp parallel for schedule(static)
    for (int c=0;c<C;++c) {
        std::vector<float> col_in(R), col_out(R);
        for (int r=0;r<R;++r) col_in[r] = tmp.at(r,c);
        sliding_min_1d(col_in.data(), col_out.data(), R, w);
        for (int r=0;r<R;++r) G_out.at(r,c) = col_out[r];
    }
}
void separable_dilation(const Grid &G_in, Grid &G_out, int w) {
    int R = G_in.rows, C = G_in.cols;
    Grid tmp(R,C,G_in.cell);
    #pragma omp parallel for schedule(static)
    for (int r=0;r<R;++r) {
        sliding_max_1d(&G_in.data[r*C], &tmp.data[r*C], C, w);
    }
    #pragma omp parallel for schedule(static)
    for (int c=0;c<C;++c) {
        std::vector<float> col_in(R), col_out(R);
        for (int r=0;r<R;++r) col_in[r] = tmp.at(r,c);
        sliding_max_1d(col_in.data(), col_out.data(), R, w);
        for (int r=0;r<R;++r) G_out.at(r,c) = col_out[r];
    }
}

// compute initial diff for given w (used for auto-tune)
std::vector<float> compute_diff_values(const Grid &G, int w) {
    Grid er(G.rows, G.cols, G.cell), op(G.rows, G.cols, G.cell);
    separable_erosion(G, er, w);
    separable_dilation(er, op, w);
    std::vector<float> diffs; diffs.reserve(G.rows * G.cols);
    for (int r=0;r<G.rows;++r) for (int c=0;c<G.cols;++c) {
        float d = G.at(r,c) - op.at(r,c);
        if (d > 1e-8f) diffs.push_back(d);
    }
    return diffs;
}

// write csv (grid and diff) for debug
void dump_grid_csv(const Grid &G, const std::string &fname) {
    std::ofstream fo(fname);
    fo << "r,c,z\n";
    for (int r=0;r<G.rows;++r) for (int c=0;c<G.cols;++c) fo << r << "," << c << "," << G.at(r,c) << "\n";
    fo.close();
    std::cerr << "Wrote grid CSV to " << fname << "\n";
}

void pmf(const pcl::PointCloud<PointT>::Ptr &incloud,
         pcl::PointCloud<PointT>::Ptr &ground_out,
         pcl::PointCloud<PointT>::Ptr &nonground_out,
         double cell_sz = 0.5,
         double slope = 0.05,
         double dh0 = 0.0,
         double dhmax = 3.0,
         int wmax = 21,
         int percentile_cell = 5,
         bool auto_tune = true,
         const std::string &dump_csv = "")
{
    // bbox
    double xmin=1e300, ymin=xmin, xmax=-xmin, ymax=-xmin;
    for (const auto &p : incloud->points) {
        xmin = std::min(xmin, (double)p.x);
        xmax = std::max(xmax, (double)p.x);
        ymin = std::min(ymin, (double)p.y);
        ymax = std::max(ymax, (double)p.y);
    }
    int cols = int(std::floor((xmax - xmin) / cell_sz)) + 1;
    int rows = int(std::floor((ymax - ymin) / cell_sz)) + 1;
    Grid G(rows, cols, cell_sz);
    G.xmin = xmin; G.xmax = xmax; G.ymin = ymin; G.ymax = ymax;

    std::vector<int> ix, iy;
    std::vector<std::vector<float>> cellvals;
    rasterize_collect(incloud, G, ix, iy, cellvals);
    fill_grid_percentile(G, cellvals, percentile_cell);
    fill_empty_cells_bfs(G);

    if (!dump_csv.empty()) dump_grid_csv(G, dump_csv);

    if (wmax < 3) wmax = 3;
    if (wmax % 2 == 0) wmax += 1;
    std::vector<int> windows;
    for (int w=3; w<=wmax; w+=2) windows.push_back(w);

    // auto-tune dh0 if requested and dh0 <= 0
    if (auto_tune && dh0 <= 0.0) {
        auto diffs = compute_diff_values(G, 3);
        if (!diffs.empty()) {
            std::sort(diffs.begin(), diffs.end());
            int idx = std::max(0, int(diffs.size()*25/100));
            double guess = diffs[idx];
            dh0 = std::max(0.005, guess); // floor
            std::cerr << "[auto-tune] dh0 set to " << dh0 << " (25th percentile of initial diffs)\n";
        } else {
            dh0 = 0.01;
            std::cerr << "[auto-tune] no diffs found; dh0 fallback to " << dh0 << "\n";
        }
    } else if (dh0 <= 0.0) {
        dh0 = 0.01; // default safety
    }

    std::vector<char> ground_mask(G.rows * G.cols, 0);

    std::cout << "Grid: rows=" << G.rows << " cols=" << G.cols << " cells=" << (G.rows*G.cols) << "\n";
    std::cout << "Points: " << incloud->size() << "\n";

    for (size_t wi=0; wi<windows.size(); ++wi) {
        int w = windows[wi];
        Grid er(G.rows, G.cols, cell_sz), op(G.rows, G.cols, cell_sz);
        separable_erosion(G, er, w);
        separable_dilation(er, op, w);
        double dh_s = dh0 + slope * ((w - 1)/2) * cell_sz;
        if (dh_s > dhmax) dh_s = dhmax;
        int marked_new = 0;
        float diff_min = std::numeric_limits<float>::infinity(), diff_max = -std::numeric_limits<float>::infinity();
        double diff_sum = 0; int diff_cnt = 0;
        for (int r=0;r<G.rows;++r) for (int c=0;c<G.cols;++c) {
            float d = G.at(r,c) - op.at(r,c);
            diff_min = std::min(diff_min, d);
            diff_max = std::max(diff_max, d);
            diff_sum += d; diff_cnt++;
            if (d < dh_s) {
                int idx = r*G.cols + c;
                if (!ground_mask[idx]) { ground_mask[idx] = 1; marked_new++; }
            }
        }
        double diff_mean = diff_sum / std::max(1, diff_cnt);
        std::cout << "w=" << w << " dh_s=" << dh_s << " diff=[" << diff_min << "," << diff_max << "] mean=" << diff_mean << " newly_marked=" << marked_new << "\n";
    }

    ground_out->clear(); nonground_out->clear();
    ground_out->reserve(incloud->size()); nonground_out->reserve(incloud->size());
    for (size_t i=0;i<incloud->points.size();++i) {
        const auto &p = incloud->points[i];
        int c=ix[i], r=iy[i]; int idx = r*G.cols + c;
        if (ground_mask[idx]) ground_out->push_back(p);
        else nonground_out->push_back(p);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pmf_cpu <input.pcd> [cell_sz] [slope] [dh0] [dhmax] [wmax] [percentile_cell] [dump_csv]\n";
        std::cerr << "  pass dh0 <= 0 to enable auto-tune (uses 25th percentile of initial diffs)\n";
        return -1;
    }
    std::string infile = argv[1];
    double cell = 0.5, slope = 0.05, dh0 = 0.0, dhmax = 3.0;
    int wmax = 21;
    int percentile_cell = 5;
    std::string dump_csv = "";
    if (argc >= 3) cell = atof(argv[2]);
    if (argc >= 4) slope = atof(argv[3]);
    if (argc >= 5) dh0 = atof(argv[4]);
    if (argc >= 6) dhmax = atof(argv[5]);
    if (argc >= 7) wmax = atoi(argv[6]);
    if (argc >= 8) percentile_cell = atoi(argv[7]);
    if (argc >= 9) dump_csv = std::string(argv[8]);

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(infile, *cloud) < 0) {
        std::cerr << "Failed to read " << infile << "\n"; return -1;
    }
    std::cout << "Loaded cloud with " << cloud->size() << " points\n";

    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr nonground(new pcl::PointCloud<PointT>());

    auto t0 = Clock::now();
    pmf(cloud, ground, nonground, cell, slope, dh0, dhmax, wmax, percentile_cell, /*auto_tune=*/true, dump_csv);
    auto t1 = Clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "PMF runtime: " << secs << " s\n";
    std::cout << "Ground points: " << ground->size() << " Non-ground: " << nonground->size() << "\n";

    try {
        if (ground->size() > 0) {
            if (pcl::io::savePCDFileBinary("pmf_ground.pcd", *ground) == 0) std::cout << "Wrote pmf_ground.pcd (" << ground->size() << " points)\n";
            else std::cerr << "Warning: failed to write pmf_ground.pcd\n";
        } else std::cout << "Ground cloud empty; skipping write\n";

        if (nonground->size() > 0) {
            if (pcl::io::savePCDFileBinary("pmf_nonground.pcd", *nonground) == 0) std::cout << "Wrote pmf_nonground.pcd (" << nonground->size() << " points)\n";
            else std::cerr << "Warning: failed to write pmf_nonground.pcd\n";
        } else std::cout << "Non-ground cloud empty; skipping write\n";
    } catch (const pcl::IOException &e) {
        std::cerr << "PCL I/O exception: " << e.what() << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception writing PCDs: " << e.what() << "\n";
    }
    return 0;
}
