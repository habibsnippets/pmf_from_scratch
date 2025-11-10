// pmf_full.cpp
// Full PMF baseline with exponential windows, per-scale dh_s, percentile-per-cell,
// OpenMP threading, timing at start and end, safe writes, merged colored PCD output.
//
// Build with CMake (see instructions below).
//
// Usage (example):
//   ./pmf_full input.pcd [options]
// or use defaults that match your requested parameter set.
//
// Author: assistant (example)

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
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using PointT = pcl::PointXYZ;
using Clock = std::chrono::high_resolution_clock;

// ---------------- Grid container ----------------
struct Grid {
    int rows, cols;
    double xmin, xmax, ymin, ymax;
    double cell;
    std::vector<float> data; // row-major
    Grid(int r=0, int c=0, double cell_=1.0) : rows(r), cols(c), xmin(0), xmax(0), ymin(0), ymax(0), cell(cell_) {
        data.assign(rows * cols, std::numeric_limits<float>::infinity());
    }
    inline float & at(int r, int c) { return data[r * cols + c]; }
    inline const float & at(int r, int c) const { return data[r * cols + c]; }
};

// ---------------- helpers ----------------
inline int clampi(int v, int a, int b) { return (v < a) ? a : ((v > b) ? b : v); }

// Rasterize collecting per-cell z-values
void rasterize_collect(const pcl::PointCloud<PointT>::Ptr &pc, Grid &G, std::vector<int> &ix, std::vector<int> &iy,
                       std::vector<std::vector<float>> &cellvals) {
    int n = (int)pc->size();
    ix.resize(n); iy.resize(n);
    cellvals.assign(G.rows * G.cols, std::vector<float>());
    double xmin = G.xmin, ymin = G.ymin, cell = G.cell;
    int cols = G.cols, rows = G.rows;
    for (int i = 0; i < n; ++i) {
        const auto &p = pc->points[i];
        int c = int(std::floor((p.x - xmin) / cell));
        int r = int(std::floor((p.y - ymin) / cell));
        c = clampi(c, 0, cols - 1);
        r = clampi(r, 0, rows - 1);
        ix[i] = c; iy[i] = r;
        cellvals[r * cols + c].push_back(p.z);
    }
}

// Fill grid using percentile per cell
void fill_grid_percentile(Grid &G, const std::vector<std::vector<float>> &cellvals, int p_percentile = 5) {
    int R = G.rows, C = G.cols;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            auto const &vals = cellvals[r * C + c];
            if (!vals.empty()) {
                if (vals.size() == 1) {
                    G.at(r, c) = vals[0];
                } else {
                    // select p-th percentile via nth_element
                    std::vector<float> tmp = vals;
                    int idx = std::max(0, (int)((tmp.size() * p_percentile) / 100));
                    if (idx >= (int)tmp.size()) idx = (int)tmp.size() - 1;
                    std::nth_element(tmp.begin(), tmp.begin() + idx, tmp.end());
                    G.at(r, c) = tmp[idx];
                }
            } else {
                G.at(r, c) = std::numeric_limits<float>::infinity();
            }
        }
    }
}

// Multi-source BFS nearest-fill (Manhattan)
void fill_empty_cells_bfs(Grid &G) {
    int R = G.rows, C = G.cols;
    std::queue<std::pair<int,int>> q;
    std::vector<char> visited(R * C, 0);
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) {
        if (!std::isinf(G.at(r, c))) {
            q.emplace(r, c);
            visited[r * C + c] = 1;
        }
    }
    if (q.empty()) return;
    const int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
    while (!q.empty()) {
        auto pr = q.front(); q.pop();
        int r = pr.first, c = pr.second;
        float val = G.at(r, c);
        for (int k = 0; k < 4; ++k) {
            int rn = r + dr[k], cn = c + dc[k];
            if (rn < 0 || rn >= R || cn < 0 || cn >= C) continue;
            int idx = rn * C + cn;
            if (!visited[idx]) {
                G.at(rn, cn) = val;
                visited[idx] = 1;
                q.emplace(rn, cn);
            }
        }
    }
}

// 1D sliding min/max (deque) with output centered on same index
void sliding_min_1d(const float *in, float *out, int n, int w) {
    if (w <= 1) { for (int i = 0; i < n; ++i) out[i] = in[i]; return; }
    std::deque<int> dq;
    int half = (w - 1) / 2;
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && in[dq.back()] >= in[i]) dq.pop_back();
        dq.push_back(i);
        int left = i - w + 1;
        while (!dq.empty() && dq.front() < left) dq.pop_front();
        int out_idx = i - half;
        if (out_idx >= 0 && out_idx < n && !dq.empty()) out[out_idx] = in[dq.front()];
    }
    // boundary fallback
    for (int i = 0; i < std::min(half, n); ++i) {
        int L = 0, R = std::min(n-1, i + half);
        float mn = std::numeric_limits<float>::infinity();
        for (int j = L; j <= R; ++j) mn = std::min(mn, in[j]);
        out[i] = mn;
    }
    for (int i = std::max(0, n - half); i < n; ++i) {
        int L = std::max(0, i - half), R = std::min(n-1, i + half);
        float mn = std::numeric_limits<float>::infinity();
        for (int j = L; j <= R; ++j) mn = std::min(mn, in[j]);
        out[i] = mn;
    }
}
void sliding_max_1d(const float *in, float *out, int n, int w) {
    if (w <= 1) { for (int i = 0; i < n; ++i) out[i] = in[i]; return; }
    std::deque<int> dq;
    int half = (w - 1) / 2;
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && in[dq.back()] <= in[i]) dq.pop_back();
        dq.push_back(i);
        int left = i - w + 1;
        while (!dq.empty() && dq.front() < left) dq.pop_front();
        int out_idx = i - half;
        if (out_idx >= 0 && out_idx < n && !dq.empty()) out[out_idx] = in[dq.front()];
    }
    // boundary fallback
    for (int i = 0; i < std::min(half, n); ++i) {
        int L = 0, R = std::min(n-1, i + half);
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = L; j <= R; ++j) mx = std::max(mx, in[j]);
        out[i] = mx;
    }
    for (int i = std::max(0, n - half); i < n; ++i) {
        int L = std::max(0, i - half), R = std::min(n-1, i + half);
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = L; j <= R; ++j) mx = std::max(mx, in[j]);
        out[i] = mx;
    }
}

// separable erosion/dilation
void separable_erosion(const Grid &Gin, Grid &Gout, int w) {
    int R = Gin.rows, C = Gin.cols;
    Grid tmp(R, C, Gin.cell);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < R; ++r) {
        sliding_min_1d(&Gin.data[r*C], &tmp.data[r*C], C, w);
    }
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        std::vector<float> col_in(R), col_out(R);
        for (int r = 0; r < R; ++r) col_in[r] = tmp.at(r, c);
        sliding_min_1d(col_in.data(), col_out.data(), R, w);
        for (int r = 0; r < R; ++r) Gout.at(r, c) = col_out[r];
    }
}

void separable_dilation(const Grid &Gin, Grid &Gout, int w) {
    int R = Gin.rows, C = Gin.cols;
    Grid tmp(R, C, Gin.cell);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < R; ++r) {
        sliding_max_1d(&Gin.data[r*C], &tmp.data[r*C], C, w);
    }
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        std::vector<float> col_in(R), col_out(R);
        for (int r = 0; r < R; ++r) col_in[r] = tmp.at(r, c);
        sliding_max_1d(col_in.data(), col_out.data(), R, w);
        for (int r = 0; r < R; ++r) Gout.at(r, c) = col_out[r];
    }
}

// Build windows vector (odd sizes) using exponential or linear growth of radius
std::vector<int> build_windows(double max_window_size_m, double cell_size, int base_radius_cells, bool exponential_growth) {
    std::vector<int> windows;
    if (base_radius_cells < 1) base_radius_cells = 1;
    long long r = base_radius_cells;
    while (true) {
        double radius_m = r * cell_size;
        if (radius_m > max_window_size_m) break;
        int w = int(2 * r + 1);
        windows.push_back(w);
        if (exponential_growth) r = r * 2;
        else r = r + base_radius_cells;
        if (r > 1e6) break;
    }
    return windows;
}

// compute per-scale dh_s
double compute_dh_s(double initial_distance, double height_threshold_scale, int scale_index,
                    double slope, int radius_cells, double cell_size, double max_distance) {
    double scaled_initial = initial_distance * std::pow(height_threshold_scale, (double)scale_index);
    double slope_component = slope * (radius_cells * cell_size);
    double dh_s = scaled_initial + slope_component;
    if (dh_s > max_distance) dh_s = max_distance;
    return dh_s;
}

// compute diffs for initial scale if needed (used for optional auto-tune)
std::vector<float> compute_diffs_for_w(const Grid &G, int w) {
    Grid er(G.rows, G.cols, G.cell), op(G.rows, G.cols, G.cell);
    separable_erosion(G, er, w);
    separable_dilation(er, op, w);
    std::vector<float> diffs; diffs.reserve(G.rows * G.cols);
    for (int r = 0; r < G.rows; ++r) for (int c = 0; c < G.cols; ++c) {
        float d = G.at(r, c) - op.at(r, c);
        if (d > 1e-8f) diffs.push_back(d);
    }
    return diffs;
}

// dump grid CSV for debugging
void dump_grid_csv(const Grid &G, const std::string &fname) {
    std::ofstream fo(fname);
    fo << "r,c,z\n";
    for (int r = 0; r < G.rows; ++r) for (int c = 0; c < G.cols; ++c) fo << r << "," << c << "," << G.at(r, c) << "\n";
    fo.close();
    std::cerr << "Wrote grid CSV to " << fname << "\n";
}


// ---------------- PMF pipeline ----------------
void pmf_pipeline(const pcl::PointCloud<PointT>::Ptr &incloud,
                  pcl::PointCloud<PointT>::Ptr &ground_out,
                  pcl::PointCloud<PointT>::Ptr &nonground_out,
                  // parameters
                  double max_window_size_m,
                  double slope,
                  double initial_distance,
                  double max_distance,
                  double cell_size,
                  int base_radius_cells,
                  bool exponential_growth,
                  double height_threshold_scale,
                  int percentile_cell,
                  int num_threads,
                  const std::string &dump_csv)
{
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    // bounding box
    double xmin = std::numeric_limits<double>::infinity(), ymin = xmin, xmax = -xmin, ymax = -xmin;
    for (const auto &p : incloud->points) {
        xmin = std::min(xmin, (double)p.x);
        xmax = std::max(xmax, (double)p.x);
        ymin = std::min(ymin, (double)p.y);
        ymax = std::max(ymax, (double)p.y);
    }
    int cols = int(std::floor((xmax - xmin) / cell_size)) + 1;
    int rows = int(std::floor((ymax - ymin) / cell_size)) + 1;
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Invalid grid size (rows/cols <= 0). Check cell_size and input cloud extents.\n";
        return;
    }
    Grid G(rows, cols, cell_size);
    G.xmin = xmin; G.xmax = xmax; G.ymin = ymin; G.ymax = ymax;

    // rasterize collect
    std::vector<int> ix, iy;
    std::vector<std::vector<float>> cellvals;
    rasterize_collect(incloud, G, ix, iy, cellvals);

    // fill grid using percentile
    fill_grid_percentile(G, cellvals, percentile_cell);
    // fill empties
    fill_empty_cells_bfs(G);

    if (!dump_csv.empty()) dump_grid_csv(G, dump_csv);

    // build windows
    std::vector<int> windows = build_windows(max_window_size_m, cell_size, base_radius_cells, exponential_growth);
    if (windows.empty()) {
        std::cerr << "No windows generated; check max_window_size_m and base_radius_cells\n";
        return;
    }
    std::cout << "Windows:";
    for (size_t i = 0; i < windows.size(); ++i) {
        int w = windows[i];
        int r = (w - 1) / 2;
        std::cout << " w=" << w << "(r=" << r << " cells, " << std::fixed << std::setprecision(3) << (r * cell_size) << "m)";
    }
    std::cout << "\n";

    // progressive filtering
    std::vector<char> ground_mask(G.rows * G.cols, 0);
    std::cout << "Grid: rows=" << G.rows << " cols=" << G.cols << " cells=" << (G.rows * G.cols) << "\n";
    std::cout << "Points: " << incloud->size() << "\n";

    for (size_t k = 0; k < windows.size(); ++k) {
        int w = windows[k];
        int radius_cells = (w - 1) / 2;
        Grid er(G.rows, G.cols, cell_size), op(G.rows, G.cols, cell_size);
        separable_erosion(G, er, w);
        separable_dilation(er, op, w);
        double dh_s = compute_dh_s(initial_distance, height_threshold_scale, (int)k, slope, radius_cells, cell_size, max_distance);

        int newly_marked = 0;
        float diff_min = std::numeric_limits<float>::infinity(), diff_max = -std::numeric_limits<float>::infinity();
        double diff_sum = 0; int cnt = 0;
        for (int r = 0; r < G.rows; ++r) for (int c = 0; c < G.cols; ++c) {
            float d = G.at(r, c) - op.at(r, c);
            diff_min = std::min(diff_min, d);
            diff_max = std::max(diff_max, d);
            diff_sum += d; ++cnt;
            if (d < dh_s) {
                int idx = r * G.cols + c;
                if (!ground_mask[idx]) { ground_mask[idx] = 1; ++newly_marked; }
            }
        }
        double diff_mean = diff_sum / std::max(1, cnt);
        std::cout << "w=" << w << " dh_s=" << dh_s << " diff=[" << diff_min << "," << diff_max << "] mean=" << diff_mean << " newly_marked=" << newly_marked << "\n";
    }

    // map back to points
    ground_out->clear(); nonground_out->clear();
    ground_out->reserve(incloud->size()); nonground_out->reserve(incloud->size());
    for (size_t i = 0; i < incloud->points.size(); ++i) {
        const auto &p = incloud->points[i];
        int c = ix[i], r = iy[i];
        int idx = r * G.cols + c;
        if (ground_mask[idx]) ground_out->push_back(p);
        else nonground_out->push_back(p);
    }
}

// ---------------- main ----------------
int main(int argc, char **argv) {
    std::cout << "PMF full - starting\n";
    auto program_start = Clock::now();
    std::time_t ts = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Start time: " << std::ctime(&ts);

    if (argc < 2) {
        std::cerr << "Usage: pmf_full input.pcd [options]\n";
        std::cerr << "This program defaults to the parameter set you provided earlier.\n";
    }

    // Default parameters (your requested set)
    std::string infile = (argc >= 2) ? std::string(argv[1]) : std::string("input.pcd");
    double max_window_size_m = 10.0;
    double slope = 0.5;
    double initial_distance = 0.1;
    double max_distance = 0.5;
    double cell_size = 0.2;
    int base_radius_cells = 2;
    bool exponential_growth = true;
    double height_threshold_scale = 1.2;
    int num_threads = 4;
    int percentile_cell = 5;
    std::string dump_csv = "";

    // optional CLI parsing (simple)
    // usage: pmf_full input.pcd max_window slope initial_distance max_distance cell_size base_radius exp_growth(0/1) height_scale num_threads percentile dump_csv
    if (argc >= 12) {
        max_window_size_m = atof(argv[2]);
        slope = atof(argv[3]);
        initial_distance = atof(argv[4]);
        max_distance = atof(argv[5]);
        cell_size = atof(argv[6]);
        base_radius_cells = atoi(argv[7]);
        exponential_growth = (atoi(argv[8]) != 0);
        height_threshold_scale = atof(argv[9]);
        num_threads = atoi(argv[10]);
        percentile_cell = atoi(argv[11]);
        if (argc >= 13) dump_csv = std::string(argv[12]);
    } else {
        std::cout << "Using default parameters. To override pass: max_window slope initial_distance max_distance cell_size base_radius exp_growth(0/1) height_scale num_threads percentile [dump_csv]\n";
        std::cout << "Default values: max_window=" << max_window_size_m << " slope=" << slope << " initial_distance=" << initial_distance
                  << " max_distance=" << max_distance << " cell_size=" << cell_size << " base_radius=" << base_radius_cells
                  << " exp=" << exponential_growth << " height_scale=" << height_threshold_scale << " threads=" << num_threads
                  << " percentile=" << percentile_cell << "\n";
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(infile, *cloud) < 0) {
        std::cerr << "Failed to read input PCD: " << infile << "\n";
        return -1;
    }
    std::cout << "Loaded cloud: " << cloud->size() << " points\n";

    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr nonground(new pcl::PointCloud<PointT>());

    pmf_pipeline(cloud, ground, nonground,
                 max_window_size_m, slope, initial_distance, max_distance, cell_size,
                 base_radius_cells, exponential_growth, height_threshold_scale,
                 percentile_cell, num_threads, dump_csv);

    auto program_end = Clock::now();
    std::chrono::duration<double> elapsed = program_end - program_start;
    std::time_t ts_end = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "End time: " << std::ctime(&ts_end);
    std::cout << "Elapsed: " << elapsed.count() << " s\n";

    // safe writes: write files into current working dir with absolute info
    try {
        char cwd_buf[4096];
        if (getcwd(cwd_buf, sizeof(cwd_buf)) == nullptr) strncpy(cwd_buf, ".", sizeof(cwd_buf));
        std::string cwd(cwd_buf);
        if (ground->size() > 0) {
            std::string gname = cwd + "/pmf_ground.pcd";
            if (pcl::io::savePCDFileBinary(gname, *ground) == 0) std::cout << "Wrote " << gname << " (" << ground->size() << ")\n";
            else std::cerr << "Failed to write ground binary PCD\n";
        } else std::cout << "Ground empty; skipping write\n";

        if (nonground->size() > 0) {
            std::string ngname = cwd + "/pmf_nonground.pcd";
            if (pcl::io::savePCDFileBinary(ngname, *nonground) == 0) std::cout << "Wrote " << ngname << " (" << nonground->size() << ")\n";
            else std::cerr << "Failed to write nonground binary PCD\n";
        } else std::cout << "Non-ground empty; skipping write\n";

        if (ground->size() > 0 || nonground->size() > 0) {
            pcl::PointCloud<pcl::PointXYZRGB> merged;
            merged.reserve(ground->size() + nonground->size());
            for (const auto &p : ground->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z; q.r = 255; q.g = 50; q.b = 50;
                merged.push_back(q);
            }
            for (const auto &p : nonground->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z; q.r = 50; q.g = 150; q.b = 255;
                merged.push_back(q);
            }
            std::string mname = cwd + "/pmf_merged_colored.pcd";
            if (pcl::io::savePCDFileBinary(mname, merged) == 0) std::cout << "Wrote merged colored PCD: " << mname << " (" << merged.size() << ")\n";
            else std::cerr << "Failed to write merged colored PCD\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception while saving: " << e.what() << "\n";
    }

    return 0;
}
