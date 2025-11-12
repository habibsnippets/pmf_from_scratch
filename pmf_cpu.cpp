// pmf_cpu.cpp
// PMF CPU baseline adjusted to match PCL behavior:
//  - linear window growth (additive)
//  - minimum z per cell (not percentile) as cell representative
//  - copy original grid to B and compute diff = B - opened
//  - incremental dh update: dh = min(dhmax, dh + slope * (w - prev_w) * cell_sz)
//  - safe writes and merged colored output
//
// Build with: cmake + make (CMakeLists provided in instructions)

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
#include <unordered_set>

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
    Grid(int r=0, int c=0, double cell_=1.0)
        : rows(r), cols(c), xmin(0), xmax(0), ymin(0), ymax(0), cell(cell_) {
        data.assign(rows*cols, std::numeric_limits<float>::infinity());
    }
    inline float& at(int r, int c) { return data[r*cols + c]; }
    inline const float& at(int r, int c) const { return data[r*cols + c]; }
};

inline int clampi(int v, int a, int b){ return v < a ? a : (v > b ? b : v); }

// rasterize collecting per-cell z-values
void rasterize_collect_min(const pcl::PointCloud<PointT>::Ptr &pc, Grid &G, std::vector<int> &ix, std::vector<int> &iy,
                       std::vector<std::vector<float>> &cellvals) {
    int n = (int)pc->size();
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

// fill grid with minimum per cell (PCL style)
void fill_grid_min(Grid &G, const std::vector<std::vector<float>> &cellvals) {
    int R = G.rows, C = G.cols;
    for (int r=0;r<R;++r){
        for (int c=0;c<C;++c){
            auto const &vals = cellvals[r*C + c];
            if (!vals.empty()) {
                float mn = vals[0];
                for (size_t i=1;i<vals.size(); ++i) if (vals[i] < mn) mn = vals[i];
                G.at(r,c) = mn;
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
    // boundary fallback
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
    // boundary fallback
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

// build linear windows (additive) from base_radius_cells up to max_w
std::vector<int> build_windows_linear_from_base(int max_w, int base_radius_cells) {
    std::vector<int> windows;
    if (base_radius_cells < 1) base_radius_cells = 1;
    int r = base_radius_cells;
    while (true) {
        int w = 2*r + 1;
        if (w > max_w) break;
        windows.push_back(w);
        r += base_radius_cells;
        if (r > 1000000) break;
    }
    if (windows.empty()) {
        // fallback to odd windows 3..max_w
        for (int w=3; w<=max_w; w+=2) windows.push_back(w);
    }
    return windows;
}

// dump grid CSV for debugging
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
         int percentile_cell = 5, // kept for backward compatibility but ignored (we use min)
         bool auto_tune = true,
         const std::string &dump_csv = "",
         int base_radius_cells = 2,
         bool exponential_growth = false)
{
    // bounding box
    double xmin=1e300, ymin=xmin, xmax=-xmin, ymax=-xmin;
    for (const auto &p : incloud->points) {
        xmin = std::min(xmin, (double)p.x);
        xmax = std::max(xmax, (double)p.x);
        ymin = std::min(ymin, (double)p.y);
        ymax = std::max(ymax, (double)p.y);
    }
    int cols = int(std::floor((xmax - xmin) / cell_sz)) + 1;
    int rows = int(std::floor((ymax - ymin) / cell_sz)) + 1;
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Invalid grid size from bbox; check input cloud extents and cell size.\n";
        return;
    }
    Grid G(rows, cols, cell_sz);
    G.xmin = xmin; G.xmax = xmax; G.ymin = ymin; G.ymax = ymax;

    // rasterize (collect per-cell z) and fill with min
    std::vector<int> ix, iy;
    std::vector<std::vector<float>> cellvals;
    rasterize_collect_min(incloud, G, ix, iy, cellvals);
    fill_grid_min(G, cellvals);
    fill_empty_cells_bfs(G);

    if (!dump_csv.empty()) dump_grid_csv(G, dump_csv);

    // copy original grid to B (PCL behavior)
    Grid B = G;

    // windows: use linear growth by base_radius_cells unless exponential_growth==true
    std::vector<int> windows;
    if (!exponential_growth) {
        windows = build_windows_linear_from_base(wmax, base_radius_cells);
    } else {
        // exponential growth (fallback): r = base, r *= 2 until window > wmax
        int r = base_radius_cells;
        while (true) {
            int w = 2*r + 1;
            if (w > wmax) break;
            windows.push_back(w);
            r *= 2;
            if (r > 1000000) break;
        }
        if (windows.empty()) for (int w=3; w<=wmax; w+=2) windows.push_back(w);
    }

    // auto-tune dh0 if requested and dh0 <= 0 (compute simple small default)
    if (auto_tune && dh0 <= 0.0) {
        // compute initial diffs at w = windows[0] if available, else w=3
        int w_init = (windows.size() ? windows[0] : 3);
        Grid er(G.rows, G.cols, cell_sz), op(G.rows, G.cols, cell_sz);
        separable_erosion(G, er, w_init);
        separable_dilation(er, op, w_init);
        std::vector<float> diffs;
        for (int r=0;r<G.rows;++r) for (int c=0;c<G.cols;++c) {
            float d = B.at(r,c) - op.at(r,c);
            if (d > 1e-8f) diffs.push_back(d);
        }
        if (!diffs.empty()) {
            std::sort(diffs.begin(), diffs.end());
            int idx = std::max(0, int(diffs.size()*25/100));
            double guess = diffs[idx];
            dh0 = std::max(0.005, guess);
            std::cerr << "[auto-tune] dh0 set to " << dh0 << " (25th percentile of initial diffs)\n";
        } else {
            dh0 = 0.01;
            std::cerr << "[auto-tune] no diffs found; dh0 fallback to " << dh0 << "\n";
        }
    } else if (dh0 <= 0.0) {
        dh0 = 0.01;
    }

    // optional debug: how many occupied cells
    {
        std::unordered_set<int> occ;
        int total = (int)ix.size();
        for (size_t i=0;i<ix.size();++i) occ.insert(iy[i]*cols + ix[i]);
        std::cerr << "Debug: points=" << incloud->size() << " occupied distinct cells=" << occ.size() << " / " << (rows*cols) << "\n";
    }

    std::vector<char> ground_mask(G.rows * G.cols, 0);

    std::cout << "Grid: rows=" << G.rows << " cols=" << G.cols << " cells=" << (G.rows*G.cols) << "\n";
    std::cout << "Points: " << incloud->size() << "\n";

    // progressive filtering: use incremental dh update like PCL/paper
    double dh = dh0;
    int prev_w = 1;
    for (size_t wi=0; wi<windows.size(); ++wi) {
        int w = windows[wi];
        Grid er(G.rows, G.cols, cell_sz), op(G.rows, G.cols, cell_sz);
        separable_erosion(G, er, w);
        separable_dilation(er, op, w);
        double dh_s = dh;
        if (dh_s > dhmax) dh_s = dhmax;

        int newly_marked = 0;
        float diff_min = std::numeric_limits<float>::infinity(), diff_max = -std::numeric_limits<float>::infinity();
        double diff_sum = 0; int diff_cnt = 0;
        for (int r=0;r<G.rows;++r) for (int c=0;c<G.cols;++c) {
            float d = B.at(r,c) - op.at(r,c); // compare against original B
            diff_min = std::min(diff_min, d);
            diff_max = std::max(diff_max, d);
            diff_sum += d; diff_cnt++;
            if (d < dh_s) {
                int idx = r*G.cols + c;
                if (!ground_mask[idx]) { ground_mask[idx] = 1; newly_marked++; }
            }
        }
        double diff_mean = diff_sum / std::max(1, diff_cnt);
        std::cout << "w=" << w << " dh_s=" << dh_s << " diff=[" << diff_min << "," << diff_max << "] mean=" << diff_mean << " newly_marked=" << newly_marked << "\n";

        // update dh incrementally (paper / PCL style)
        dh = std::min(dhmax, dh + slope * double(w - prev_w) * cell_sz);
        prev_w = w;
    }

    // small post-filter: remove tiny ground islands (connected components smaller than min_size_cells)
    int R = G.rows, C = G.cols;
    std::vector<int> label(R*C, -1);
    int curLabel = 0;
    std::vector<int> stack;
    for (int r=0;r<R;++r){
      for (int c=0;c<C;++c){
        int idx = r*C + c;
        if (ground_mask[idx] && label[idx] == -1){
          stack.clear();
          stack.push_back(idx);
          label[idx] = curLabel;
          for (size_t s=0; s<stack.size(); ++s){
            int v = stack[s];
            int vr = v / C, vc = v % C;
            const int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
            for (int k=0;k<4;++k){
              int nr = vr + dr[k], nc = vc + dc[k];
              if (nr<0||nr>=R||nc<0||nc>=C) continue;
              int ni = nr*C + nc;
              if (ground_mask[ni] && label[ni] == -1){
                label[ni] = curLabel;
                stack.push_back(ni);
              }
            }
          }
          ++curLabel;
        }
      }
    }
    if (curLabel > 0) {
        std::vector<int> sizes(curLabel,0);
        for (int i=0;i<R*C;++i) if (label[i] >= 0) sizes[label[i]]++;
        int min_size_cells = 3; // conservative default; increase to remove more tiny speckles
        for (int i=0;i<R*C;++i){
           if (label[i] >= 0 && sizes[label[i]] < min_size_cells) ground_mask[i] = 0;
        }
    }

    // map back to points
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
        std::cerr << "Usage: pmf_cpu <input.pcd> [cell_sz] [slope] [dh0] [dhmax] [wmax] [percentile_cell] [dump_csv] [base_radius_cells] [exponential(0/1)]\n";
        std::cerr << " Defaults: cell_sz=0.5 slope=0.05 dh0=0(auto-tune) dhmax=3.0 wmax=21 percentile=5 base_radius_cells=2 exponential=0\n";
        return -1;
    }
    std::string infile = argv[1];
    double cell = 0.5, slope = 0.05, dh0 = 0.0, dhmax = 3.0;
    int wmax = 21;
    int percentile_cell = 5;
    std::string dump_csv = "";
    int base_radius_cells = 2;
    bool exponential_growth = false;

    if (argc >= 3) cell = atof(argv[2]);
    if (argc >= 4) slope = atof(argv[3]);
    if (argc >= 5) dh0 = atof(argv[4]);
    if (argc >= 6) dhmax = atof(argv[5]);
    if (argc >= 7) wmax = atoi(argv[6]);
    if (argc >= 8) percentile_cell = atoi(argv[7]);
    if (argc >= 9) dump_csv = std::string(argv[8]);
    if (argc >= 10) base_radius_cells = atoi(argv[9]);
    if (argc >= 11) exponential_growth = (atoi(argv[10]) != 0);

    #ifdef _OPENMP
    omp_set_num_threads(4); // safe default; you can change or expose as arg if you want
    #endif

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(infile, *cloud) < 0) {
        std::cerr << "Failed to read " << infile << "\n"; return -1;
    }
    std::cout << "Loaded cloud with " << cloud->size() << " points\n";

    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr nonground(new pcl::PointCloud<PointT>());

    auto t0 = Clock::now();
    pmf(cloud, ground, nonground, cell, slope, dh0, dhmax, wmax, percentile_cell, /*auto_tune=*/true, dump_csv, base_radius_cells, exponential_growth);
    auto t1 = Clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "PMF runtime: " << secs << " s\n";
    std::cout << "Ground points: " << ground->size() << " Non-ground: " << nonground->size() << "\n";

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
        std::cerr << "Exception while saving PCDs: " << e.what() << "\n";
    }

    return 0;
}
