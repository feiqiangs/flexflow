// =============================================================================
// gdr_vs_memcpy.cu
//
// FlexFlow 微基准：对比 GPU<->Host 内存的 H2D / D2H 搬运路径的带宽与时延。
//
//   场景 A (GDR)        : 通过本机网卡做 RDMA loopback（QP 连到自身），
//                         数据经 NIC 在 GPU 显存与 Host 内存之间搬运。
//                         需要 GPUDirect RDMA（nvidia-peermem / nv_peer_mem）。
//                         支持「多 NIC × 多 QP」并发 + RDMA WRITE / READ。
//   场景 B (cudaMemcpy) : 直接 cudaMemcpyAsync 走 PCIe DMA 做 H2D / D2H。
//
// 关键考量：
//   1. GPU 与网卡亲和性：按 PCI 拓扑 / NUMA 给所有 NIC 打分排序，默认选最近的
//      若干张卡（--nics N），用以观察「多卡并发能否逼近 PCIe 上限」。
//   2. Warm-up：两种场景在正式计时前都做预热。
//   3. 计时统一：两种场景均用 steady_clock 量端到端时延，保证可比。
//
// 用法：
//   ./gdr_vs_memcpy [--gpu N] [--nics K] [--qp-per-nic Q]
//                   [--opcode write|read] [--gid IDX] [--csv out.csv]
//   - 单 QP 基线：       --nics 1 --qp-per-nic 1 --opcode write
//   - 多 NIC×多 QP 并发：--nics 4 --qp-per-nic 2 --opcode write
//   - RDMA READ 对比：   --opcode read
// =============================================================================

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(_e));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define DIE(...)                                                               \
    do {                                                                       \
        fprintf(stderr, "[FATAL] " __VA_ARGS__);                               \
        fprintf(stderr, "\n");                                                 \
        exit(EXIT_FAILURE);                                                    \
    } while (0)

enum Direction { H2D = 0, D2H = 1 };
enum OpCode    { OP_WRITE = 0, OP_READ = 1 };
static const char* dir_name(int d) { return d == H2D ? "H2D" : "D2H"; }

// ============================================================================
// 1. GPU <-> NIC 亲和性排序
// ============================================================================
struct NicCandidate {
    std::string name;     // ibv 设备名，如 mlx5_bond_0
    std::string pci_bdf;  // 0000:04:00.0
    int         numa = -1;
    int         score = 0;
};

static std::string read_first_line(const fs::path& p) {
    std::ifstream f(p);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

static std::string sysfs_pci_bdf(const fs::path& device_link) {
    std::error_code ec;
    if (!fs::exists(device_link, ec)) return {};
    fs::path real = fs::read_symlink(device_link, ec);
    if (ec) {
        real = fs::canonical(device_link, ec);
        if (ec) return {};
    }
    return real.filename().string();
}

static int sysfs_numa(const fs::path& dir) {
    std::string s = read_first_line(dir / "numa_node");
    if (s.empty()) return -1;
    return atoi(s.c_str());
}

static std::string gpu_pci_bdf(int gpu) {
    char buf[64] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(buf, sizeof(buf), gpu));
    std::string s(buf);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

static int common_prefix_len(const std::string& a, const std::string& b) {
    int n = 0;
    while (n < (int)a.size() && n < (int)b.size() && a[n] == b[n]) ++n;
    return n;
}

// 返回按「离 gpu 由近到远」排序的 NIC 列表
static std::vector<NicCandidate> rank_nics_for_gpu(int gpu) {
    int num = 0;
    ibv_device** list = ibv_get_device_list(&num);
    if (!list || num == 0) DIE("no RDMA/IB devices found (ibv_get_device_list)");

    std::string gpu_bdf = gpu_pci_bdf(gpu);
    int gpu_numa = sysfs_numa(fs::path("/sys/bus/pci/devices") / gpu_bdf);
    printf("[affinity] GPU %d PCI=%s NUMA=%d\n", gpu, gpu_bdf.c_str(), gpu_numa);

    std::vector<NicCandidate> cands;
    for (int i = 0; i < num; ++i) {
        NicCandidate c;
        c.name = ibv_get_device_name(list[i]);
        fs::path ibsys = fs::path("/sys/class/infiniband") / c.name / "device";
        c.pci_bdf = sysfs_pci_bdf(ibsys);
        c.numa = sysfs_numa(ibsys);
        c.score = common_prefix_len(gpu_bdf, c.pci_bdf) * 10;
        if (gpu_numa >= 0 && c.numa == gpu_numa) c.score += 5;
        cands.push_back(c);
    }
    ibv_free_device_list(list);

    std::sort(cands.begin(), cands.end(),
              [](const NicCandidate& a, const NicCandidate& b) { return a.score > b.score; });
    for (auto& c : cands)
        printf("[affinity]   nic %-12s PCI=%-13s NUMA=%d score=%d\n",
               c.name.c_str(), c.pci_bdf.c_str(), c.numa, c.score);
    return cands;
}

// 在 Ethernet(RoCE) 端口上找 RoCEv2 GID index
static int find_roce_v2_gid(ibv_context* ctx, uint8_t port) {
    const char* devname = ibv_get_device_name(ctx->device);
    fs::path base = fs::path("/sys/class/infiniband") / devname /
                    "ports" / std::to_string(port) / "gid_attrs" / "types";
    for (int idx = 0; idx < 16; ++idx) {
        std::string t = read_first_line(base / std::to_string(idx));
        if (t.find("RoCE v2") != std::string::npos) return idx;
    }
    return 0;
}

// ============================================================================
// 2. RDMA loopback 引擎：多 NIC × 多 QP
// ============================================================================
struct NicCtx {
    std::string   name;
    ibv_context*  ctx  = nullptr;
    ibv_pd*       pd   = nullptr;
    uint8_t       port = 1;
    int           gid_index = 0;
    ibv_port_attr port_attr{};
    ibv_gid       gid{};
    ibv_mr*       host_mr = nullptr;  // host 缓冲在本 NIC 的 PD 上的注册
    ibv_mr*       dev_mr  = nullptr;  // device 缓冲在本 NIC 的 PD 上的注册
};

struct Lane {
    NicCtx*  nic = nullptr;
    ibv_cq*  cq  = nullptr;
    ibv_qp*  qp  = nullptr;
};

struct GdrEngine {
    std::vector<NicCtx> nics;
    std::vector<Lane>   lanes;
    int    opcode = OP_WRITE;

    void open_nic(NicCtx& n, const std::string& dev_name, int gid_idx,
                  void* host_p, void* dev_p, size_t bytes) {
        int num = 0;
        ibv_device** list = ibv_get_device_list(&num);
        ibv_device* dev = nullptr;
        for (int i = 0; i < num; ++i)
            if (dev_name == ibv_get_device_name(list[i])) { dev = list[i]; break; }
        if (!dev) DIE("device %s not found", dev_name.c_str());
        n.name = dev_name;
        n.ctx = ibv_open_device(dev);
        ibv_free_device_list(list);
        if (!n.ctx) DIE("ibv_open_device(%s) failed", dev_name.c_str());

        n.pd = ibv_alloc_pd(n.ctx);
        if (!n.pd) DIE("ibv_alloc_pd failed");
        if (ibv_query_port(n.ctx, n.port, &n.port_attr)) DIE("ibv_query_port failed");

        n.gid_index = (gid_idx >= 0) ? gid_idx : 0;
        if (n.port_attr.link_layer == IBV_LINK_LAYER_ETHERNET && gid_idx < 0)
            n.gid_index = find_roce_v2_gid(n.ctx, n.port);
        if (ibv_query_gid(n.ctx, n.port, n.gid_index, &n.gid)) DIE("ibv_query_gid failed");

        int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        n.host_mr = ibv_reg_mr(n.pd, host_p, bytes, acc);
        n.dev_mr  = ibv_reg_mr(n.pd, dev_p,  bytes, acc);
        if (!n.host_mr || !n.dev_mr)
            DIE("ibv_reg_mr failed on %s (GPUDirect RDMA 未启用?)", dev_name.c_str());

        printf("[rdma] %s port=%u link=%s gid_index=%d mtu=%d\n",
               dev_name.c_str(), n.port,
               n.port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ? "RoCE" : "IB",
               n.gid_index, 128 << n.port_attr.active_mtu);
    }

    void connect_self(ibv_qp* qp, NicCtx& n) {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = n.port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            DIE("modify_qp -> INIT failed");

        ibv_qp_attr r{};
        r.qp_state = IBV_QPS_RTR;
        r.path_mtu = n.port_attr.active_mtu;
        r.dest_qp_num = qp->qp_num;          // loopback：目的就是自己
        r.rq_psn = 0;
        r.max_dest_rd_atomic = 1;
        r.min_rnr_timer = 12;
        r.ah_attr.port_num = n.port;
        r.ah_attr.sl = 0;
        if (n.port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
            r.ah_attr.is_global = 1;
            r.ah_attr.grh.dgid = n.gid;
            r.ah_attr.grh.sgid_index = n.gid_index;
            r.ah_attr.grh.hop_limit = 1;
        } else {
            r.ah_attr.is_global = 0;
            r.ah_attr.dlid = n.port_attr.lid;
        }
        if (ibv_modify_qp(qp, &r,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            DIE("modify_qp -> RTR failed");

        ibv_qp_attr s{};
        s.qp_state = IBV_QPS_RTS;
        s.sq_psn = 0;
        s.timeout = 14;
        s.retry_cnt = 7;
        s.rnr_retry = 7;
        s.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &s,
                IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
            DIE("modify_qp -> RTS failed");
    }

    Lane make_lane(NicCtx& n) {
        Lane l;
        l.nic = &n;
        l.cq = ibv_create_cq(n.ctx, 1024, nullptr, nullptr, 0);
        if (!l.cq) DIE("ibv_create_cq failed");
        ibv_qp_init_attr ia{};
        ia.send_cq = l.cq;
        ia.recv_cq = l.cq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = 512;
        ia.cap.max_recv_wr = 16;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        l.qp = ibv_create_qp(n.pd, &ia);
        if (!l.qp) DIE("ibv_create_qp failed");
        connect_self(l.qp, n);
        return l;
    }

    void setup(const std::vector<NicCandidate>& ranked, int num_nics, int qp_per_nic,
               int opc, int gid_idx, void* host_p, void* dev_p, size_t bytes) {
        opcode = opc;
        int use = std::min<int>(num_nics, (int)ranked.size());
        nics.resize(use);
        for (int i = 0; i < use; ++i)
            open_nic(nics[i], ranked[i].name, gid_idx, host_p, dev_p, bytes);
        for (int i = 0; i < use; ++i)
            for (int q = 0; q < qp_per_nic; ++q)
                lanes.push_back(make_lane(nics[i]));
        printf("[rdma] engine: %d NIC(s) x %d QP = %zu lanes, opcode=%s\n",
               use, qp_per_nic, lanes.size(), opcode == OP_WRITE ? "WRITE" : "READ");
    }

    // 解析某条 lane 上一次操作的 (本地 addr/lkey, 远端 addr/rkey)
    // WRITE: 把 local 写到 remote；READ: 把 remote 读进 local。
    // 目标都让数据净移动方向 = dir。
    void resolve(Lane& l, int dir, void* host_p, void* dev_p,
                 uint64_t& loc, uint32_t& lkey, uint64_t& rem, uint32_t& rkey) {
        ibv_mr* hmr = l.nic->host_mr;
        ibv_mr* dmr = l.nic->dev_mr;
        bool data_to_gpu = (dir == H2D);
        // 数据终点在 GPU：WRITE 时 local=host,remote=gpu；READ 时 local=gpu,remote=host
        bool local_is_host = (opcode == OP_WRITE) ? data_to_gpu : !data_to_gpu;
        if (local_is_host) {
            loc = (uint64_t)host_p; lkey = hmr->lkey;
            rem = (uint64_t)dev_p;  rkey = dmr->rkey;
        } else {
            loc = (uint64_t)dev_p;  lkey = dmr->lkey;
            rem = (uint64_t)host_p; rkey = hmr->rkey;
        }
    }

    void post_n(Lane& l, uint64_t loc, uint32_t lkey, uint64_t rem, uint32_t rkey,
                uint32_t len, int n) {
        for (int i = 0; i < n; ++i) {
            ibv_sge sge{};
            sge.addr = loc; sge.length = len; sge.lkey = lkey;
            ibv_send_wr wr{};
            wr.wr_id = (uint64_t)i;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = (opcode == OP_WRITE) ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
            wr.send_flags = (i == n - 1) ? IBV_SEND_SIGNALED : 0;
            wr.wr.rdma.remote_addr = rem;
            wr.wr.rdma.rkey = rkey;
            ibv_send_wr* bad = nullptr;
            int rc = ibv_post_send(l.qp, &wr, &bad);
            if (rc) DIE("ibv_post_send failed rc=%d", rc);
        }
    }

    void poll_one(Lane& l) {
        ibv_wc wc{};
        for (;;) {
            int n = ibv_poll_cq(l.cq, 1, &wc);
            if (n < 0) DIE("ibv_poll_cq failed");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS)
                DIE("WC error on %s: %s", l.nic->name.c_str(), ibv_wc_status_str(wc.status));
            return;
        }
    }

    // 把 batch 个 size 大小的传输分摊到所有 lane 上并发执行，等全部完成。
    void run_batch(int dir, size_t size, int batch, void* host_p, void* dev_p) {
        int L = (int)lanes.size();
        int per = batch / L, rem = batch % L;
        std::vector<int> nops(L);
        for (int i = 0; i < L; ++i) nops[i] = per + (i < rem ? 1 : 0);
        // 先全部投递（HW 在各 QP/NIC 上并发处理），再统一收割
        for (int i = 0; i < L; ++i) {
            if (nops[i] == 0) continue;
            uint64_t loc, rem_addr; uint32_t lkey, rkey;
            resolve(lanes[i], dir, host_p, dev_p, loc, lkey, rem_addr, rkey);
            post_n(lanes[i], loc, lkey, rem_addr, rkey, (uint32_t)size, nops[i]);
        }
        for (int i = 0; i < L; ++i)
            if (nops[i] > 0) poll_one(lanes[i]);
    }

    void destroy() {
        for (auto& l : lanes) { if (l.qp) ibv_destroy_qp(l.qp); if (l.cq) ibv_destroy_cq(l.cq); }
        for (auto& n : nics) {
            if (n.host_mr) ibv_dereg_mr(n.host_mr);
            if (n.dev_mr)  ibv_dereg_mr(n.dev_mr);
            if (n.pd) ibv_dealloc_pd(n.pd);
            if (n.ctx) ibv_close_device(n.ctx);
        }
    }
};

// ============================================================================
// 3. 统计
// ============================================================================
struct Stat { double bw_gbps; double lat_us; };
static Stat make_stat(size_t bytes_per_iter, double total_sec, int iters) {
    Stat s;
    s.lat_us = total_sec / iters * 1e6;
    s.bw_gbps = (double)bytes_per_iter * iters / total_sec / 1e9;
    return s;
}

// ============================================================================
// 4. 场景 A：GDR（多 lane）基准
// ============================================================================
static Stat bench_gdr(GdrEngine& eng, int dir, void* host_p, void* dev_p,
                      size_t size, int batch, int iters, int warmup) {
    for (int i = 0; i < warmup; ++i) eng.run_batch(dir, size, batch, host_p, dev_p);
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) eng.run_batch(dir, size, batch, host_p, dev_p);
    auto t1 = Clock::now();
    return make_stat(size * (size_t)batch, std::chrono::duration<double>(t1 - t0).count(), iters);
}

// ============================================================================
// 5. 场景 B：cudaMemcpyAsync 基准
// ============================================================================
static Stat bench_memcpy(int dir, void* host_p, void* dev_p, cudaStream_t stream,
                         size_t size, int batch, int iters, int warmup) {
    auto submit = [&]() {
        for (int i = 0; i < batch; ++i) {
            if (dir == H2D)
                CUDA_CHECK(cudaMemcpyAsync(dev_p, host_p, size, cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpyAsync(host_p, dev_p, size, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    for (int i = 0; i < warmup; ++i) submit();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) submit();
    auto t1 = Clock::now();
    return make_stat(size * (size_t)batch, std::chrono::duration<double>(t1 - t0).count(), iters);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    int gpu = 0, gid_index = -1, num_nics = 1, qp_per_nic = 1, opcode = OP_WRITE;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gpu")            gpu = atoi(next().c_str());
        else if (a == "--nics")      num_nics = atoi(next().c_str());
        else if (a == "--qp-per-nic")qp_per_nic = atoi(next().c_str());
        else if (a == "--opcode")    opcode = (next() == "read") ? OP_READ : OP_WRITE;
        else if (a == "--gid")       gid_index = atoi(next().c_str());
        else if (a == "--csv")       csv_path = next();
        else if (a == "-h" || a == "--help") {
            printf("usage: %s [--gpu N] [--nics K] [--qp-per-nic Q] "
                   "[--opcode write|read] [--gid IDX] [--csv out.csv]\n", argv[0]);
            return 0;
        }
    }
    if (num_nics < 1) num_nics = 1;
    if (qp_per_nic < 1) qp_per_nic = 1;

    CUDA_CHECK(cudaSetDevice(gpu));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
    printf("=== FlexFlow GDR vs cudaMemcpy H2D/D2H benchmark ===\n");
    printf("[gpu] %d: %s\n", gpu, prop.name);
    printf("[cfg] nics=%d qp_per_nic=%d opcode=%s\n",
           num_nics, qp_per_nic, opcode == OP_WRITE ? "WRITE" : "READ");

    std::vector<NicCandidate> ranked = rank_nics_for_gpu(gpu);

    std::vector<size_t> sizes;
    for (size_t s = (1 << 10); s <= (256ULL << 20); s <<= 1) sizes.push_back(s);
    size_t max_size = sizes.back();
    const int batch = 16, iters = 50, warmup = 10;

    void* host_p = nullptr; void* dev_p = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_p, max_size));
    memset(host_p, 0xA5, max_size);
    CUDA_CHECK(cudaMalloc(&dev_p, max_size));

    GdrEngine eng;
    eng.setup(ranked, num_nics, qp_per_nic, opcode, gid_index, host_p, dev_p, max_size);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::ofstream fout;
    if (!csv_path.empty()) {
        fout.open(csv_path);
        fout << "direction,size_bytes,batch,nics,qp_per_nic,opcode,"
                "gdr_bw_gbps,gdr_lat_us,memcpy_bw_gbps,memcpy_lat_us,bw_ratio_gdr_over_memcpy\n";
    }
    const char* opc = opcode == OP_WRITE ? "WRITE" : "READ";

    printf("\n%-4s %10s | %12s %11s | %12s %11s | %8s\n",
           "dir", "size", "GDR GB/s", "GDR us", "cMemcpy GB/s", "cMemcpy us", "A/B bw");
    printf("--------------------------------------------------------------------------------------\n");

    for (int dir = H2D; dir <= D2H; ++dir) {
        for (size_t size : sizes) {
            Stat a = bench_gdr(eng, dir, host_p, dev_p, size, batch, iters, warmup);
            Stat b = bench_memcpy(dir, host_p, dev_p, stream, size, batch, iters, warmup);
            double ratio = (b.bw_gbps > 0) ? a.bw_gbps / b.bw_gbps : 0.0;

            const char* unit = size < (1 << 20) ? "KB" : "MB";
            double dsz = size < (1 << 20) ? size / 1024.0 : size / (1024.0 * 1024.0);
            printf("%-4s %7.1f%s | %12.3f %11.2f | %12.3f %11.2f | %8.2f\n",
                   dir_name(dir), dsz, unit, a.bw_gbps, a.lat_us, b.bw_gbps, b.lat_us, ratio);
            if (fout)
                fout << dir_name(dir) << "," << size << "," << batch << ","
                     << num_nics << "," << qp_per_nic << "," << opc << ","
                     << a.bw_gbps << "," << a.lat_us << ","
                     << b.bw_gbps << "," << b.lat_us << "," << ratio << "\n";
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    eng.destroy();
    CUDA_CHECK(cudaFree(dev_p));
    CUDA_CHECK(cudaFreeHost(host_p));
    if (fout) { fout.close(); printf("\nCSV written to %s\n", csv_path.c_str()); }
    printf("\n=== done ===\n");
    return 0;
}
