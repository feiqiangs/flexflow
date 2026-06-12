// =============================================================================
// gdr_vs_memcpy.cu
//
// FlexFlow 微基准：对比 GPU<->Host 内存的两条 H2D / D2H 搬运路径的带宽与时延。
//
//   场景 A (GDR)        : 通过本机网卡做 RDMA loopback（QP 连到自身），
//                         数据经 NIC 在 GPU 显存与 Host 内存之间搬运。
//                         需要 GPUDirect RDMA（nvidia-peermem / nv_peer_mem）。
//   场景 B (cudaMemcpy) : 直接 cudaMemcpyAsync 走 PCIe DMA 做 H2D / D2H。
//
// 关键考量：
//   1. GPU 与网卡亲和性：自动按 PCI 拓扑 / NUMA 选出离指定 GPU 最近的 IB 设备
//      （可用 --nic 覆盖），避免跨 NUMA / 跨 PCIe switch 影响 GDR 结果。
//   2. Warm-up：两种场景在正式计时前都做预热（建链/首包/页锁定/PCIe 训练）。
//   3. 计时统一：两种场景均用 steady_clock 量端到端时延，保证可比。
//
// 编译： 见同目录 Makefile（需要 CUDA + rdma-core-devel）。
// 运行： ./gdr_vs_memcpy [--gpu N] [--nic mlx5_x] [--gid IDX] [--csv out.csv]
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

// ----------------------------------------------------------------------------
// 错误检查宏
// ----------------------------------------------------------------------------
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
static const char* dir_name(int d) { return d == H2D ? "H2D" : "D2H"; }

// ============================================================================
// 1. GPU <-> NIC 亲和性选择
// ============================================================================
//
// 思路：取 GPU 的 PCI BDF（cudaDeviceGetPCIBusId），与每个 IB 设备的 PCI BDF
// （/sys/class/infiniband/<dev>/device 的软链接目标）做匹配，按以下优先级打分：
//   (a) 同一 PCIe 段（domain:bus 前缀越长越近，代表挂在同一 switch 下）；
//   (b) 同一 NUMA node；
// 选总分最高者。可用 --nic 手工指定覆盖自动选择。

struct NicCandidate {
    std::string name;     // ibv 设备名，如 mlx5_0
    std::string pci_bdf;  // 0000:1a:00.0
    int         numa = -1;
    int         score = 0;
};

// 读取 sysfs 文件首行
static std::string read_first_line(const fs::path& p) {
    std::ifstream f(p);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

// 由 /sys/.../device 软链接解析出 PCI BDF（取末段，去掉可能的 functions 多层）
static std::string sysfs_pci_bdf(const fs::path& device_link) {
    std::error_code ec;
    if (!fs::exists(device_link, ec)) return {};
    fs::path real = fs::read_symlink(device_link, ec);
    if (ec) {
        real = fs::canonical(device_link, ec);
        if (ec) return {};
    }
    return real.filename().string();  // e.g. 0000:1a:00.0
}

static int sysfs_numa(const fs::path& dir) {
    std::string s = read_first_line(dir / "numa_node");
    if (s.empty()) return -1;
    return atoi(s.c_str());
}

// 取 GPU 的 PCI BDF（小写，形如 0000:1a:00.0）
static std::string gpu_pci_bdf(int gpu) {
    char buf[64] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(buf, sizeof(buf), gpu));
    std::string s(buf);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;  // domain:bus:dev.func
}

// 公共前缀长度（字符级），用于近似 PCIe 拓扑距离
static int common_prefix_len(const std::string& a, const std::string& b) {
    int n = 0;
    while (n < (int)a.size() && n < (int)b.size() && a[n] == b[n]) ++n;
    return n;
}

// 选出离 gpu 最近的 NIC；force_name 非空时直接用指定设备
static NicCandidate pick_nic_for_gpu(int gpu, const std::string& force_name) {
    int num = 0;
    ibv_device** list = ibv_get_device_list(&num);
    if (!list || num == 0) DIE("no RDMA/IB devices found (ibv_get_device_list)");

    std::string gpu_bdf = gpu_pci_bdf(gpu);
    fs::path gpu_sys = fs::path("/sys/bus/pci/devices") / gpu_bdf;
    int gpu_numa = sysfs_numa(gpu_sys);

    printf("[affinity] GPU %d PCI=%s NUMA=%d\n", gpu, gpu_bdf.c_str(), gpu_numa);

    std::vector<NicCandidate> cands;
    for (int i = 0; i < num; ++i) {
        NicCandidate c;
        c.name = ibv_get_device_name(list[i]);
        fs::path ibsys = fs::path("/sys/class/infiniband") / c.name / "device";
        c.pci_bdf = sysfs_pci_bdf(ibsys);
        c.numa = sysfs_numa(ibsys);

        // 打分：PCI 公共前缀（×10）+ 同 NUMA 奖励
        c.score = common_prefix_len(gpu_bdf, c.pci_bdf) * 10;
        if (gpu_numa >= 0 && c.numa == gpu_numa) c.score += 5;
        cands.push_back(c);

        printf("[affinity]   nic %-10s PCI=%-12s NUMA=%d score=%d\n",
               c.name.c_str(), c.pci_bdf.c_str(), c.numa, c.score);
    }
    ibv_free_device_list(list);

    if (!force_name.empty()) {
        for (auto& c : cands)
            if (c.name == force_name) {
                printf("[affinity] using forced NIC %s\n", c.name.c_str());
                return c;
            }
        DIE("forced NIC '%s' not found", force_name.c_str());
    }

    auto best = std::max_element(cands.begin(), cands.end(),
        [](const NicCandidate& a, const NicCandidate& b) { return a.score < b.score; });
    printf("[affinity] selected NIC %s (closest to GPU %d)\n", best->name.c_str(), gpu);
    return *best;
}

// ============================================================================
// 2. RDMA loopback 上下文（QP 连接到自身）
// ============================================================================
struct RdmaCtx {
    ibv_context*   ctx  = nullptr;
    ibv_pd*        pd   = nullptr;
    ibv_cq*        cq   = nullptr;
    ibv_qp*        qp   = nullptr;
    uint8_t        port = 1;
    int            gid_index = 0;
    ibv_port_attr  port_attr{};
    ibv_gid        gid{};

    void open(const std::string& dev_name, int gid_idx) {
        int num = 0;
        ibv_device** list = ibv_get_device_list(&num);
        ibv_device* dev = nullptr;
        for (int i = 0; i < num; ++i)
            if (dev_name == ibv_get_device_name(list[i])) { dev = list[i]; break; }
        if (!dev) DIE("device %s not found", dev_name.c_str());

        ctx = ibv_open_device(dev);
        ibv_free_device_list(list);
        if (!ctx) DIE("ibv_open_device(%s) failed", dev_name.c_str());

        pd = ibv_alloc_pd(ctx);
        if (!pd) DIE("ibv_alloc_pd failed");

        cq = ibv_create_cq(ctx, 1024, nullptr, nullptr, 0);
        if (!cq) DIE("ibv_create_cq failed");

        if (ibv_query_port(ctx, port, &port_attr)) DIE("ibv_query_port failed");

        // RoCE（Ethernet 链路层）必须用 GID；自动挑一个 RoCEv2 GID 或用指定 index
        gid_index = (gid_idx >= 0) ? gid_idx : 0;
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET && gid_idx < 0)
            gid_index = find_roce_v2_gid();
        if (ibv_query_gid(ctx, port, gid_index, &gid)) DIE("ibv_query_gid failed");

        create_and_connect_qp();
        printf("[rdma] %s port=%u link=%s gid_index=%d mtu=%d qpn=%u\n",
               dev_name.c_str(), port,
               port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ? "RoCE" : "IB",
               gid_index, 128 << port_attr.active_mtu, qp->qp_num);
    }

    // 在 RoCE 端口上找一个 RoCEv2 + IPv4 的 GID index（读 sysfs gid_attrs）
    int find_roce_v2_gid() {
        const char* devname = ibv_get_device_name(ibv_context_ops_dev());
        // 简化：直接读 /sys/class/infiniband/<dev>/ports/<port>/gid_attrs/types/*
        // 找到 "RoCE v2"。失败则回退 0。
        fs::path base = fs::path("/sys/class/infiniband") / devname /
                        "ports" / std::to_string(port) / "gid_attrs" / "types";
        for (int idx = 0; idx < 16; ++idx) {
            std::string t = read_first_line(base / std::to_string(idx));
            if (t.find("RoCE v2") != std::string::npos) return idx;
        }
        return 0;
    }
    // 为拿设备名做的小工具：从 ctx 反查
    ibv_device* ibv_context_ops_dev() { return ctx->device; }

    void create_and_connect_qp() {
        ibv_qp_init_attr ia{};
        ia.send_cq = cq;
        ia.recv_cq = cq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr  = 512;
        ia.cap.max_recv_wr  = 16;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        qp = ibv_create_qp(pd, &ia);
        if (!qp) DIE("ibv_create_qp failed");

        // INIT
        ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_INIT;
        a.pkey_index      = 0;
        a.port_num        = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            DIE("modify_qp -> INIT failed");

        // RTR（目的地 = 自身：dest_qp_num = 本 QP）
        ibv_qp_attr r{};
        r.qp_state           = IBV_QPS_RTR;
        r.path_mtu           = port_attr.active_mtu;
        r.dest_qp_num        = qp->qp_num;
        r.rq_psn             = 0;
        r.max_dest_rd_atomic = 1;
        r.min_rnr_timer      = 12;
        r.ah_attr.port_num   = port;
        r.ah_attr.sl         = 0;
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
            // RoCE：必须走 GRH，dgid = 本端 gid
            r.ah_attr.is_global  = 1;
            r.ah_attr.grh.dgid          = gid;
            r.ah_attr.grh.sgid_index    = gid_index;
            r.ah_attr.grh.hop_limit     = 1;
            r.ah_attr.grh.traffic_class = 0;
        } else {
            r.ah_attr.is_global = 0;
            r.ah_attr.dlid      = port_attr.lid;  // loopback 到本端 LID
        }
        if (ibv_modify_qp(qp, &r,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            DIE("modify_qp -> RTR failed");

        // RTS
        ibv_qp_attr s{};
        s.qp_state      = IBV_QPS_RTS;
        s.sq_psn        = 0;
        s.timeout       = 14;
        s.retry_cnt     = 7;
        s.rnr_retry     = 7;
        s.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &s,
                IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
            DIE("modify_qp -> RTS failed");
    }

    ibv_mr* reg(void* addr, size_t len) {
        ibv_mr* mr = ibv_reg_mr(pd, addr, len,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr) DIE("ibv_reg_mr(%p,%zu) failed (GPUDirect RDMA 未启用?)", addr, len);
        return mr;
    }

    // 发起 n 个 RDMA_WRITE：把 local 内容写到 remote（loopback 同 QP）。
    // 仅最后一个置 SIGNALED，完成即代表整批完成。
    void post_write_batch(uint64_t local_addr, uint32_t lkey,
                          uint64_t remote_addr, uint32_t rkey,
                          uint32_t len, int n) {
        for (int i = 0; i < n; ++i) {
            ibv_sge sge{};
            sge.addr   = local_addr;
            sge.length = len;
            sge.lkey   = lkey;

            ibv_send_wr wr{};
            wr.wr_id      = (uint64_t)i;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            wr.opcode     = IBV_WR_RDMA_WRITE;
            wr.send_flags = (i == n - 1) ? IBV_SEND_SIGNALED : 0;
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey        = rkey;

            ibv_send_wr* bad = nullptr;
            int rc = ibv_post_send(qp, &wr, &bad);
            if (rc) DIE("ibv_post_send failed rc=%d (i=%d)", rc, i);
        }
        poll_one();
    }

    void poll_one() {
        ibv_wc wc{};
        for (;;) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) DIE("ibv_poll_cq failed");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS)
                DIE("WC error: %s", ibv_wc_status_str(wc.status));
            return;
        }
    }

    void destroy(std::vector<ibv_mr*>& mrs) {
        for (auto* mr : mrs) if (mr) ibv_dereg_mr(mr);
        if (qp) ibv_destroy_qp(qp);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// ============================================================================
// 3. 统计与打印
// ============================================================================
struct Stat { double bw_gbps; double lat_us; };

static Stat make_stat(size_t bytes_per_iter, double total_sec, int iters) {
    Stat s;
    s.lat_us  = total_sec / iters * 1e6;
    s.bw_gbps = (double)bytes_per_iter * iters / total_sec / 1e9;  // GB/s (1e9)
    return s;
}

// ============================================================================
// 4. 场景 A：GDR（RDMA loopback）基准
// ============================================================================
static Stat bench_gdr(RdmaCtx& rc, int dir,
                      ibv_mr* host_mr, ibv_mr* dev_mr, void* host_p, void* dev_p,
                      size_t size, int batch, int iters, int warmup) {
    // H2D: 源=host(local) 目的=GPU(remote)；D2H: 源=GPU(local) 目的=host(remote)
    uint64_t loc, rem; uint32_t lkey, rkey;
    if (dir == H2D) {
        loc = (uint64_t)host_p; lkey = host_mr->lkey;
        rem = (uint64_t)dev_p;  rkey = dev_mr->rkey;
    } else {
        loc = (uint64_t)dev_p;  lkey = dev_mr->lkey;
        rem = (uint64_t)host_p; rkey = host_mr->rkey;
    }

    // warm-up（建链后首批传输 + PCIe / GDR 通路训练）
    for (int i = 0; i < warmup; ++i)
        rc.post_write_batch(loc, lkey, rem, rkey, (uint32_t)size, batch);

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        rc.post_write_batch(loc, lkey, rem, rkey, (uint32_t)size, batch);
    auto t1 = Clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    return make_stat(size * (size_t)batch, sec, iters);
}

// ============================================================================
// 5. 场景 B：cudaMemcpyAsync 基准
// ============================================================================
static Stat bench_memcpy(int dir, void* host_p, void* dev_p,
                         cudaStream_t stream, size_t size, int batch,
                         int iters, int warmup) {
    auto submit_batch = [&]() {
        for (int i = 0; i < batch; ++i) {
            if (dir == H2D)
                CUDA_CHECK(cudaMemcpyAsync(dev_p, host_p, size,
                                           cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpyAsync(host_p, dev_p, size,
                                           cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    for (int i = 0; i < warmup; ++i) submit_batch();

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) submit_batch();
    auto t1 = Clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    return make_stat(size * (size_t)batch, sec, iters);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    int gpu = 0;
    int gid_index = -1;          // -1 = 自动
    std::string force_nic;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gpu")       gpu = atoi(next().c_str());
        else if (a == "--nic")  force_nic = next();
        else if (a == "--gid")  gid_index = atoi(next().c_str());
        else if (a == "--csv")  csv_path = next();
        else if (a == "-h" || a == "--help") {
            printf("usage: %s [--gpu N] [--nic mlx5_x] [--gid IDX] [--csv out.csv]\n", argv[0]);
            return 0;
        }
    }

    CUDA_CHECK(cudaSetDevice(gpu));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
    printf("=== FlexFlow GDR vs cudaMemcpy H2D/D2H benchmark ===\n");
    printf("[gpu] %d: %s\n", gpu, prop.name);

    // ---- 亲和性选 NIC + 建 RDMA loopback ----
    NicCandidate nic = pick_nic_for_gpu(gpu, force_nic);
    RdmaCtx rc;
    rc.open(nic.name, gid_index);

    // ---- 测试规模：1KB ~ 256MB ----
    std::vector<size_t> sizes;
    for (size_t s = (1 << 10); s <= (256ULL << 20); s <<= 1) sizes.push_back(s);
    size_t max_size = sizes.back();
    const int batch  = 16;   // 每次测量批量提交的传输数
    const int iters  = 50;   // 计时迭代次数
    const int warmup = 10;   // 预热次数

    // ---- 分配并注册内存（最大规模一次分配，复用）----
    void* host_p = nullptr;
    void* dev_p  = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_p, max_size));   // pinned host
    memset(host_p, 0xA5, max_size);
    CUDA_CHECK(cudaMalloc(&dev_p, max_size));         // device

    ibv_mr* host_mr = rc.reg(host_p, max_size);
    ibv_mr* dev_mr  = rc.reg(dev_p, max_size);        // 依赖 GPUDirect RDMA
    std::vector<ibv_mr*> mrs = {host_mr, dev_mr};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- 输出表头 ----
    std::ofstream fout;
    if (!csv_path.empty()) {
        fout.open(csv_path);
        fout << "direction,size_bytes,batch,"
                "gdr_bw_gbps,gdr_lat_us,memcpy_bw_gbps,memcpy_lat_us,bw_ratio_gdr_over_memcpy\n";
    }

    printf("\n%-4s %10s | %12s %11s | %12s %11s | %8s\n",
           "dir", "size", "GDR GB/s", "GDR us", "cMemcpy GB/s", "cMemcpy us", "A/B bw");
    printf("--------------------------------------------------------------------------------------\n");

    for (int dir = H2D; dir <= D2H; ++dir) {
        for (size_t size : sizes) {
            Stat a = bench_gdr(rc, dir, host_mr, dev_mr, host_p, dev_p,
                               size, batch, iters, warmup);
            Stat b = bench_memcpy(dir, host_p, dev_p, stream,
                                  size, batch, iters, warmup);

            double ratio = (b.bw_gbps > 0) ? a.bw_gbps / b.bw_gbps : 0.0;

            const char* unit = size < (1 << 20) ? "KB" : "MB";
            double dsz = size < (1 << 20) ? size / 1024.0 : size / (1024.0 * 1024.0);
            printf("%-4s %7.1f%s | %12.3f %11.2f | %12.3f %11.2f | %8.2f\n",
                   dir_name(dir), dsz, unit,
                   a.bw_gbps, a.lat_us, b.bw_gbps, b.lat_us, ratio);

            if (fout)
                fout << dir_name(dir) << "," << size << "," << batch << ","
                     << a.bw_gbps << "," << a.lat_us << ","
                     << b.bw_gbps << "," << b.lat_us << "," << ratio << "\n";
        }
    }

    // ---- 清理 ----
    CUDA_CHECK(cudaStreamDestroy(stream));
    rc.destroy(mrs);
    CUDA_CHECK(cudaFree(dev_p));
    CUDA_CHECK(cudaFreeHost(host_p));
    if (fout) { fout.close(); printf("\nCSV written to %s\n", csv_path.c_str()); }

    printf("\n=== done ===\n");
    return 0;
}
