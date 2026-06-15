# GDR vs cudaMemcpy H2D/D2H 基准

对比 GPU 显存与 Host 内存之间两条数据搬运通路的**带宽**与**时延**：

| 场景 | 路径 | 实现 |
|------|------|------|
| **A — GDR** | GPU 显存 ⇄ NIC ⇄ Host 内存 | RDMA loopback（QP 连到自身）+ GPUDirect RDMA |
| **B — cudaMemcpy** | GPU 显存 ⇄ PCIe DMA ⇄ Host 内存 | `cudaMemcpyAsync` |

两个方向都测：`H2D`（Host→Device）与 `D2H`（Device→Host）。

## 设计要点

1. **GPU–NIC 亲和性**：程序自动按 PCI 拓扑（`cudaDeviceGetPCIBusId` 的 BDF 与
   `/sys/class/infiniband/<dev>/device` 的 BDF 公共前缀）+ NUMA node 打分，选出离指定
   GPU 最近的 IB 网卡。跨 NUMA / 跨 PCIe switch 会显著拉低 GDR 带宽，因此默认就近选卡，
   也可用 `--nic mlx5_x` 手动指定。
2. **Warm-up**：两种场景在正式计时前都做 `warmup` 次预热——
   GDR 预热建链后的首批传输与 GDR 通路；cudaMemcpy 预热页锁定/PCIe 训练。
3. **计时统一**：两条路径都用 `std::chrono::steady_clock` 量端到端时延，保证可比。
   每个尺寸批量提交 `batch` 个传输、重复 `iters` 次取平均。

## 依赖与构建

- CUDA Toolkit（`nvcc`）
- `rdma-core-devel` / `libibverbs-dev`（编译期需要 `<infiniband/verbs.h>`）
- 运行期需 **GPUDirect RDMA**：加载内核模块 `nvidia-peermem`（新驱动）或 `nv_peer_mem`
  ```bash
  sudo modprobe nvidia-peermem      # 或 nv_peer_mem
  lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
  ```

```bash
cd flexflow/bench
make CUDA_HOME=/usr/local/cuda-11.8 SM_ARCH=80   # A100=80, H100=90, V100=70
```

> 注：当前构建机仅有 `libibverbs.so.1` 运行库、无 `-dev` 头文件，且无 GPU/IB 硬件。
> 请在实际的 GPU + RDMA 网卡节点上安装 `rdma-core-devel` 后编译运行。
> 若只有 `libibverbs.so.1` 没有 `.so` 软链：`make IBVERBS_LIB=/usr/lib64/libibverbs.so.1`

## 运行

```bash
# 单 QP 基线（离 GPU 最近的 1 张卡，RDMA WRITE）
./gdr_vs_memcpy --gpu 0 --nics 1 --qp-per-nic 1 --opcode write --csv result_base.csv

# 多 NIC × 多 QP 并发（测能否逼近 PCIe 上限）
./gdr_vs_memcpy --gpu 0 --nics 4 --qp-per-nic 2 --opcode write --csv result_multi.csv

# RDMA READ 对比
./gdr_vs_memcpy --gpu 0 --nics 4 --qp-per-nic 2 --opcode read  --csv result_read.csv

# 可选参数：
#   --gpu N         选择 GPU（默认 0）
#   --nics K        使用离 GPU 最近的 K 张网卡并发（默认 1，按亲和性排序）
#   --qp-per-nic Q  每张网卡上的 QP 数（默认 1）；总并发 lane = K×Q
#   --opcode write|read  RDMA 操作类型（默认 write）
#   --gid IDX       手动指定 RoCE GID index（默认自动找 RoCEv2）
#   --csv FILE      输出 CSV
```

> 说明：多 lane 模式把每次测量的 `batch` 个传输按 round-robin 分摊到 `K×Q` 条
> lane（不同 NIC/QP）上**并发**投递，再统一收割完成，用以观察 GDR 聚合带宽上限。

输出示例（列：方向 / 尺寸 / GDR 带宽 / GDR 时延 / cudaMemcpy 带宽 / cudaMemcpy 时延 / 带宽比）：

```
dir       size |     GDR GB/s      GDR us | cMemcpy GB/s  cMemcpy us |   A/B bw
--------------------------------------------------------------------------------------
H2D     1.0KB  |        0.512        2.00 |        0.231        4.43 |     2.22
...
H2D   256.0MB  |       23.110       11.08 |       25.640        9.99 |     0.90
D2H   256.0MB  |       22.870       11.20 |       25.310       10.12 |     0.90
```

## 结果解读

- **小包**：GDR 时延受 RDMA 提交/完成开销影响；cudaMemcpy 受 launch 开销影响。
- **大包**：两者都趋近 PCIe 物理带宽上限，GDR 因多一跳 NIC 通常略低于直连 DMA。
- 若 GDR 带宽远低于预期，优先检查：① `nvidia-peermem` 是否加载；
  ② 选中的网卡是否与 GPU 同 NUMA / 同 PCIe switch（看程序打印的 `[affinity]` 行）。
