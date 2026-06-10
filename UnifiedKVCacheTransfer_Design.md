# 统一 KVCache 传输库设计文档

> 目标：设计一个类似 FlexFlow 的 KVCache 传输库，**统一 sglang 的 PD 分离传输与 HiCache 的 KVCache 传输**，支持同机内与跨节点传输，自动适应机内拓扑亲和性，并支持 GPU→SSD 传输。要求 API 在语义上统一且简洁。

---

## 1. 背景与问题

sglang 中存在两套机制完全不同的 KVCache 传输：

| 维度 | PD 分离（Prefill ↔ Decode） | HiCache（L1/L2/L3 分级缓存） |
|---|---|---|
| 寻址 | 基址指针 + 页索引，**dst 指针显式交换** | 内容哈希 key + host_indices，**dst 由 store 解析** |
| 语义 | 点对点 push（WRITE 到指定 peer） | put/get（键值存取） |
| 目的地 | 远端 peer 的 VRAM | 本地 host buffer / 远端存储对象 |
| 同步 | bootstrap 交换指针 + notif 完成 | key 存在性查询，无指针握手 |
| 层级 | GPU→GPU（跨进程/跨节点） | GPU→CPU→SSD/分布式（同进程多级） |
| 底层 | NIXL / Mooncake RDMA WRITE | Mooncake / NIXL / 3FS / file，zero-copy |

**统一的关键洞察**：两者最终都是"把若干 `(源地址, 长度)` 搬到若干 `(目的地址, 长度)`"。
区别只在**目的地址怎么来**——PD 靠预先交换的远端指针，HiCache 靠 store 对 key 的内部放置。
因此把"目的地解析"抽象为可插拔的 **Locator**，即可在同一底层模型下统一两套机制。

---

## 2. 设计原则

1. **单一传输入口**：所有传输归一为 `transfer(src: Locator, dst: Locator)`。
2. **以 Locator 多态统一寻址**：本地页 / 远端页 / 键值 / 文件，差异全部收敛到 Locator 解析。
3. **页索引作为统一寻址单位**：PD 的 `kv_indices` 与 HiCache 的 `host_indices` 本质同构。
4. **注册与传输分离**：注册重量级、一次性；传输轻量级、热路径。
5. **元数据交换不进热路径**：建链（peer 连接 / store 绑定）一次性完成。
6. **拓扑亲和路径选择内置且自动**，通过 `hint` 暴露少量策略开关。
7. **统一异步模型**：所有 `submit` 返回 `TransferHandle`，`poll/wait` 语义一致。
8. **新增传输类型可插拔**：增加一种 Locator + 后端即可，不动上层 API。

---

## 3. 核心抽象

```text
1. MemHandle      —— 一次注册得到的内存区域句柄（统一 VRAM/DRAM/SSD/远端）
2. PageLayout     —— 描述一个分页内存池的几何（页大小、每层 item_len、层数）
3. Locator        —— "一端"的定位方式，多态：
      LocalPages(handle, indices)     本地分页（PD 的 src、HiCache 的 host buffer）
      PeerPages(peer_id, indices)     远端 peer 分页（PD 的 dst）
      StorageKeys(keys)               键值（HiCache 的 L3）
      FileRegion(handle, offsets)     SSD 文件区间（GPU→SSD）
4. TransferHandle —— 异步传输的 future，poll()/wait()
```

**`Locator` 是统一 PD 与 HiCache 的枢纽**：一次传输永远是 `transfer(src, dst)`，
引擎负责把两端各自解析成 `[(addr, len, device)]` 描述符列表，再交给最优后端执行。

---

## 4. API 接口设计

### 4.1 初始化接口

```python
class FlexFlow:
    def __init__(
        self,
        rank: int,
        device: int,                          # 本 rank 的 GPU
        role: str = "both",                   # "peer"(PD) / "cache"(HiCache) / "both"
        backends: list[str] = ("auto",),      # auto 探测: nvlink, rdma, gds, store
        metadata_server: str | None = None,   # etcd/redis/bootstrap 地址；None=纯P2P
        topology: "Topology | None" = None,   # None 则自动发现
        selector_config: "SelectorConfig | None" = None,  # 路径选择器调参；None=默认
    ): ...

    def discover_topology(self) -> "Topology": ...   # NVML+sysfs，亲和性来源
    def update_selector_config(self, cfg: "SelectorConfig"): ...  # 运行时可热更新
```

一个进程一个 `FlexFlow`，内部聚合：NIXL/Mooncake agent + 拓扑 + 路径选择器 + 完成轮询线程。
`role` 决定加载哪些能力，但 API 表面统一。

**`selector_config` —— 初始化阶段可指定路径选择器的打分权重 / QoS 阈值 / SplitPath 策略**
（沿用老 FlexFlow 的 `SelectorConfig` 语义；不传则用默认值）：

```python
from flexflow import FlexFlow, SelectorConfig

cfg = SelectorConfig()
# 打分权重
cfg.w_bandwidth   = 1.0   # 带宽收益权重
cfg.w_queue_delay = 0.5   # 拥塞惩罚权重
cfg.w_hops        = 0.2   # 跳数惩罚权重
cfg.w_qos_risk    = 0.3   # QoS 干扰风险权重
cfg.w_latency     = 0.5   # TTFT 延迟惩罚权重

# QoS 阈值（链路利用率超过则拒绝该路径，保护推理流量）
cfg.nvlink_util_threshold = 0.80
cfg.rdma_util_threshold   = 0.80
cfg.cnic_util_threshold   = 0.70
cfg.snic_util_threshold   = 0.85

# SplitPath 多路拆分
cfg.enable_split_path    = True
cfg.max_split_paths      = 2
cfg.split_benefit_margin = 0.15  # 拆分须比单路快 15% 才采用

eng = FlexFlow(rank=0, device=0, role="both", selector_config=cfg)
# 也可运行时热更新：eng.update_selector_config(cfg)
```

> 设计取舍：`SelectorConfig` 是**引擎级**的全局策略（拓扑亲和、QoS 保护、拆分阈值），
> 在初始化时指定一次即可全程生效；而**单次传输级**的临时策略（如某请求 TTFT 敏感、
> 优先级）则通过 `submit(..., hint=TransferHint(...))` 传入，二者分工清晰。

### 4.2 内存池 / 显存池注册接口

一次性注册大块、返回句柄（RDMA 注册昂贵，少而大为佳）：

```python
def register_pool(
    self,
    tensors: torch.Tensor | list[torch.Tensor],  # KV pool（按层的 K/V buffer）
    layout: PageLayout,                            # 页几何，统一寻址单位
    mem_type: str | None = None,                   # None=从 tensor.device 自动推断
    name: str | None = None,                       # 便于跨端引用
) -> MemHandle: ...

def register_file(self, path: str, size: int) -> MemHandle:   # SSD/GDS
    ...

def deregister(self, handle: MemHandle): ...
```

- `mem_type` 自动推断（参照 NIXL：`device==-1 → DRAM`，否则 `VRAM`；文件 → `FILE`）。
- `PageLayout` 让"页索引"成为统一寻址单位——PD 的 `kv_indices` 和 HiCache 的 `host_indices`
  本质都是页槽位，归一到这里。

### 4.3 端点发现 / 元数据交换接口

统一 PD 的 bootstrap 与 HiCache 的 store 寻址，均为一次性建链，不进数据热路径：

```python
# 点对点：交换 agent 元数据 + 远端 pool 基址（对应 PD 的 KVArgsRegisterInfo）
def connect_peer(self, peer_id: str, endpoint: str) -> "PeerRef": ...
def export_pool_metadata(self, handle: MemHandle) -> bytes: ...   # 发给对端
def import_pool_metadata(self, peer_id: str, meta: bytes): ...    # 收对端的

# 键值：绑定一个 store 命名空间（对应 HiCache 的 storage backend）
def bind_store(self, namespace: str, **store_cfg) -> "StoreRef": ...
```

### 4.4 异步传输接口（核心，统一入口）

```python
def submit(
    self,
    src: Locator,
    dst: Locator,
    *,
    hint: TransferHint = DEFAULT,   # ttft_sensitive / bandwidth_first / 优先级
) -> TransferHandle: ...

def poll(self, h: TransferHandle) -> Status:   # Pending/Done/Failed
    ...
def wait(self, handles: list[TransferHandle], timeout=None): ...
```

`submit` 内部流程（统一逻辑的落点）：

1. 解析 `src`、`dst` → 各得一个 `[(addr, len, dev)]` 描述符列表
   （KV-store 的 dst 在此步做 key→放置解析 + 存在性检查）。
2. `group_concurrent_contiguous` 合并连续页（沿用 sglang 的优化）。
3. **PathSelector 按拓扑亲和性选后端/路由**（NVLink / PCIe / RDMA / GDS / relay）。
4. 提交到对应后端（NIXL `initialize_xfer`+`transfer` / Mooncake `batch_transfer` / GDS）。
5. 注册 `notif`，由后台线程驱动完成状态。

### 4.5 语义糖（薄封装，不引入新机制）

```python
# HiCache 风格（键值），内部就是 submit(LocalPages, StorageKeys)
def put(self, keys, src: Locator, hint=...) -> TransferHandle: ...
def get(self, keys, dst: Locator, hint=...) -> TransferHandle: ...
def exists(self, keys) -> list[bool]: ...

# PD 风格（点对点 push），内部就是 submit(LocalPages, PeerPages)
def push(self, peer: PeerRef, src: Locator, dst: Locator, room=None) -> TransferHandle: ...
```

---

## 5. 四条路径如何映射到同一 API

### ① PD 同机（NVLink）

```python
h = eng.push(peer, src=LocalPages(prefill_pool, kv_idx),
                   dst=PeerPages("decode0", dst_idx))
# PathSelector 探测 src/dst 同节点 → 选 NVLink/P2P 后端
```

### ② PD 跨节点（RDMA）

```python
# 同一行代码，PathSelector 发现跨节点 → 选 RDMA WRITE
h = eng.push(peer, src=LocalPages(prefill_pool, kv_idx),
                   dst=PeerPages("decode_remote", dst_idx))
```

### ③ HiCache L2 ↔ L3（host ↔ 分布式存储）

```python
eng.put(keys=page_hashes, src=LocalPages(host_pool, host_idx))   # backup
eng.get(keys=page_hashes, dst=LocalPages(host_pool, host_idx))   # prefetch
```

### ④ GPU → SSD（GDS）

```python
ssd = eng.register_file("/mnt/nvme/kv.bin", size)
h = eng.submit(src=LocalPages(gpu_pool, kv_idx),
               dst=FileRegion(ssd, offsets))
# PathSelector 见 VRAM→FILE → 选 GPUDirect Storage 后端
```

**统一性体现**：用户永远只写 `src Locator → dst Locator`，路径/后端/亲和性全部由引擎决定。

---

## 6. 拓扑亲和性层

借鉴 FlexFlow 的 PathSelector，但内置化：

```python
class PathSelector:
    def select(self, src_descs, dst_descs, hint) -> Route:
        # 1. 由 src/dst 的 (device, node) 判定关系:
        #    同卡 / 同节点NVLink / 同节点PCIe / 跨节点RDMA / 存储
        # 2. 查 NodeAgent 实时链路利用率(NVLink/PCIe/RDMA)
        # 3. 打分: 带宽收益 - 拥塞惩罚 - 跳数 - QoS风险 (- TTFT惩罚)
        # 4. 拥塞时可选中继GPU / 拆分多路(SplitPath)
        # 返回 Route(backend, hops, links)
```

关键：**亲和性判断的输入来自 `register_pool` 时记录的 `(device, numa, nic)` 和拓扑图**。
`hint.ttft_sensitive=True` 时惩罚多跳（PD 首 token 延迟敏感场景直接受益）。

PathSelector 的行为由初始化时传入的 `SelectorConfig` 驱动（见 4.1）：

```python
@dataclass
class SelectorConfig:
    # 打分权重（score = w_bandwidth*带宽收益 - w_queue_delay*拥塞 - w_hops*跳数
    #                  - w_qos_risk*QoS风险 - w_latency*TTFT延迟）
    w_bandwidth: float = 1.0
    w_queue_delay: float = 0.5
    w_hops: float = 0.2
    w_qos_risk: float = 0.3
    w_latency: float = 0.5
    # QoS 阈值（链路利用率超过则拒绝该路径，保护推理流量）
    nvlink_util_threshold: float = 0.80
    rdma_util_threshold: float = 0.80
    cnic_util_threshold: float = 0.70
    snic_util_threshold: float = 0.85
    # SplitPath 多路拆分
    enable_split_path: bool = True
    max_split_paths: int = 2
    split_benefit_margin: float = 0.15   # 拆分须比单路快 15% 才采用
    relay_benefit_margin: float = 0.10   # 中继须比直连快 10% 才采用
```

> `SelectorConfig` 是引擎级全局策略，初始化时 `FlexFlow(..., selector_config=cfg)` 指定，
> 或运行时 `update_selector_config()` 热更新；单请求级策略走 `TransferHint`。

---

## 7. 完整 Python 接口草图

```python
# ---------- 类型 ----------
class MemType(Enum): VRAM; DRAM; HOST_PINNED; FILE; REMOTE_VRAM

@dataclass
class PageLayout:
    page_size: int
    item_lens: list[int]      # 每层每 token 字节数
    num_layers: int
    is_mla: bool = False

class Locator: ...
@dataclass
class LocalPages(Locator):  handle: MemHandle; indices: np.ndarray
@dataclass
class PeerPages(Locator):   peer_id: str;     indices: np.ndarray
@dataclass
class StorageKeys(Locator): keys: list[str]
@dataclass
class FileRegion(Locator):  handle: MemHandle; offsets: np.ndarray

@dataclass
class TransferHint:
    ttft_sensitive: bool = False
    priority: int = 0
    allow_relay: bool = True
    allow_split: bool = True

@dataclass
class SelectorConfig:
    # 打分权重
    w_bandwidth: float = 1.0
    w_queue_delay: float = 0.5
    w_hops: float = 0.2
    w_qos_risk: float = 0.3
    w_latency: float = 0.5
    # QoS 阈值（保护推理流量）
    nvlink_util_threshold: float = 0.80
    rdma_util_threshold: float = 0.80
    cnic_util_threshold: float = 0.70
    snic_util_threshold: float = 0.85
    # SplitPath / Relay
    enable_split_path: bool = True
    max_split_paths: int = 2
    split_benefit_margin: float = 0.15
    relay_benefit_margin: float = 0.10

# ---------- 引擎 ----------
class FlexFlow:
    def __init__(self, rank, device, role="both", backends=("auto",),
                 metadata_server=None, topology=None,
                 selector_config: SelectorConfig | None = None): ...
    def update_selector_config(self, cfg: SelectorConfig): ...   # 运行时热更新
    # 注册
    def register_pool(self, tensors, layout, mem_type=None, name=None) -> MemHandle: ...
    def register_file(self, path, size) -> MemHandle: ...
    def deregister(self, handle): ...
    # 建链
    def connect_peer(self, peer_id, endpoint) -> PeerRef: ...
    def export_pool_metadata(self, handle) -> bytes: ...
    def import_pool_metadata(self, peer_id, meta): ...
    def bind_store(self, namespace, **cfg) -> StoreRef: ...
    # 传输（统一入口 + 语义糖）
    def submit(self, src: Locator, dst: Locator, *, hint=TransferHint()) -> TransferHandle: ...
    def push(self, peer, src, dst, room=None, hint=TransferHint()) -> TransferHandle: ...
    def put(self, keys, src, hint=TransferHint()) -> TransferHandle: ...
    def get(self, keys, dst, hint=TransferHint()) -> TransferHandle: ...
    def exists(self, keys) -> list[bool]: ...
    # 完成
    def poll(self, h) -> Status: ...
    def wait(self, hs, timeout=None): ...
    def shutdown(self): ...
```

---

## 8. 关键设计决策与理由

1. **以 Locator 多态统一寻址**，而非给 PD 和 HiCache 各开一套 API——这是语义统一的核心。
   新增传输类型（如 CXL、对象存储）只需加一种 Locator + 后端，不动上层。
2. **页索引作为统一寻址单位**（`PageLayout` + `indices`）——PD 的 `kv_indices` 和
   HiCache 的 `host_indices` 本就同构，统一后 `group_concurrent_contiguous` 合并优化对两者都生效。
3. **注册与传输分离**：`register_pool` 一次性、重量级；`submit` 轻量、热路径。
   符合 RDMA 注册昂贵的现实（NIXL/Mooncake 都这么做）。
4. **元数据交换不进热路径**：`connect_peer`/`bind_store` 一次完成，
   对应 PD 的 bootstrap 和 HiCache 的 store 绑定。
5. **路径选择内置且自动**，但用 `hint` 暴露策略钮（TTFT 敏感、优先级）——既简洁又可调。
6. **统一异步模型**：所有 `submit` 返回 `TransferHandle`，`poll/wait` 统一——
   上层调度器（sglang scheduler）只需对接一种完成语义。
7. **完成同步两种实现，一个接口**：点对点用 `notif`+轮询（PD），键值用返回码（HiCache），
   但 `poll()` 对外一致。
8. **GPU→SSD 自然纳入**：SSD 只是 `MemType.FILE` 的一个注册区域，
   `VRAM→FILE` 触发 GDS 后端，零特殊路径。

---

## 9. 总结

设计的核心是 **`transfer(src: Locator, dst: Locator)` 这一个统一入口** +
**可插拔的 Locator（本地页 / 远端页 / 键值 / 文件）** +
**一次性的 register/connect/bind 建链** + **内置的拓扑亲和路径选择器**。

- PD 分离 = `push(LocalPages → PeerPages)`
- HiCache = `put/get(LocalPages ↔ StorageKeys)`
- GPU→SSD = `submit(LocalPages → FileRegion)`

三者复用同一套注册、描述符生成、路径选择、异步完成机制，对外语义统一且简洁，
对内通过 Locator 解析和 PathSelector 吸收所有差异。

---

## 10. 典型场景接口使用示例

> 以下示例均基于第 4/7 节定义的 API。代码以"可读优先"为目标，省略了部分异常处理。

### 10.1 PD 分离：Prefill 端（发送方）

Prefill 实例启动时一次性注册整个 KV pool，建链后对每个请求把 KV 推送到 decode。

```python
import numpy as np
from flexflow import FlexFlow, PageLayout, LocalPages, PeerPages, TransferHint

# ---------- 1. 初始化（每进程一个引擎）----------
eng = FlexFlow(
    rank=0,
    device=0,
    role="peer",                                   # PD 点对点角色
    backends=("auto",),                            # 自动探测 nvlink / rdma
    metadata_server="http://bootstrap:8998",       # PD 的 bootstrap server
)

# ---------- 2. 注册 KV 显存池（一次性，重量级）----------
# kv_buffers: 模型每层的 K/V buffer（来自 token_to_kv_pool.get_contiguous_buf_infos）
layout = PageLayout(
    page_size=16,
    item_lens=kv_item_lens,        # 每层每 token 字节数
    num_layers=num_layers,
    is_mla=False,
)
prefill_pool = eng.register_pool(kv_buffers, layout, name="prefill_kv")

# ---------- 3. 建链：把本端 pool 元数据导出，导入 decode 端的元数据 ----------
# （metadata 通过 bootstrap server / ZMQ 交换，这里示意）
my_meta = eng.export_pool_metadata(prefill_pool)
send_to_bootstrap(my_meta)                         # 发布给 decode

# ---------- 4. 每个请求：把 KV 推送到 decode ----------
def on_prefill_done(room, peer_id, prefill_kv_idx, decode_kv_idx):
    """room: 请求唯一标识；*_kv_idx: 双方各自的页槽位。"""
    peer = eng.connect_peer(peer_id, endpoint=decode_endpoint)  # 已缓存则复用

    handle = eng.push(
        peer,
        src=LocalPages(prefill_pool, prefill_kv_idx),  # 本端 KV 页
        dst=PeerPages(peer_id, decode_kv_idx),         # 远端 decode 的页槽位
        room=room,
        hint=TransferHint(ttft_sensitive=True),        # 首 token 敏感 → 偏好低跳数
    )
    return handle

# ---------- 5. 轮询完成（在调度循环里）----------
def poll_loop(inflight: dict):
    for room, h in list(inflight.items()):
        st = eng.poll(h)
        if st == "Done":
            release_kv_pages(room)                     # 可释放 prefill 侧页
            del inflight[room]
        elif st == "Failed":
            handle_failure(room)
            del inflight[room]
```

### 10.2 PD 分离：Decode 端（接收方）

Decode 端同样注册自己的 KV pool，并把基址/元数据上报给 prefill；之后被动接收。

```python
eng = FlexFlow(rank=0, device=0, role="peer",
                 metadata_server="http://bootstrap:8998")

# 注册 decode 自己的 KV 显存池
decode_pool = eng.register_pool(decode_kv_buffers, layout, name="decode_kv")

# 上报：把本端 pool 基址 + agent 元数据发给 prefill（对应 PD 的 _register_kv_args）
my_meta = eng.export_pool_metadata(decode_pool)
register_to_prefill_via_bootstrap(
    room="None",                       # 一次性注册，非单请求
    agent_meta=my_meta,
    dst_kv_ptrs=decode_pool.base_ptrs, # prefill 将据此计算 dst 地址
)

# 每请求：分配 dst 页槽位，把 dst_kv_indices 发给 prefill，然后等待
def on_new_request(room, num_pages):
    decode_kv_idx = alloc_kv_pages(num_pages)
    notify_prefill(room, dst_kv_indices=decode_kv_idx)   # prefill 据此 push

    # 用一个"接收句柄"轮询该 room 是否传完（由 notif 驱动）
    return eng.wait_room(room)          # 或在调度循环里 poll
```

> 方向说明：PD 分离是 **prefill 主动 WRITE push 到 decode**。Decode 只需注册内存、
> 上报指针、分配 dst 页并轮询完成，数据搬运由 prefill 发起。

### 10.3 HiCache 场景一：Write-back（L1 GPU → L2 host）

把 GPU 显存里的 KV 页写回到 CPU host buffer（即 HiCache 驱逐/卸载时的 L1→L2 搬运）。
源和目的都是本进程已注册的内存池，只是 `mem_type` 不同（VRAM → HOST_PINNED），
引擎的 PathSelector 识别出 `VRAM→DRAM` 后自动选择本地 DMA 拷贝（PCIe / NVLink-C2C）。

```python
import numpy as np
from flexflow import FlexFlow, PageLayout, LocalPages, TransferHint

eng = FlexFlow(rank=0, device=0, role="cache")

# 注册 L1 GPU KV 显存池（mem_type 自动推断为 VRAM）
gpu_pool = eng.register_pool(gpu_kv_buffers, layout, name="gpu_kv")

# 注册 L2 host buffer（pinned memory，mem_type 自动推断为 HOST_PINNED）
host_pool = eng.register_pool(host_kv_buffer, layout, name="host_kv")

# 写回：把 GPU 页槽位的数据搬到 host 页槽位（device-to-host）
def write_back(gpu_kv_idx, host_idx):
    handle = eng.submit(
        src=LocalPages(gpu_pool, gpu_kv_idx),     # L1 GPU 页
        dst=LocalPages(host_pool, host_idx),      # L2 host 页
        hint=TransferHint(priority=0),            # 后台驱逐，低优先级
        # PathSelector 见 VRAM→HOST_PINNED，自动选本地 DMA（cudaMemcpyAsync）
    )
    return handle

# 反向：L2 host → L1 GPU（命中后回填，host-to-device），方向对调即可
def load_back(host_idx, gpu_kv_idx):
    return eng.submit(
        src=LocalPages(host_pool, host_idx),
        dst=LocalPages(gpu_pool, gpu_kv_idx),
        hint=TransferHint(ttft_sensitive=True),   # 回填在关键路径上
    )
```

> 说明：L1↔L2 是同进程内的本地传输，引擎用本地 DMA 后端实现，完成语义与跨节点
> 传输一致（同样返回 `TransferHandle`，用 `poll`/`wait` 等待）。这样 HiCache 的
> L1↔L2↔L3 三级搬运可以全部走同一套 API，无需上层再单独维护 CUDA stream 拷贝逻辑。

### 10.4 HiCache 场景二：Backup（L2 host → L3 分布式存储，写）

把 host buffer 里的 KV 页按内容哈希 key 写入分布式存储。

```python
eng = FlexFlow(rank=0, device=0, role="cache")

# 注册 L2 host buffer（pinned memory），mem_type 自动推断为 DRAM/HOST_PINNED
host_pool = eng.register_pool(host_kv_buffer, layout, name="host_kv")

# 绑定 L3 存储命名空间（Mooncake / NIXL / 3FS / file 后端）
store = eng.bind_store("kvcache", backend="mooncake",
                       metadata_server="etcd://meta:2379")

# 写：keys = 每页的 prefix 内容哈希；host_idx = 这些页在 host pool 的槽位
def backup(page_hashes, host_idx):
    handle = eng.put(
        keys=page_hashes,
        src=LocalPages(host_pool, host_idx),
        hint=TransferHint(priority=0),     # 后台写，低优先级
    )
    return handle

# 写前去重由引擎内部 exists 处理；也可显式查询
def backup_dedup(page_hashes, host_idx):
    present = eng.exists(page_hashes)                  # [bool]
    todo = [(k, i) for k, i, p in zip(page_hashes, host_idx, present) if not p]
    if not todo:
        return None
    keys, idx = zip(*todo)
    return eng.put(keys=list(keys), src=LocalPages(host_pool, np.array(idx)))
```

### 10.5 HiCache 场景三：Prefetch（L3 → L2 host，读）

命中查询后，把存储里的 KV 页读回 host buffer。

```python
def prefetch(page_hashes, host_idx):
    # 1. 查哪些 key 在存储里命中
    present = eng.exists(page_hashes)
    hit = [(k, i) for k, i, p in zip(page_hashes, host_idx, present) if p]
    if not hit:
        return None, []
    keys, idx = zip(*hit)

    # 2. 读回 host pool 对应页槽位
    handle = eng.get(
        keys=list(keys),
        dst=LocalPages(host_pool, np.array(idx)),
        hint=TransferHint(ttft_sensitive=True),   # 命中预取在关键路径上
    )
    return handle, list(keys)

# 3. 完成后，host→GPU 的 L2→L1 回填见 10.3 的 load_back（同样走本库 submit）
```

### 10.6 HiCache 场景四：GPU → SSD offload（GPUDirect Storage）

显存 KV 直接落盘到本地 NVMe，绕过 CPU。

```python
eng = FlexFlow(rank=0, device=0, role="cache", backends=("auto", "gds"))

gpu_pool = eng.register_pool(gpu_kv_buffers, layout, name="gpu_kv")
ssd = eng.register_file("/mnt/nvme/kv_offload.bin", size=cap_bytes)

def offload_to_ssd(gpu_kv_idx, file_offsets):
    handle = eng.submit(
        src=LocalPages(gpu_pool, gpu_kv_idx),
        dst=FileRegion(ssd, file_offsets),
        # PathSelector 见 VRAM→FILE，自动选 GPUDirect Storage 后端
    )
    return handle

def load_from_ssd(file_offsets, gpu_kv_idx):
    return eng.submit(
        src=FileRegion(ssd, file_offsets),
        dst=LocalPages(gpu_pool, gpu_kv_idx),
    )
```

### 10.7 批量提交与统一等待

无论 PD 还是 HiCache，完成语义一致，可混合批量等待：

```python
handles = []
handles.append(eng.push(peer, LocalPages(prefill_pool, idx_a), PeerPages("d0", didx_a)))
handles.append(eng.put(keys=hashes, src=LocalPages(host_pool, hidx)))
handles.append(eng.submit(LocalPages(gpu_pool, gidx), FileRegion(ssd, offs)))

# 统一等待（可设超时）
eng.wait(handles, timeout=5.0)

# 或逐个非阻塞轮询
for h in handles:
    st = eng.poll(h)   # Pending / Done / Failed
```

---

### 示例小结

| 场景 | 调用 | src Locator | dst Locator |
|---|---|---|---|
| PD 同机/跨节点 | `push()` | `LocalPages` | `PeerPages` |
| HiCache L1→L2 写回 | `submit()` | `LocalPages`(VRAM) | `LocalPages`(HOST_PINNED) |
| HiCache L2→L1 回填 | `submit()` | `LocalPages`(HOST_PINNED) | `LocalPages`(VRAM) |
| HiCache backup | `put()` | `LocalPages` | `StorageKeys`（隐式） |
| HiCache prefetch | `get()` | `StorageKeys`（隐式） | `LocalPages` |
| GPU→SSD offload | `submit()` | `LocalPages` | `FileRegion` |
| SSD→GPU load | `submit()` | `FileRegion` | `LocalPages` |

所有场景共享：`register_pool` 注册 → 建链（`connect_peer`/`bind_store`/`register_file`）→
异步提交（返回 `TransferHandle`）→ `poll`/`wait` 统一完成。
