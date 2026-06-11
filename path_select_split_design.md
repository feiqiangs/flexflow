# 基于 CoDel 排队信号的 flexFlow 选路策略（设计文档 v1.6）

> 本文整理自多轮讨论结论，目标：用「排队动力学信号」作为拥塞判定信号，解决时间尺度错配问题，并给出可自校准、抗抖动的完整选路算法。
>
> **本版定位（v1.6）**：在 v1.5 基础上，**明确 SplitPath 任务切分策略**。依据：SGLang 的 PD 分离 / hicache KVCache 传输本质是一系列**显存/内存切片**（`transfer_blocks` = 一批 `(src_addr, dst_addr, length)` 三元组，由 `group_concurrent_contiguous` 按连续索引合成段），天然适合按链路可用带宽做**带宽加权切分**。全程**不依赖任何 NVML 机制数据**。

---

## 0. 设计动机（为什么这样设计）

拥塞判定不采用「利用率采样 + 趋势外推」的预判方式，原因：

- **时间尺度错配**：拥塞源是亚毫秒 burst，~100ms 采样会把 burst 平均掉，无法反映瞬时撞车风险。
- **指标性质错误**：利用率是拥塞的「果」且会饱和；排队是「因」且领先。
- **决策快但数据旧**：决策仅 ~15μs，而采样喂入的链路状态可旧达 ~100ms。
- **采集有副作用 + 外部依赖**：高频进驱动采样有锁竞争，且是带外、跨进程、依赖驱动版本/权限的旁路数据源，引入额外故障面与运维复杂度。

**核心思路**：被动观测自己的 relay 队列排队动力学 + 传输结果反馈，用 CoDel（Controlled Delay）式判据做拥塞判定。判定与校准所需信息（sojourn、T_xfer、actual 吞吐）全部来自 relay 自身的 enqueue/dequeue/complete 事件，属「自观测闭环」，**完全不依赖任何 NVML 机制数据**。

---

## 1. 信号定义：sojourn time（驻留时延）

每个 relay 传输任务进出队列时打两个时间戳：

```
sojourn = dequeue_ts − enqueue_ts        // 任务在队列里排队等待的时间
```

- 计时函数：`clock_gettime(CLOCK_MONOTONIC)` / `std::chrono::steady_clock`（经 vDSO，单次 ~30ns，分辨率 ~ns）。
- **不用** `gettimeofday`（μs 分辨率不够）；**慎用裸 rdtsc**（跨核/变频坑）。
- sojourn 反映「积压」：链路被 AllReduce 抢占 → relay 服务变慢 → 队列堆积 → sojourn 自然变大。**burst 影响被队列自动捕获，无需采样**。

---

## 2. CoDel 拥塞判定核心

### 2.1 两个参数（注意：不是队列长度阈值）

| 参数 | 含义 | 量级（NVLink relay） | 来源 |
|---|---|---|---|
| **target** | 可容忍的排队驻留时延上限 | 几十 μs ~ 1ms | 实测单任务传输时延 ×1.5~2 |
| **interval** | 确认拥塞所需的持续观察窗口 | 几十 ms（≈1 decode step 周期） | SGLang decode step 周期 ×1.5~2 |

**关键约束**：`target << interval`（单传输尺度 << 干扰周期尺度），否则丧失「区分单次 burst 与持续拥塞」的能力。

### 2.2 判据（min-sojourn 持续超标）

```
在滑动窗口内追踪所有出队任务的 min(sojourn)：
  if min_sojourn ≥ target 持续 ≥ interval:
      标记该 link 为 CONGESTED
  else:
      标记为 OK
```

- **取 min 而非 avg/单次**：OS 调度抖动、中断等噪声**单向偏大**（只增不减）；只要 interval 内有一个任务顺利通过（min 低），即说明链路仍可疏通，不算拥塞。**min 操作天然滤掉 μs 级系统噪声**。
- 前提：interval 内样本足够多（NVLink relay 高频任务可满足，~30ms 内有几十~上百样本）。

### 2.3 为什么不用裸计数阈值

队列长度阈值对 **block 大小敏感**（10 个大 block ≠ 100 个小 block）；CoDel 用时延，物理含义普适、跨硬件鲁棒。

---

## 3. 参数自校准（动态生成，免人工调参）

两个参数都不写死，运行时自动学习；冷启动用经典值兜底。

### 3.1 target 自校准

```
记录每个 relay 任务的实际传输时延 T_xfer（dequeue → 完成）
维护 T_xfer 的滑动 P10（代表「最顺畅时的传输速度」）
target = k × P10(T_xfer)，  k ∈ [1.5, 2.0]
冷启动兜底：target = 1ms
```

推导示例：70B / TP=8 / block=64tokens ≈ 2MB，NVLink 有效 ~150GB/s → T_xfer ≈ 13μs → target ≈ 26μs。
（注意：**绝不能套用 CoDel 网络场景的 5ms**，机内 NVLink 小 2~3 个量级。）

### 3.2 interval 自校准

```
方式A（首选）：从 SGLang 指标推：step周期 ≈ batch_size / decode_throughput
方式B（纯排队信号，无需 NVML）：对 relay 队列的 sojourn 序列做自相关，取 burst 主周期
方式C：CoDel 原版动态收缩 interval / sqrt(count)（拥塞越持续响应越快）
interval = 1.5~2 × decode_step_period
冷启动兜底：interval = 100ms
```

### 3.3 计时噪声地板保护

若实测 `T_xfer < 5μs`（接近计时+调度噪声地板），**不在单 block 粒度判定**，改用**一批 block 的聚合 sojourn**，把被测时间尺度抬到几十 μs 以上。

---

## 4. 防抖动：滞回 + 冷却

「超阈值就切、降回就回切」会导致路径震荡（flapping：切走→队列泄空→判不拥塞→切回→又拥塞）。必须加：

- **双阈值滞回**：进入 CONGESTED 用 target；退出用 `target × 0.5`（退出更严格）。
- **冷却期（hold-down timer）**：切路后 N×interval 内不回切。
- **relay_benefit_margin=10%**：中转须比直连好 10% 才切，作为同类防抖。

---

## 4.5 NVLink relay pacing（link 级硬限速）

### 4.5.1 为什么需要 pacing

NVLink 上的流量优先级是**倒置**的：SGLang 的 TP AllReduce / DeepEP 的 MoE all-to-all 是**推理关键路径**（直接决定 TTFT/TPOT），而 FlexFlow 的 KVCache relay 是**可延迟的后台搬运**。后台流量绝不能抢前台关键路径的带宽。

§2 的 CoDel 是**被动观测 + 事后降级**（撞了车再让）；pacing 是**主动限速 + 事前预防**（从源头不开快车）。两者**互补**：

- **pacing（硬限制，link 级）**：给 NVLink relay 设带宽天花板，从源头限制后台流量规模，保证推理流量永远有余量。
- **CoDel（软降级，path 级）**：pacing 天花板下仍发生排队（推理流量突增）时，再触发选路降级。

这是**速率层 + 选路层**的双层防护，比单靠 CoDel 稳得多。

### 4.5.2 软系数与硬限速拆分

`link_type_bias` （§5.2 Score 静态偏好项，软）与 pacing 限速（硬）是两种不同性质的东西，**是两个独立参数**，不用同一值兼任：

| 名称 | 性质 | 作用 |
|---|---|---|
| `link_type_bias(link_type)` | 软（Score 静态偏好项） | 选路打分时对链路类型的先验偏好 |
| `pacing_rate(link)` | **硬**（速率上限 Bps） | NVLink relay 在该 link 上的带宽天花板 |

### 4.5.3 CoDel-sojourn 驱动的 AIMD pacing 算法

静态限速（如固定 30% NVLink）会同时犯两个错：推理空闲时浪费带宽、推理满载时 30% 仍可能太多。所以 pacing_rate **自适应**，复用已有 sojourn 信号做反馈：

```
relay 在每条 NVLink link 上维护一个令牌桶 / leaky bucket，速率 = pacing_rate

闭环调节（复用 §2 的 min_sojourn 信号，无需新增 NVML 采样）：
  if min_sojourn < target:           # 推理流量未受影响，relay 可多用
      pacing_rate += α               # 加性增（缓慢试探）
  if min_sojourn ≥ target:           # 出现排队 = 开始抢推理流量
      pacing_rate *= β  (β<1)        # 乘性减（快速退让）

  clamp 到 [pacing_rate_min, pacing_rate_max]
  pacing_rate_max = NVLink 理论带宽 × X%   # 防失控
  pacing_rate_min = 保底搬运速率               # 防饥死（避免 KVCache 永远搬不完反卡 prefill）
```

要点：
- **退让要快、试探要慢**（AIMD 非对称）：推理流量一受影响 relay 立即让路。退让响应必须 < 1 个 decode step（应对 DeepEP all-to-all 突发）。
- **信号复用**：调速用的就是 §2 的 `min_sojourn / target`，不引入任何新数据源。
- **饥死保护**：`pacing_rate_min` 保证 relay 不被压到 0，避免反向死锁。
- **冷启动**：`pacing_rate` 初值从保守值起，宁可慢慢 AIMD 爬上去。
- **粒度**：建议 **link 级令牌桶 + relay 子系统全局上限** 两层。

### 4.5.4 pacing 与 CoDel 为何不重复

两者同用 `min_sojourn` 信号但**作用在不同执行层**：pacing 是 link 级**速率控制**（调多快发），CoDel 降级是 path 级**路径选择**（换不换路）。这与之前 QoSRisk/SojournPenalty 同信号同层的「重复计分」不同，是合理的职责分工。

---

## 5. 完整选路算法

选路分三阶：**候选枚举 → 硬约束过滤 → Score 排序选优**。

**Score 在选路中的作用**：Score 是「在通过硬约束的 viable 候选中选最优」的**排序键**。职责划分清晰且不可互代——
- **硬约束过滤（[4]）** 回答「**能不能用**」（可行性）：被判 CONGESTED / 显存不足的路径直接出局，根本不进 Score 计算。
- **Score（[3]）** 回答「**哪个更好**」（优选）：只在 viable 候选间做相对排序，取最高分为 winner。
- winner 还要再过 **margin 门槛（[6]）** 与 **hold-down（[8]）** 才真正生效。

### 5.1 主流程伪代码

```
输入：TransferRequest(src, dst, bytes, ttft_sensitive)
输出：chosen_path（单路，或 SplitPath 多路）

select_path(req):

  # —— 阶段 A：候选枚举 ——
  [1] candidates = enumerate_paths(req.src, req.dst)      # 带 (src,dst) 缓存
        # 候选类型：Direct / NVLinkRelay / CrossNodeRelay / StorageRelay
        # best_direct = candidates 中的 Direct 路径（永远保留作兜底）

  # —— 阶段 B：查信号 + 硬约束过滤（决定“能不能用”）——
  viable = []
  for path in candidates:
      [2] for link in path.links:                        # 事件驱动查 CoDel 检测器，无采样
              link.congested   = CodelDetector[link].is_congested()
              link.min_sojourn = CodelDetector[link].current_min_sojourn()
              link.target      = CodelDetector[link].target()

      [4] # violates_constraints：任一硬约束不满足则出局
          if any(link.congested for link in path.links):       # CoDel 判拥塞
              continue
          if path.has_relay and relay_free_mem(path) < min_relay_memory:
              continue                                          # 中转 GPU 显存不足
          if path.uses_nvlink and pacing_rate(path) <= pacing_rate_min:
              continue                                          # pacing 被压到地板（推理流量已挤满）
          # 交叉验证仅用吞吐反馈 EWMA（report_transfer_result 回灌），不引入 NVML
      viable.append(path)

  [5] if viable is empty:                                 # 全被过滤 → direct 兜底
          return best_direct                              # direct 永不被淘汰

  # —— 阶段 C：Score 排序选优（决定“哪个更好”）——
  [3] for path in viable:                                 # 各项已归一化到 [0,1]，详见 §5.2
          # NVLink relay 路径的可用带宽受 pacing 天花板限制
          path.score = w1 * BW_avail(path)               # NVLink relay: min(EWMA吞吐, pacing_rate)/BW_ref
                     - w2 * SojournPenalty(path)          # Σ_link max(0, min_sojourn - target)/target
                     - w3 * Hops(path)
                     - w5 * LatencyPenalty(path, req.ttft_sensitive)
                     + b  * link_type_bias(path)          # 静态链路类型偏好（可选，不依赖 sojourn）
          # 可选：relay 持续打满 pacing 上限 → 加饱和惩罚，引导 SplitPath 分流

      winner = argmax(path.score for path in viable)      # ← Score 在这里决定选哪条

  [6] # 保守回退：winner 若为中转，须明显优于 best_direct 才切
      if winner.has_relay and winner.score < best_direct.score * (1 + relay_benefit_margin):
          winner = best_direct                            # 收益不足 margin → 退回 direct

  [8] # 冷却/滞回：winner 在 hold-down 内刚切换过 → 维持现状（防 flapping）
      if within_hold_down(winner):
          winner = current_path

  [7] # SplitPath：多路并行（细节见 §5.5）
      #   仅当：≥2 条 link-disjoint viable 路径 且 block 数 ≥ 路径数 且 总字节 ≥ split_threshold
      if can_split(viable) and total_bytes(req) >= split_threshold:
          # 按可用带宽加权把 block_list 分配到多路（目标：最小化 makespan）
          split_plan = split_blocks_by_bandwidth(req.block_list, viable)
          if est_makespan(split_plan) < winner.est_time * (1 - split_margin):
              return build_split_path(split_plan)         # 收益足够才分流

  return winner
```

> 要点：**Score 高 ≠ 一定被选**。它是「优选建议」，最终还受 [5] 兜底、[6] margin、[8] hold-down 三道闸约束。被硬约束判出局的路径连 Score 都不会算。

### 5.2 打分项定义与权重标定

§5.1 的 Score 本质是**多目标加权**。前提：所有项必须先**归一化到 [0,1]**（无量纲），否则带宽（GB/s）、时延（μs）、跳数（整数）量纲不同，权重失去可比意义。所有输入数据均来自 relay 自观测（吞吐反馈 EWMA / 静态拓扑配置），**不依赖 NVML**。

**① BW_avail（可用带宽，收益项，越大越好）**
```
一般路径：BW_raw = min over links ( link_capacity − EWMA_used_bw )      // 路径瓶颈可用带宽
NVLink relay：BW_raw = min( EWMA_used_bw_observed, current_pacing_rate ) // 受 pacing 天花板限制
BW_avail = BW_raw / BW_ref                                            // BW_ref = 本机最快链路理论带宽
```
- 对 NVLink relay：pacing 压得越狠，这条路 Score 越低，选路自然倾向别的路径——pacing 结果就是这样反馈进选路的。

**② SojournPenalty（拥塞惩罚项）**
```
SojournPenalty = Σ_link max(0, min_sojourn − target) / target       // 已除以 target，天然归一化
```

**③ Hops（跳数，成本项，越少越好）**
```
Hops_norm = (n_hops − 1) / (max_hops − 1)                           // direct=0，归一到 [0,1]
```

> **为何删除原 QoSRisk 项（v1.4 → v1.5）**：原 QoSRisk = Σ congestion_prob · qos_weight，其中 `congestion_prob` 与 SojournPenalty 同源于 `min_sojourn`。两项同信号、同在 Score 里相加 → 同一拥塞事实被罚两次（重复计分）。且 v1.4 引入 pacing 后，「NVLink 抢占风险」已由 link 级硬限速从源头接管，QoSRisk 软项使命重叠。故删除。若仍需表达「链路类型的先验偏好」，改用下方静态 `link_type_bias`。

**⑤ link_type_bias（链路类型静态偏好，可选）**
```
link_type_bias(path) = avg_link type_pref(link_type)                 // 静态配置，不依赖 sojourn
```
- 纯静态常数（如机内 NVLink 优于跨节点 RDMA 优于 StorageRelay），**与拥塞信号正交**，不与 SojournPenalty 重复。若不需要可令 b=0 去掉。

**⑤ LatencyPenalty(ttft)（首 token 延迟惩罚，条件项）**
```
base_latency        = Σ_link ( prop_delay + serialize_delay )
LatencyPenalty(ttft)= ttft_sensitive ? (base_latency / latency_budget) : 0
```

**权重标定（三段式）**：

1. **量纲归一化**：各项全压到 [0,1]，权重才可比。
2. **业务优先级初值**（约束 `Σwᵢ = 1`；QoSRisk 删除后原 w4=0.15 并入 w1/w2）：

| 权重 | 对应项 | 建议初值 | 理由 |
|---|---|---|---|
| w1 | BW_avail | 0.45 | 主收益；原 0.35 + 并入部分原 w4 |
| w2 | SojournPenalty | 0.35 | 核心拥塞信号；原 0.30 + 并入部分原 w4 |
| w3 | Hops | 0.10 | 次要成本，已有 relay_benefit_margin 兜底 |
| w5 | LatencyPenalty | 0.10 | 条件生效，仅 ttft 敏感时起作用 |
| b | link_type_bias（可选） | 0～0.05 | 静态偏好，默认可为 0；不占用 Σw=1 主预算 |

3. **数据驱动校准**：离线 trace 重放搜 w 组合（目标 = P99 时延↓ + 有效吞吐↑ + 切换震荡↓）；在线 AIMD 轻量自适应（配合 §4 hold-down 防抖）；机内 / 跨节点用两套权重分档。

### 5.3 选路过程全参数与变量说明表

#### A. 请求级输入

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `src`, `dst` | 传输源/目的（GPU/节点） | 上层 TransferRequest 直接给定 |
| `bytes` | 本次传输字节数 | 上层给定；用于 serialize_delay、小包探测判断 |
| `ttft_sensitive` | 是否首 token 延迟敏感（prefill/首包） | 上层标记；控制 LatencyPenalty 是否生效 |

#### B. 候选与路径属性

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `candidates` | (src,dst) 的全部候选路径 | `enumerate_paths`，带缓存；静态拓扑推导 |
| `best_direct` | 直连兜底路径 | candidates 中的 Direct 项；永不淘汰 |
| `path.links` | 路径上的链路序列 | 拓扑展开 |
| `n_hops` | 路径链路数 | `len(path.links)`；静态 |
| `max_hops` | 候选集最大跳数 | `max(n_hops over candidates)`；用于 Hops 归一化 |
| `path.has_relay` | 是否经中转 | 路径类型判定（非 Direct 即 true） |

#### C. CoDel 检测器状态（每 link 一个，事件驱动，无采样）

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `sojourn` | 单任务排队驻留时延 | `dequeue_ts − enqueue_ts`，steady_clock 打点 |
| `min_sojourn` | 滑动窗内 min(sojourn) | 检测器维护的滑动窗最小值；CoDel 判据用 |
| `target` | 可容忍排队时延上限（拥塞阈值） | `k × P10(T_xfer)`, k∈[1.5,2]；冷启动 1ms |
| `interval` | 确认拥塞的持续窗口 | `1.5~2 × decode_step_period`；冷启动 100ms |
| `T_xfer` | 单任务实际传输时延 | `complete_ts − dequeue_ts`，喂 target 自校准 |
| `P10(T_xfer)` | T_xfer 的滑动 10 分位 | 检测器内 Percentile 估计；代表最顺畅速度 |
| `link.congested` | 该 link 是否拥塞 | `is_congested()`：min_sojourn≥target 持续≥interval（带滞回） |

#### D. Score 打分项（均归一化到 [0,1]）

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `BW_avail` | 可用带宽收益 | 一般：`min(link_capacity−EWMA_used_bw)/BW_ref`；NVLink relay：`min(EWMA吞吐, pacing_rate)/BW_ref` |
| `link_capacity` | 链路理论容量 | 静态拓扑常量 |
| `EWMA_used_bw` | 链路已用带宽（实测） | `report_transfer_result` 吞吐回灌的 EWMA（不依赖 NVML） |
| `BW_ref` | 带宽归一化基准 | 本机最快链路理论带宽（如 NVLink ~150GB/s） |
| `SojournPenalty` | 当下拥塞惩罚 | `Σ_link max(0, min_sojourn−target)/target` |
| `Hops` | 跳数成本 | `Hops_norm = (n_hops−1)/(max_hops−1)` |
| `link_type_bias` | 链路类型静态偏好（可选） | 静态配置表（机内 NVLink>跨节点 RDMA>Storage）；不依赖 sojourn |
| `LatencyPenalty` | 首包延迟惩罚（条件） | `ttft_sensitive ? base_latency/latency_budget : 0` |
| `base_latency` | 路径静态时延 | `Σ_link(prop_delay + serialize_delay)` |
| `prop_delay` | 链路传播时延 | 静态拓扑常量 |
| `serialize_delay` | 序列化时延 | `bytes / link_capacity` |
| `latency_budget` | TTFT 预算（SLO） | 配置项（如 50ms） |
| `w1..w5`, `b` | 各项权重 | 初值见 §5.2；离线搜索 + 在线 AIMD 校准 |
| `path.score` | 路径综合得分 | 上述加权求和；viable 内排序键 |

#### E. 决策约束与防抖参数

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `min_relay_memory` | 中转 GPU 最小显存门槛 | 配置项；硬约束 |
| `relay_free_mem(path)` | 中转节点可用显存 | 中转 GPU 上报 |
| `relay_benefit_margin` | 中转须优于直连的比例 | 配置项（默认 10%）；[6] 保守回退 |
| `hold-down (N×interval)` | 切路后冷却期 | 配置项 N；[8] 防 flapping |
| `last_switch` / `current_path` | 上次切换时刻 / 当前路径 | 检测器/选路器状态；判 within_hold_down |
| `split_margin` | 多路相比单路最优的加速收益门槛 | 配置项；[7]/§5.5 SplitPath |
| `split_threshold` | 触发切分的最小总字节 | 配置项；低于此不切（避免调度开销） |
| `block_list` | 传输任务的 block 列表 | SGLang `transfer_blocks`：`(src_addr,dst_addr,size)`，由 `group_concurrent_contiguous` 生成 |
| `rate_i` | 路径 i 有效速率 | `BW_avail_i × BW_ref`（NVLink relay 含 pacing 上限） |
| `share_i` | 路径 i 承载比例 | `rate_i / Σ rate_j`；带宽加权 |
| `makespan` | 多路并行的最慢完成时间 | `max_i(assigned_bytes_i / rate_i)`；split 优化目标 |
| `target × 0.5` | 退出 CONGESTED 的滞回下阈 | 由 target 派生；§4 滞回 |

#### F. NVLink relay pacing 参数（§4.5，link 级硬限速）

| 参数/变量 | 作用 | 来源 / 测量计算方法 |
|---|---|---|
| `pacing_rate(link)` | NVLink relay 在该 link 的带宽天花板（硬限速） | AIMD 闭环动态调节，复用 min_sojourn 信号；link 级令牌桶执行 |
| `pacing_rate_max` | pacing 上限（防失控） | `NVLink 理论带宽 × X%`，配置项 |
| `pacing_rate_min` | pacing 下限（保底搬运/防饥死） | 配置项；低于此值该 NVLink relay 路径进 violate |
| `α`（加性增步长） | min_sojourn<target 时提速 | 配置项；试探慢（小步） |
| `β`（乘性减因子, <1） | min_sojourn≥target 时退让 | 配置项；退让快（响应 < 1 decode step） |
| `X%` | NVLink 给 relay 的带宽上限比例 | 配置项；pacing_rate_max 推导用 |
| relay 全局上限 | 限制整个 relay 子系统总预算 | 配置项；与 link 级桶双层限制 |

### 5.4 与「队列空时盲投」盲区的处理

排队信号只在**已有任务排队**时有效；首个任务/队列空时无信号 → 盲投。缓解：

- **小包探测**：队列空时，首个任务用小 block 试探，实测吞吐反馈。
- **吞吐反馈闭环**：复用 `report_transfer_result` 的 EWMA（actual 吞吐回灌 LinkState）作为空队列期的唯一辅助信号。
- 全程不依赖外部采样。

### 5.5 SplitPath 任务切分策略

**背景（来自 SGLang 代码事实）**：PD 分离 / hicache 的 KVCache 传输任务本身就是一个 **block 列表**——
- PD 分离（`mooncake/conn.py`）：最终落到 `engine.batch_transfer_sync(session, src_addrs, dst_addrs, lengths)`，即一批 `(src_addr, dst_addr, length)` 三元组（`transfer_blocks`）。
- block 由 `group_concurrent_contiguous(prefill_kv_indices, dst_kv_indices)` 生成：把 KV pool 里**连续的索引段**（diff==1）合并成一个 block，断点处切开；block 大小 ∝ 该连续段 token 数 × `item_len`。
- 还天然按 **layer**、**K/V 分离**组织（`enable_custom_mem_pool` 时已是 per-layer ThreadPoolExecutor 并发）；hicache（`memory_pool_host.py`）以 `page_size` 分页。

**结论**：传输任务是「一堆有明确 size、独立 src/dst 的 block」，切分它并分配到不同链路几乎零额外成本（block 边界、并发框架 SGLang 已备好）。SplitPath 即「把 block_list 按各 viable 路径的可用带宽加权分配」。

#### 5.5.1 切分粒度

- **以 block（contiguous group）为最小切分单位**，对齐 SGLang 的 `transfer_blocks`，**绝不切碎单个 block**——保持单次 RDMA/NVLink 传输的地址连续性，避免破坏 `batch_transfer_sync` 的合并优化。
- 若单个 block 过大（> 路径公平份额）可在 **layer / page 边界**进一步拆（这些边界 SGLang 已暴露），但不在 block 内部任意字节处切。

#### 5.5.2 带宽加权分配（目标：最小化 makespan）

```
输入：block_list = [(src, dst, size_b), ...]，viable 路径集 P

[1] 每条路径 i 的有效速率（已含 pacing 上限与拥塞折扣）：
      rate_i = BW_avail_i × BW_ref            # NVLink relay: min(EWMA吞吐, pacing_rate)
[2] 路径 i 的目标承载比例：
      share_i = rate_i / Σ_j rate_j           # 带宽加权
[3] 贪心装箱（LPT：块按 size 降序，依次塞给“当前预计完成时间最早”的路径）：
      for blk in sort_desc(block_list, by=size):
          pick path i = argmin( assigned_bytes_i / rate_i )   # 预计完成时刻最早
          assign blk → path i
[4] makespan = max_i ( assigned_bytes_i / rate_i )            # 最慢路径决定总耗时
```

- **目标是最小化 makespan（最慢那条路的完成时间），不是平均分**——慢路分少、快路分多，让各路**几乎同时完成**。
- LPT（Longest-Processing-Time-first）贪心是经典 makespan 近似，块粒度足够细时接近最优。
- 每条 NVLink 子路径分到的量**各自仍受其 pacing_rate 约束**（§4.5），即 split 不会绕过限速；某子路径 pacing 被压到地板时，它的 rate_i→0，自然分不到 block。

#### 5.5.3 不切分的保护条件（避免切分开销 > 收益）

满足任一即**退回单路 winner**，不做 split：

- **路径不足**：link-disjoint 的 viable 路径 < 2。
- **块数不足**：`len(block_list) < len(viable)`（块比路径还少，没法分）。
- **总量太小**：`total_bytes < split_threshold`（小传输的多路调度/重组开销占比过高）。
- **收益不足**：`est_makespan(split) ≥ winner.est_time × (1 − split_margin)`（多路相比单路最优没有显著加速）。

#### 5.5.4 与 pacing / CoDel 的关系

- **split 在 Score/过滤之后**：只在已通过硬约束（含 pacing 地板、CoDel 拥塞）的 viable 路径间分配，被判出局的路径不参与 split。
- **rate_i 实时反映 pacing**：推理流量挤占某 NVLink 子路径 → 其 pacing_rate 下降 → rate_i 下降 → 下次切分自动少分给它，形成「split 比例随 pacing 动态再平衡」。
- **重组**：各子路径独立完成各自 block 子集，目的端按 dst_kv_indices 原位写入，**无需额外重排**（block 的 dst 地址本就独立）。

---

## 6. 组件设计

```cpp
// 新增：每条 link 一个检测器
class CodelCongestionDetector {
public:
    void on_enqueue(uint64_t task_id);          // 打 enqueue_ts
    void on_dequeue(uint64_t task_id);          // 算 sojourn，喂入窗口
    void on_complete(double xfer_time);         // 喂 T_xfer 给 target 自校准

    bool   is_congested() const;                // CoDel 判定 + 滞回
    double current_min_sojourn() const;
    double target() const;                       // 自校准当前值
    double interval() const;

private:
    SlidingWindow<double> sojourn_window_;       // min-sojourn 追踪
    Percentile<double>    xfer_p10_;             // target 自校准源
    double target_  = 1e-3;                      // 冷启动 1ms
    double interval_= 0.1;                       // 冷启动 100ms
    bool   congested_ = false;
    TimePoint congested_since_;
    TimePoint last_switch_;                      // hold-down
};
```

挂载点：
- `NodeAgent` 持有 `unordered_map<LinkId, CodelCongestionDetector>`。
- relay adapter 在 enqueue/dequeue/complete 三处回调检测器。
- `PathSelector::score` 和 `violates_constraints` 读检测器状态。

```cpp
// 新增：每条 NVLink link 一个 pacing 控制器（link 级硬限速）
class NvlinkPacingController {
public:
    bool   allow(size_t bytes);                  // 令牌桶：是否放行本次传输
    void   on_signal(double min_sojourn,
                     double target);             // 复用 CoDel 信号做 AIMD 调速
    double rate() const;                         // 当前 pacing_rate
private:
    double rate_ = pacing_rate_min;              // 冷启动从保守值起
    double rate_max_ = /* NVLink × X% */;
    double rate_min_ = /* 保底搬运 */;
    double alpha_ = /* 加性增 */;
    double beta_  = /* 乘性减 <1 */;
    TokenBucket bucket_;
};
```

- pacing 控制器与 CoDel 检测器**同 source 信号（min_sojourn）、不同作用层**：pacing 控速率，CoDel 控选路。
- relay 发起传输前先过 `pacing.allow(bytes)`；`PathSelector` 读 `pacing.rate()` 入 BW_avail 与 pacing 地板硬约束。

---

## 7. 验证方法

1. **检测器正确性**：注入已知拥塞（SGLang 满载），看是否在 ~1 interval 内报 CONGESTED，burst 间隙是否不误报。
2. **降级有效性**：降级后 NVLink relay 队列是否真泄空。
3. **无震荡**：观察是否 flapping（验证滞回/hold-down）。
4. **计时噪声底**：微基准实测本机 `steady_clock` 单次开销分布 + 空队列 sojourn 噪声底，确定 target 最小可设值。
5. **Score 排序合理性**：构造已知优劣的候选集，验证 winner 选择与 margin/hold-down 闸门行为符合预期。
6. **pacing 有效性**：推理满载时观察 pacing_rate 是否迅速乘性退让、推理空闲时是否加性爬升；验证对 DeepEP all-to-all 突发的退让响应 < 1 decode step。
7. **饥死保护**：验证 pacing_rate 不会被压到 0，KVCache relay 不出现反向死锁（prefill 被卡）。

---

## 8. 落地步骤

1. 实现 `CodelCongestionDetector`（min-sojourn 窗口 + target P10 自校准 + 滞回）。
2. `LinkState` 增加 `pending_tasks` / `min_sojourn_ms` 字段。
3. relay adapter 三处回调接入。
4. `PathSelector`：`score` 接入 §5.2 五项（BW_avail 对 NVLink relay 取 pacing 上限）；`violates_constraints` 加 CoDel 拥塞分支与 pacing 地板分支。
5. 实现 `NvlinkPacingController`（令牌桶 + AIMD 调速，复用 CoDel min_sojourn 信号）；relay 发起前过 pacing.allow。
6. 空队列盲区由吞吐反馈 EWMA 补位；拥塞判定与 pacing 调速链路全程不依赖 NVML。
7. SplitPath：`build_split_path` 按 §5.5 带宽加权 LPT 装箱切分 `block_list`（对齐 SGLang `transfer_blocks` 粒度，不切碎单块）；各子路径复用 `batch_transfer_sync`。
8. 跑第 7 节验证。
