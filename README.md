# FlexFlow

统一的 KVCache 传输库 —— 在同一套语义下统一 sglang 的 **PD 分离传输** 与 **HiCache 多级缓存传输**，
支持同机内 / 跨节点传输，自动适应机内拓扑亲和性，并支持 GPU→SSD 传输。

核心是 `FlexFlow` 引擎 + 单一传输入口 `transfer(src: Locator, dst: Locator)`：

- PD 分离 = `push(LocalPages → PeerPages)`
- HiCache = `put/get(LocalPages ↔ StorageKeys)`
- GPU→SSD = `submit(LocalPages → FileRegion)`

## 设计文档

详见 [`UnifiedKVCacheTransfer_Design.md`](./UnifiedKVCacheTransfer_Design.md)，包含：

- 背景与统一洞察
- 设计原则与核心抽象（MemHandle / PageLayout / Locator / TransferHandle）
- API 接口设计（初始化、内存池注册、建链、异步传输）
- 拓扑亲和性路径选择
- 典型场景使用示例（PD 分离、HiCache L1↔L2↔L3、GPU↔SSD）

## 快速预览

```python
from flexflow import FlexFlow, PageLayout, LocalPages, PeerPages, TransferHint

eng = FlexFlow(rank=0, device=0, role="both")
pool = eng.register_pool(kv_buffers, layout)
handle = eng.submit(src=LocalPages(pool, src_idx),
                    dst=LocalPages(host_pool, dst_idx))
eng.wait([handle])
```

## License

Apache-2.0
