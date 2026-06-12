# FlexFlow GDR vs cudaMemcpy 基准 Makefile
#
# 依赖：
#   - CUDA Toolkit (nvcc)              -> 设备内存 / cudaMemcpyAsync
#   - rdma-core-devel (libibverbs-dev) -> RDMA loopback (场景 A)
#   - 运行时需 GPUDirect RDMA: 加载 nvidia-peermem (或 nv_peer_mem) 内核模块
#
# 用法：
#   make                     # 编译
#   make run                 # 用默认 GPU0 跑
#   make CUDA_HOME=/usr/local/cuda-11.8
#
# 若系统只有 libibverbs.so.1 而无 .so 软链接（缺 -dev 包的开发链接），
# 用 IBVERBS_LIB 指定具体库文件：
#   make IBVERBS_LIB=/usr/lib64/libibverbs.so.1

CUDA_HOME   ?= /usr/local/cuda
NVCC        ?= $(CUDA_HOME)/bin/nvcc

# 默认链接 -libverbs；若无 .so 软链可改成具体路径
IBVERBS_LIB ?= -libverbs

# GPU 架构（按需修改：80=A100, 90=H100, 70=V100）
SM_ARCH     ?= 80

NVCCFLAGS   = -O3 -std=c++17 -gencode arch=compute_$(SM_ARCH),code=sm_$(SM_ARCH)
LDFLAGS     = -lcudart $(IBVERBS_LIB)

TARGET = gdr_vs_memcpy
SRC    = gdr_vs_memcpy.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --gpu 0 --csv result.csv

clean:
	rm -f $(TARGET) result.csv

.PHONY: all run clean
