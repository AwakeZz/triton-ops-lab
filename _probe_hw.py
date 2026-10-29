import torch
p = torch.cuda.get_device_properties(0)
print("GPU        :", p.name)
print("SM 数      :", p.multi_processor_count)
print("L2 大小    :", p.L2_cache_size/1024/1024, "MB")
for attr in ["shared_memory_per_multiprocessor","shared_memory_per_block","regs_per_multiprocessor","max_threads_per_multi_processor","warp_size"]:
    print(f"{attr:34s}:", getattr(p, attr, "n/a"))
