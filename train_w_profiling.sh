/usr/local/bin/nsys profile \
    -t cuda,nvtx \
    -w true \
    --cuda-memory-usage=true \
    -s cpu \
    -f true \
    -x true \
    -o ant_base_w_compile \
    accelerate launch --config_file 1gpu.yaml --gpu_ids 6 -m scripts.train --abstractor --use_text_cache