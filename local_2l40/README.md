# RiskPO gốc trên 2 NVIDIA L40

Chạy bằng Python trên server Linux; dùng môi trường đã có nếu nó vượt qua kiểm tra.
Không cần cài lại requirements hoặc mở notebook. Giữ cả `local_4a100/`, `local_2l40/`
và `verl/` trong repository: bản L40 dùng chung runner, scorer và exporter.

## Cấu hình thí nghiệm

Đối chiếu với `RiskPO_QUATRO_Qwen3_8B_Colab.ipynb` hiện tại (tên cũ nhưng bên trong
là **Qwen/Qwen3-4B**). Dấu vết notebook nằm trong `notebook_comparison.json`.

| Tham số | Giá trị giữ nguyên |
|---|---|
| Model / dtype | Qwen3-4B gốc / BF16 |
| Dataset | GSM8K train/test, scorer final_numeric_v2 |
| Thinking | False |
| LoRA | rank 8, alpha 16, q/k/v/o/gate/up/down projections |
| Global train batch / PPO minibatch | 20 prompt / 20 prompt |
| Rollout | 5 response/prompt → 100 response/step |
| Phân phối | 2 GPU → 50 response/GPU; FSDP size 2, vLLM TP 1 |
| Actor microbatch | 1 response/GPU; 50 lần tích lũy/PPO minibatch |
| Learning rate / số bước | 1e-6 / 200 |
| Prompt / response tối đa | 1024 / 1024 token |
| Sampling | temperature 1, top_p 1, top_k -1 |

Khác biệt phương pháp có chủ đích: baseline dùng
`grpo_bundle_RVaR_quantile_tracking`, bật quantile tracking, policy loss `vanilla`,
`loss_agg_mode=token-mean`. Không dùng advantage/loss QUATRO. Các quantile .2/.9,
bundle 5, lr_q .1 và w_mix 1.5 giữ theo nhánh RiskPO gốc trong notebook.

Giữ actor microbatch=1 vì actor hiện tính `token-mean` riêng mỗi microbatch rồi
lấy trung bình qua tích lũy gradient. Tăng microbatch hoặc dynamic batching có thể
đổi trọng số giữa response dài/ngắn. Tối ưu bên dưới không sửa công thức loss.
Đổi số GPU/thứ tự thực thi vẫn có thể đổi chuỗi ngẫu nhiên và sai số số học;
không cam kết kết quả giống từng bit.

## Các profile tốc độ

| Profile | CPU offload actor/optimizer | Log-prob microbatch | vLLM max sequences/engine | Token budget | CUDA graph |
|---|---|---|---|---|---|
| `l40_2_fast` (mặc định) | tắt / tắt | 5 | 50 | 4096 | tắt |
| `l40_2_safe` | bật / bật | 1 | 24 | 2048 | tắt |
| `l40_2_graph` (thử nghiệm) | tắt / tắt | 5 | 50 | 4096 | bật |

`fast` giảm copy CPU/GPU, gom log-prob không gradient và tăng concurrency rollout.
Cả ba vẫn bật gradient checkpointing, remove padding, vLLM sleep/free cache;
không dùng quantization và không bật torch.compile cho actor. TP=1 cho phép mỗi
GPU sinh response độc lập, không chia tensor mỗi token giữa hai GPU.

Đây là các **cấu hình ứng viên**, chưa có benchmark L40 thực tế. `fast` cần nhiều
VRAM hơn `safe`; `graph` cần thêm VRAM/thời gian capture và phải thử với LoRA.
Không tự đổi profile, giảm batch, cắt token hoặc chuyển method khi OOM.
Không tự đặt `NCCL_P2P_DISABLE`: giữ cấu hình mạng/GPU của server và kiểm tra NCCL.

## 1. Kết nối và xem cấu hình

Trong terminal SSH, từ thư mục repository:

```bash
source .venv/bin/activate
python local_2l40/train.py --show-config
CUDA_VISIBLE_DEVICES=0,1 python local_4a100/check_environment.py \
  --gpu-count 2 --gpu-family L40
```

Nếu venv chưa có, làm theo `local_4a100/README.md`. File
`local_2l40/requirements.txt` dùng cùng bộ phiên bản của notebook; không thêm
dependency mới. Máy có kho nội bộ dùng `requirements-internal.txt`. Việc chuyển
profile GPU không tự xử lý lỗi pip backtracking của môi trường chưa cài xong.

## 2. Kiểm tra trước training

Thay các đường dẫn ví dụ bằng thư mục thật trên server:

```bash
CUDA_VISIBLE_DEVICES=0,1 python local_2l40/train.py \
  --offline --model-path /data/models/Qwen3-4B \
  --data-dir /data/riskpo-data --work-dir /data/riskpo-l40 \
  --prepare-only
```

Model cần đủ weights/tokenizer gốc; data cần
`/data/riskpo-data/raw/gsm8k/train.parquet` và `test.parquet` đã preprocess theo
notebook. Lệnh kiểm tra GPU/import, all-reduce NCCL 2 rank, dataset/prompt/scorer,
Hydra config; không xác nhận đủ peak VRAM cho training.

## 3. Đo tốc độ ngắn trên máy thật

Chỉ chạy sau khi đã xem cấu hình. Mỗi lệnh bên dưới **thực sự training 5 bước**,
tạo run riêng để đo thời gian, không tạo checkpoint/model hay điểm accuracy cuối:

```bash
CUDA_VISIBLE_DEVICES=0,1 python local_2l40/train.py \
  --offline --model-path /data/models/Qwen3-4B \
  --data-dir /data/riskpo-data --work-dir /data/riskpo-l40 \
  --hardware-profile l40_2_fast --benchmark-steps 5
```

Đổi `--hardware-profile` sang `l40_2_safe` hoặc `l40_2_graph` và chạy tuần tự để
so sánh trên cùng máy. Không chạy nhiều profile đồng thời. Xem `performance.json`
trong mỗi run: median step, thời gian gen/log-prob/update, throughput tổng theo
token, chiều dài response và peak allocated/reserved VRAM; bỏ step đầu khỏi thống kê.
Các response có thể khác do scheduling/RNG: so cả độ dài lẫn thời gian, không chỉ
nhìn một step. Run đo ngắn có horizon khác và không dùng làm kết quả so sánh phương pháp.
Nếu các step liên tục có grad_norm=0, cần kiểm tra tín hiệu học; tốc độ không chứng
minh training hiệu quả. Report liệt kê các zero_gradient_steps để đối chiếu.

## 4. Chạy full và lưu model local

Ví dụ dùng profile mặc định, bỏ `--benchmark-steps`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python local_2l40/train.py \
  --offline --model-path /data/models/Qwen3-4B \
  --data-dir /data/riskpo-data --work-dir /data/riskpo-l40 \
  --hardware-profile l40_2_fast
```

Một lệnh launcher quản lý cả hai GPU. Không bọc thêm `torchrun train.py`.
Chạy trong phiên `tmux` sẵn có trên server nếu cần duy trì khi mất SSH.

Chạy đủ 200 bước, đánh giá test một lần ở cuối và lưu checkpoint đủ **2 rank**.
Không backup Drive, không dừng sau 40 bước, không resume Colab. Chỉ lưu checkpoint
cuối như bản local đã thống nhất; ngắt trước lần lưu sẽ mất tiến độ run đó.

Kết quả:

```text
/data/riskpo-l40/runs/<run>/results.json
/data/riskpo-l40/runs/<run>/train.log
/data/riskpo-l40/runs/<run>/evaluation/200.jsonl
/data/riskpo-l40/runs/<run>/final_model/       # LoRA đã merge thành model đầy đủ
/data/riskpo-l40/checkpoints/<run>/global_step_200/actor/
```

Export merge trên CPU nên cần đủ RAM/disk; kết quả test là policy trước merge.
Cấu hình 4 A100 cũ vẫn là mặc định của `local_4a100/train.py`.

## Kiểm chứng và nguồn

Kiểm tra CPU: đối chiếu config toàn bộ với nhánh RiskPO của notebook QUATRO,
kiểm tra profile, rank count, kết quả đủ/thiếu, và phân tích log đo tốc độ.
Chưa chạy GPU training trên server L40; không khẳng định speedup hay đủ VRAM.

- NVIDIA L40: https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf
- Cách chỉnh token budget vLLM V0: https://docs.vllm.ai/en/v0.8.3/performance/optimization.html
- Code runtime đã đối chiếu: `verl/workers/actor/dp_actor.py`,
  `verl/workers/fsdp_workers.py`, `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`.
