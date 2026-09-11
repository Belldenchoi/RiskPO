# Chạy RiskPO gốc trên 4 A100 bằng Python

Server **2 L40** dùng launcher `local_2l40/train.py`; xem [hướng dẫn 2 L40](../local_2l40/README.md).
Launcher 4 A100 này giữ cấu hình cũ làm mặc định.

Bộ file này chuyển từ `RiskPO_Original_Qwen3_4B_4xA100_Local_Full.ipynb`.
Giữ Qwen/Qwen3-4B, LoRA rank 8/alpha 16, tắt thinking, RiskPO gốc có quantile
tracking và vanilla loss. Dùng 4 GPU trên cùng một máy, FSDP size 4 và vLLM TP 1.
Global batch và PPO minibatch vẫn là 20 prompt, mỗi prompt sinh 5 câu trả lời:
100 response được chia trên 4 GPU. Learning rate vẫn `1e-6`, microbatch mỗi GPU
là 1. Mặc định GSM8K, 200 bước; `full` vẫn là LoRA.

## 1. Tạo venv và cài requirements

Cần **Linux x86_64, Python 3.10, NVIDIA driver tương thích CUDA 12.4** và đủ 4 A100
được cấp cho phiên chạy. Script không cài Python hệ thống hoặc NVIDIA driver.
Chuyển **cả repository RiskPO** sang máy đích vì runner dùng các module trong `verl/`.
Không chuyển venv Windows sang Linux.

Từ thư mục gốc repository `RiskPO`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r local_4a100/requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

Hoặc dùng `bash local_4a100/setup_env.sh`, rồi `source .venv/bin/activate`.
Mọi lệnh training dưới đây dùng chính Python trong venv vừa tạo.

`requirements.txt` giữ các phiên bản đã pin trong notebook, chọn wheel PyTorch
CUDA 12.4 và wheel FlashAttention dựng sẵn cho CPython 3.10/Linux. Các dependency
mà notebook không pin vẫn được pip resolve; mỗi run lưu `requirements-freeze.txt`.

**Máy công ty không ra Internet:** tạo venv không thay thế bước cung cấp thư viện.
File requirements mặc định cần PyPI, PyTorch và GitHub. Nếu công ty có kho nội bộ,
IT cần đưa đủ package, gồm các wheel CUDA/FlashAttention tương ứng, vào kho đó:

```bash
python -m pip install --index-url https://KHO-NOI-BO/simple \
  -r local_4a100/requirements-internal.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

`requirements-internal.txt` có cùng yêu cầu phiên bản và không chứa URL công khai.
Nếu đã có wheelhouse đầy đủ do IT cung cấp, có thể thay lệnh cài dependencies bằng:

```bash
python -m pip install --no-index --find-links /data/wheelhouse \
  -r local_4a100/requirements-internal.txt
```

Bộ mã này **không kèm wheelhouse, Python runtime, trọng số Qwen3-4B hoặc dataset**.

## 2. Kiểm tra máy trước khi chạy

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python local_4a100/check_environment.py
```

Lệnh tạo `environment_report.json`, kiểm tra phiên bản, `pip check`, GPU, import
PyTorch/vLLM/FlashAttention và scorer. Không tải gì, không cài gì, không training.
Nếu môi trường chưa được cài, có thể chạy script bằng Python hiện có để xem
các thư viện còn thiếu. Cần Python 3.8 trở lên cho script kiểm tra; training cần 3.10.
Kiểm tra NCCL giữa 4 GPU được chạy thêm bởi `train.py` trước khi training.

## 3. Chuẩn bị model và GSM8K

Nếu máy có mạng hoặc đã cấu hình proxy được công ty cho phép, runner có thể lấy
Qwen3-4B và GSM8K khi chưa có cache:

```bash
python local_4a100/train.py --work-dir /data/riskpo --prepare-only
```

Với máy không có mạng, dùng `--offline`, thư mục model đầy đủ và dataset đã preprocess:

```text
/data/models/Qwen3-4B/               # config, tokenizer và toàn bộ .safetensors
/data/riskpo-data/raw/gsm8k/train.parquet
/data/riskpo-data/raw/gsm8k/test.parquet
```

Trên máy có mạng và có các dependencies, tạo hai file dataset để chuyển sang:

```bash
python local_4a100/prepare_data.py --output-dir /path/to/transfer/raw/gsm8k
```

Preprocess giữ nguyên prompt, reward và split trong notebook. File parquet GSM8K
thô tải từ Hub cần qua bước này; Telemath không thay thế cho GSM8K của baseline.
Model local phải là checkpoint **Qwen/Qwen3-4B gốc, không quantize**, gồm đủ các
weight shard và tokenizer. Không dùng model ID của API Viettel làm đường dẫn model.

## 4. Kiểm tra cấu hình rồi training

Chỉ kiểm tra môi trường/NCCL, dữ liệu, tokenizer, cấu hình; chưa training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python local_4a100/train.py \
  --offline \
  --model-path /data/models/Qwen3-4B \
  --data-dir /data/riskpo-data \
  --work-dir /data/riskpo \
  --prepare-only
```

Chạy đủ 200 bước bằng cách bỏ `--prepare-only`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python local_4a100/train.py \
  --offline \
  --model-path /data/models/Qwen3-4B \
  --data-dir /data/riskpo-data \
  --work-dir /data/riskpo
```

Chỉ khởi động **một** tiến trình `train.py`; nó quản lý 4 GPU qua Ray/FSDP/vLLM.
Không bọc thêm `torchrun train.py` hoặc mở bốn lệnh training độc lập.
Mỗi lần gọi tạo run mới; không tự resume run cũ. Tùy chọn dataset khác được giữ:
`--dataset easymath` (200 bước) hoặc `--dataset dapo` (500 bước).

## 5. Kết quả local

```text
/data/riskpo/runs/<run_name>/
  config.yaml, resolved_config.yaml, run_metadata.json
  requirements-freeze.txt, data_manifest.json
  train.log
  tensorboard/
  evaluation/200.jsonl
  results.json
  final_model/                     # model đầy đủ đã gộp LoRA, tokenizer, checksum
/data/riskpo/checkpoints/<run_name>/global_step_200/
  actor/                          # model/optimizer/extra của đủ 4 rank + adapter
  data.pt
```

Giữ hành vi notebook local: không Drive, không giới hạn phiên 40 bước; chỉ đánh giá
test và lưu checkpoint ở cuối. Nếu run dừng trước lần lưu cuối, chưa có checkpoint
mới để tiếp tục. Export chỉ bắt đầu khi training, final checkpoint và evaluation
đều đạt kiểm tra. Merge thực hiện trên CPU nên cần đủ RAM và disk cho base,
checkpoint và model xuất ra.

`results.json` là điểm policy cuối run trước merge; chưa benchmark lại model BF16
sau merge. Có thể chạy lại bước xuất nếu training đã xong nhưng export bị ngắt:

```bash
python local_4a100/export_model.py \
  --base /data/models/Qwen3-4B \
  --adapter /data/riskpo/checkpoints/RUN/global_step_200/actor/lora_adapter \
  --output /data/riskpo/runs/RUN/final_model \
  --thinking false --step 200
```

Exporter không ghi đè `final_model` hoặc `final_model.partial` đã tồn tại. Lệnh
export riêng không tự cập nhật cờ export trong `results.json`; xem
`final_model/export_manifest.json` để xác nhận lần xuất riêng.

## Các file chính và kiểm chứng

- `train.py`: điều phối kiểm tra, train, final evaluation, lưu và xuất model.
- `config.py`: builder cấu hình được giữ nguyên từ notebook 4 A100.
- `prepare_data.py`, `reward.py`, `thinking_dataset.py`: xử lý dữ liệu và reward gốc.
- `audit_data.py`, `validate_config.py`, `gpu_checks.py`, `distributed_preflight.py`: kiểm tra trước training.
- `collect_results.py`, `export_model.py`: xác minh kết quả và xuất model đầy đủ.

Đã kiểm tra cấu trúc Python và đối chiếu cấu hình với notebook bằng kiểm tra CPU.
Chưa cài bộ CUDA, training hoặc export model thật trên máy Windows phát triển.
Chạy unit checks từ gốc repo: `python -m unittest discover -s local_4a100/tests -v`.
