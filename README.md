# Veyra

`veyra` is a lightweight Python CLI for running local ONNX causal language models. It opens a polished REPL with slash commands, history, autocomplete, autosuggestions, and streaming output.

## Install

From this checkout:

```bash
uv tool install
```

From GitHub:

```bash
uv tool install git+https://github.com/Jdudeo5972/veyra-cli.git
pipx install git+https://github.com/Jdudeo5972/veyra-cli.git
```

## Development

```bash
uv sync
uv run veyra
```

## Usage

```bash
veyra
veyra fetch
veyra run "Hello"
veyra add ./models/foo
veyra add C:\Users\Jack\Models
veyra inspect ./models/foo
veyra models
veyra update
```

Inside the shell:

```text
/model fetch
/model list
/model use NAME
/model add PATH
/model test
/model remove NAME
/mode qwen
/profile name Nova
/device list
/device help openvino
/stats on
/theme rainbow
/chat list
/chat export markdown
```

`/model add PATH` can point at one model directory or a folder containing multiple model directories. `/model test` runs a one-token smoke test and reports load time, first-token time, total time, and the sampled token.

During generation, Ctrl+C stops generation. On Windows terminals, double-tapping Tab also requests a stop between generated tokens.

## Fetching Models

`veyra fetch` and `/model fetch` list compatible private or public repositories from the `veyra-ai` Hugging Face organization. If a repository contains several exports under `onnx/`, Veyra asks which variant to install and recommends Int8 when available. Only the selected model variant, any required companion ONNX files, and tokenizer/config metadata are downloaded.

Sign in before fetching private models:

```bash
hf auth login
```

Compatible ONNX repositories must include `tokenizer.json` alongside their model configuration.

## Shell Commands

Core:

```text
/help
/status
/exit
/quit
/clear
```

Models:

```text
/model
/model list
/model use NAME
/model fetch
/model refresh
/model update
/model update all
/model add PATH
/model inspect
/model test [NAME]
/model remove NAME
```

Prompting and generation:

```text
/mode base|chatml|qwen|gemma|mistral|llama3
/system TEXT
/temp VALUE
/tokens N
/topk N
/topp VALUE
/repetition VALUE
/stats on|off
```

Profile, device, and appearance:

```text
/profile show
/profile name NAME
/profile mode MODE
/device
/device list
/device help DEVICE
/device cpu|directml|openvino
/theme list
/theme veyra|warm|red|pink|lime|green|blue|cyan|purple|orange|gray|rainbow|mono
/autoload on|off
```

Chats:

```text
/chat new
/chat list
/chat load NAME
/chat rename NAME
/chat export markdown
/chat path
```

## Prompt Modes

Base mode sends the user text directly to the model as a raw completion prompt.

ChatML and Qwen modes format prompts like:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

Gemma mode follows the Gemma tokenizer template, using `<bos>`, `<start_of_turn>user`, `<start_of_turn>model`, and `<end_of_turn>`. Mistral mode uses `[INST] ... [/INST]`. Llama 3 mode uses the `<|start_header_id|>` chat header format.

When possible, Veyra infers the prompt mode from `config.json` and `tokenizer_config.json` when adding a model.

## Model Profiles

Each registered model can carry a small profile:

- prompt mode
- assistant display name

Use `/profile name NAME` to change the assistant label from `Veyra ›` to something else for the active model. Use `/profile mode MODE` to persist a preferred prompt mode for that model.

## Devices

Veyra defaults to CPU. Use `/device list` to see ONNX Runtime execution providers available in your current Python environment.

Common providers:

- `cpu`: standard `onnxruntime`, or the CPU provider included with the Windows DirectML build
- `directml`: included by default on 64-bit Windows through `onnxruntime-directml`
- `openvino`: uses OpenVINO `AUTO` selection across supported CPU and GPU devices when an OpenVINO-enabled ONNX Runtime build is installed

Run `/device help openvino` or another provider name for a short install hint. DirectML and OpenVINO use separate ONNX Runtime builds and cannot be installed together reliably in the same Python environment. The Windows OpenVINO combination tested for this release is `onnxruntime-openvino==1.24.1` with `openvino==2025.4.1`.

GPU acceleration is not always faster for small autoregressive models because each generated token requires a separate runtime call. Use `/stats on` to compare providers on the model and hardware you actually use.

### Future Device Plans

The following providers will return after they have been tested end to end with Veyra models:

- NVIDIA CUDA
- NVIDIA TensorRT
- AMD ROCm
- Apple Core ML

## Stats

Use `/stats on` to show a muted stats line under each response:

```text
stats: 24 tokens | 18.42 tok/s | first token 0.31s | total 1.30s | device cpu
```

## Files

Config is stored at `~/.config/veyra/config.json`.

Models are stored at `~/.local/share/veyra/models/`.

Chats are JSONL files in `~/.local/share/veyra/chats/`.

Prompt history is stored at `~/.local/state/veyra/history.txt`.

## Model Architecture

Veyra treats architecture as metadata and uses ONNX graph inputs and outputs as the source of truth wherever possible.

Currently tested support includes:

- Veyra2 Llama-style cached exports
- Gemma/Gemma 3 cached exports
- Qwen2/Qwen2.5/Qwen3-style cached exports
- split Qwen3.5/Next-style exports using `inputs_embeds`, `embed_tokens.onnx`, and recurrent/conv cache state
- SmolLM2-style cached exports

Unsupported required inputs are reported clearly by `veyra inspect PATH`.

## Updating

`veyra update` prints install commands and does not self-modify:

```bash
uv tool install git+https://github.com/Jdudeo5972/veyra-cli.git
pipx install git+https://github.com/Jdudeo5972/veyra-cli.git
```

## Versioning

Veyra uses calendar versions in `YEAR.MONTH.DAY` format, displayed and tagged with a leading `v`, such as `v2026.7.31`. Additional releases on the same day append a counter, such as `v2026.7.31.1`.

## License

Veyra is licensed under the [Apache License 2.0](LICENSE).
