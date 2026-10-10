# Veyra

`veyra` is a lightweight Python CLI for running local ONNX and Transformers language models. It opens a polished REPL with slash commands, history, autocomplete, autosuggestions, and streaming output.

## Install

From this checkout:

```bash
uv tool install .
```

From GitHub:

```bash
uv tool install git+https://github.com/Jdudeo5972/veyra-cli.git
pipx install git+https://github.com/Jdudeo5972/veyra-cli.git
```

The standard installation includes ONNX Runtime, Transformers, PyTorch, and Safetensors. Transformers is the default when a model repository provides Safetensors weights; ONNX remains selectable for compatible exports.

## Development

```bash
uv sync
uv run veyra
```

## Usage

```bash
veyra
veyra fetch
veyra fetch owner/model
veyra fetch owner/model --trust-remote-code
veyra run "Hello"
veyra add ./models/foo
veyra add ./models/foo --runtime transformers --trust-remote-code
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
/model info
/model trust on
/model remove NAME
/mode qwen
/profile name Nova
/device list
/device help openvino
/stats on
/seed 42
/context auto
/retry
/doctor
/theme rainbow
/chat list
/chat export markdown
```

`/model add PATH` can point at one model directory or a folder containing multiple model directories. Mixed local folders default to Transformers/Safetensors; use `/model add PATH onnx` or `veyra add PATH --runtime onnx` to select ONNX explicitly. `/model test` runs a one-token smoke test and reports load time, first-token time, total time, and the sampled token. `/model info` shows architecture, runtime, weights, context limit, cache support, and profile metadata.

During generation, Ctrl+C stops generation. On Windows terminals, double-tapping Tab also requests a stop between generated tokens.

## Fetching Models

`veyra fetch` and `/model fetch` list compatible private or public repositories from the `veyra-ai` Hugging Face organization. You can also fetch any compatible Hub repository directly with `veyra fetch owner/model` or `/model fetch owner/model`.

Repositories may contain root-level Safetensors weights and ONNX exports in `onnx/`. Veyra shows both runtime choices and defaults to Transformers when Safetensors weights are available; ONNX remains available as the lightweight option. Only the selected weight format plus tokenizer/config metadata is downloaded.

Sign in before fetching private or gated models:

```bash
veyra auth login
```

The token is entered through a masked prompt, is never placed in Veyra's config or command history, and is stored by `huggingface_hub` in its standard local token store. Veyra rejects classic write tokens and fine-grained tokens with detected write permissions, and its Hub integration only lists and downloads files. Create either a read token or, preferably, a fine-grained token granting only read access to the models you need at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

For gated models, first accept the model's access terms in your browser. Then use `veyra auth status` to check the active account and permission without displaying the token. `veyra auth logout` removes the active saved Hugging Face credential. An `HF_TOKEN` environment variable takes precedence over saved credentials; Veyra reports this and will not attempt to overwrite or remove it.

Compatible repositories must include a root-level `tokenizer.json` and either an `.onnx` or `.safetensors` model. Veyra reports a clear error instead of guessing or borrowing a tokenizer from another model.

### Custom Model Code

Some Hugging Face models define their architecture or tokenizer in Python files stored in the model repository. Veyra blocks that code by default. After reviewing the repository, opt in while fetching or adding the model:

```bash
veyra fetch owner/model --trust-remote-code
veyra add ./models/foo --runtime transformers --trust-remote-code
```

For the selected model, `/model trust on` enables custom code immediately and `/model trust off` blocks it again. The setting is saved per model and preserved by model updates. Custom model code runs locally with your user account's permissions, so only enable it for repositories you trust.

## Shell Commands

Core:

```text
/help
/status
/doctor
/retry
/exit
/quit
/clear
```

Models:

```text
/hf [status|login|logout]
/model
/model list
/model use NAME
/model fetch [REPO_ID] [--trust-remote-code]
/model refresh
/model update
/model update all
/model add PATH [onnx|transformers] [--trust-remote-code]
/model inspect
/model info [NAME]
/model test [NAME]
/model trust [on|off]
/model remove NAME
```

Prompting and generation:

```text
/mode base|template|chatml|qwen|gemma|mistral|llama3
/system TEXT
/temp VALUE
/tokens N
/topk N
/topp VALUE
/repetition VALUE
/seed N|random
/context N|auto
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

Template mode uses the model tokenizer's own chat template through Transformers. Veyra detects both `chat_template.jinja` and templates embedded in `tokenizer_config.json`; new Veyra instruct models select this mode automatically.

ChatML and Qwen modes format prompts like:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

Gemma mode follows the Gemma tokenizer template, using `<bos>`, `<start_of_turn>user`, `<start_of_turn>model`, and `<end_of_turn>`. Mistral mode uses `[INST] ... [/INST]`. Llama 3 mode uses the `<|start_header_id|>` chat header format.

When possible, Veyra infers the prompt mode from `config.json`, `tokenizer_config.json`, and `chat_template.jinja` when adding a model.

## Model Profiles

Each registered model can carry a small profile:

- prompt mode
- assistant display name
- generation settings, including the token budget, sampling values, seed, and context length

Use `/profile name NAME` to change the assistant label from `Veyra ›` to something else for the active model. Use `/profile mode MODE` to persist a preferred prompt mode for that model. Changes made with `/tokens`, `/temp`, `/topk`, `/topp`, `/repetition`, `/seed`, and `/context` are restored whenever that model is selected again.

## Context Windows

Veyra reads the model context limit from `config.json`, tokenizer metadata, or a static ONNX sequence dimension. `/context auto` uses that maximum; `/context N` sets a smaller working window and rejects values above the model limit. If no limit is declared, Veyra will not accept a custom value it cannot validate and `/doctor` reports the missing metadata.

In conversational modes, Veyra uses a sliding history window. When the formatted conversation plus the output token budget would exceed the selected context, the oldest complete turns are dropped while the newest user message is preserved. If the current message itself cannot fit, generation stops with a clear error. Base mode has no chat history to slide.

Use `/seed N` for repeatable sampling or `/seed random` for nondeterministic generation. `/retry` replaces the most recent assistant response in the effective chat history and regenerates the last user turn.

## Diagnostics

`/doctor` checks the CLI and Python versions, writable data directories, installed runtimes, tokenizer, model weights or graph support, model metadata, and model session initialization. It does not run a benchmark or generate tokens.

## Devices

Veyra defaults to CPU. Use `/device list` to see ONNX Runtime execution providers available in your current Python environment.

The Transformers runtime currently uses CPU mode. DirectML and OpenVINO selections apply to ONNX models.

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

Veyra treats architecture as metadata. ONNX models use graph inputs and outputs as the source of truth wherever possible. Transformers models select `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` from `config.json`, use native KV caching, and honor the tokenizer's chat template.

Currently tested support includes:

- Veyra2 Llama-style cached exports
- Gemma/Gemma 3 cached exports
- Qwen2/Qwen2.5/Qwen3-style cached exports
- split Qwen3.5/Next-style exports using `inputs_embeds`, `embed_tokens.onnx`, and recurrent/conv cache state
- SmolLM2-style cached exports
- Safetensors causal language models supported by Transformers `AutoModelForCausalLM`
- Safetensors encoder-decoder models supported by Transformers `AutoModelForSeq2SeqLM`
- T5Gemma 2 pretrained encoder-decoder checkpoints through Transformers
- standalone `chat_template.jinja` files used by Veyra instruct models

Encoder-decoder models default to Base mode unless their tokenizer includes a chat template. Their encoder context limit applies to the input prompt; `/tokens` controls the separate decoder output budget.

Unsupported required inputs are reported clearly by `veyra inspect PATH`.

## Updating

`veyra update` prints install commands and does not self-modify:

```bash
uv tool install git+https://github.com/Jdudeo5972/veyra-cli.git
pipx install git+https://github.com/Jdudeo5972/veyra-cli.git
```

## Versioning

Veyra uses calendar versions in `YEAR.MONTH.DD` format, displayed and tagged with a leading `v`, such as `v2026.10.04`. Additional releases on the same day append a counter, such as `v2026.10.04.1`.

## License

Veyra is licensed under the [Apache License 2.0](LICENSE).
