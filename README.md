# Veyra

`veyra` is a lightweight Python CLI for running local ONNX causal language models on CPU-first machines, including Raspberry Pi-class hardware. It opens a polished REPL with slash commands, history, autocomplete, and simple streaming output.

Repository: [Jdudeo5972/veyra-cli](https://github.com/Jdudeo5972/veyra-cli)

## Install

```bash
uv tool install .
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
veyra inspect ./models/foo
veyra models
```

Inside the shell:

```text
/model fetch
/model list
/model use NAME
/mode base
/mode chatml
/mode qwen
/mode gemma
/mode mistral
/mode llama3
/chat list
/chat export markdown
```

## Prompt Modes

Base mode sends the user text directly to the model as a raw completion prompt.

ChatML and Qwen modes format prompts like:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

Gemma mode uses `<start_of_turn>user` / `<start_of_turn>model` turns. Mistral mode uses `[INST] ... [/INST]`. Llama 3 mode uses the `<|start_header_id|>` chat header format.

If a system prompt is set with `/system TEXT`, it is placed before the user turn.

## Files

Config is stored at `~/.config/veyra/config.json`.

Models are stored at `~/.local/share/veyra/models/`.

Chats are JSONL files in `~/.local/share/veyra/chats/`.

Prompt history is stored at `~/.local/state/veyra/history.txt`.

## Raspberry Pi Notes

Veyra uses CPU-only ONNX Runtime by default with small thread counts: two intra-op threads and one inter-op thread. Int8 ONNX models are recommended. Full-sequence re-evaluation is used in v0, so the first generation and longer responses may be slow on small CPUs.

## Devices

Veyra defaults to CPU. Use `/device list` in the shell to see ONNX Runtime execution providers available in your current Python environment. CUDA requires a CUDA-enabled ONNX Runtime build, typically `onnxruntime-gpu`, plus compatible NVIDIA drivers/CUDA/cuDNN.

## Model Architecture

Veyra2 currently uses Llama-style model config. Veyra3 may use a Gemma-family config. The CLI treats architecture as metadata and uses ONNX graph inputs and outputs as the source of truth wherever possible. It supports standard decoder-only cached exports, Gemma-style exports, Qwen2/2.5/3-style cached exports, and split Qwen3.5/Next-style exports that use `inputs_embeds`, `embed_tokens.onnx`, and recurrent/conv cache state. Unsupported required inputs are reported clearly by `veyra inspect`.

## Updating

`veyra update` prints update instructions and does not self-modify. Typical commands are:

```bash
uv tool upgrade veyra
pipx upgrade veyra
```
