# Veyra

`veyra` is a lightweight Python CLI for running local ONNX causal language models on CPU-first machines, including Raspberry Pi-class hardware. It opens a polished REPL with slash commands, history, autocomplete, and simple streaming output.

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
/chat list
/chat export markdown
```

## Prompt Modes

Base mode sends the user text directly to the model as a raw completion prompt.

ChatML mode formats prompts like:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

If a system prompt is set with `/system TEXT`, it is placed before the user turn.

## Files

Config is stored at `~/.config/veyra/config.json`.

Models are stored at `~/.local/share/veyra/models/`.

Chats are JSONL files in `~/.local/share/veyra/chats/`.

Prompt history is stored at `~/.local/state/veyra/history.txt`.

## Raspberry Pi Notes

Veyra uses CPU-only ONNX Runtime by default with small thread counts: two intra-op threads and one inter-op thread. Int8 ONNX models are recommended. Full-sequence re-evaluation is used in v0, so the first generation and longer responses may be slow on small CPUs.

## Model Architecture

Veyra2 currently uses Llama-style model config. Veyra3 may use a Gemma-family config. The CLI treats architecture as metadata and uses ONNX graph inputs and outputs as the source of truth wherever possible. Current supported causal LM inputs are `input_ids`, `attention_mask`, and `position_ids`; unsupported required inputs are reported clearly by `veyra inspect`.

## Updating

`veyra update` prints update instructions and does not self-modify. Typical commands are:

```bash
uv tool upgrade veyra
pipx upgrade veyra
```
