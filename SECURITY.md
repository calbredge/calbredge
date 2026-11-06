# Calbredge Security & Privacy

What the agent (`calbredge`) does on your machine, what it records, and when anything leaves
it.

## What the agent collects

When you run `calbredge run`, it records the following about your machine:

- Hostname, operating system, and CPU model
- Total RAM and GPU model (if present)
- Number of physically present CPU cores (from `/sys/devices/system/cpu/possible`)
- Python version and runtime version
- Active power mode (for example `25W` or `MAXN`; on Jetson via `nvpmodel`) and whether the
  processor speeds were locked
- A SHA-256 hash of your model file (a fingerprint, not the model)
- The model's input and output names, types and shapes
- Benchmark results: p50/p90/p99 latency, throughput, batch size, protocol version, run status

On hardware that reports them, it also records:

- Average and peak power draw in watts, and energy per inference in joules: on an NVIDIA
  Jetson, and on an RK3588 with an external meter
- Peak temperature in °C, and whether it passed the throttling threshold: on an NVIDIA Jetson
  only, read just after the measured iterations

The run bundle (`~/.calbredge/runs/<id>/`) also holds the latency of every measured iteration,
the run's log, installed package versions, the command line and a fixed allowlist of
environment variables; credentials in any of them are replaced with `<redacted>`
(`docs/RESULT_SCHEMA.md` §6).

## What the agent does NOT collect

- Your model weights
- Your training data or dataset
- Your source code or repository contents (only the commit hash, and whether there were
  uncommitted changes)
- Screenshots or system state
- Any file other than the model you point it at and what sits beside it for that model (a
  compiled engine, a converted model and its build record)

## The --offline flag

Running with `--offline` guarantees zero network calls. Results are saved to
`~/.calbredge/results/` and `~/.calbredge/runs/` and never transmitted.

```bash
calbredge run --model your_model.onnx --offline
```

`calbredge doctor` and `calbredge inspect` are read-only and never upload. `calbredge init
--example` downloads the example model from the ONNX Model Zoo and checks its SHA-256;
without `--example`, `init` makes no network calls.

## Uploads

An upload happens only when you give `calbredge run` an upload URL, with `--upload` or with a
saved configuration, and do not pass `--offline`. The result is then sent over HTTPS to that
URL and nowhere else. Your model file is never transmitted: only the result (latency numbers,
the hardware description above, the model hash, power and temperature) is sent. How an
uploaded result is stored and who can see it is up to the service at that URL.

Uploads carry an API key. To save one with the URL:

```bash
calbredge config --api-key YOUR_KEY --upload-url https://<your API host>/api/benchmarks
```

This writes `~/.calbredge/config.yaml`. You can also pass the key per run with `--api-key` or
the `CALBREDGE_API_KEY` environment variable.

## Open inspection

The agent's source is in this repository: you can read exactly what runs on your hardware
before installing it. The default protocol is
`protocol/calbredge_protocol/protocols/v1.0.yaml`, and the result's fields are specified in
`docs/RESULT_SCHEMA.md`.

## Verification

`--dry-run` runs the full benchmark and prints the complete result that would be uploaded,
without sending anything:

```bash
calbredge run --model your_model.onnx --dry-run
```

For a description of the environment, suitable for auditing or a support request:

```bash
calbredge run --model your_model.onnx --dry-run --debug
```

## Reporting a vulnerability

Report a vulnerability privately, through GitHub's private vulnerability reporting (the
repository's Security tab), never in a public issue.
