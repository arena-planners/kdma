# kdma

Arena wrapper for the **KDMA** (Knowledge Distillation Multi-Agent) navigator. Adapted from [xupei0610/KDMA](https://github.com/xupei0610/KDMA) (IROS 2021).

## Run

```sh
arena launch robot.mobile:=drl robot.mobile.planner:=kdma
```

Requires a global plan. Defaults to `nav2/navfn`.

## Files

- `planner.py`: entry point. Goal-aligned ego-frame obs + pedestrian relative state produces `[v, omega]`.
- `networks.py`: vendored `Policy` / `ActorNetwork` / `CriticNetwork`.
- `planner.yaml`: observation manifest.
- `weights.yaml`: pulls `kdma_policy.ckpt` from [arena-rosnav/kdma](https://huggingface.co/arena-rosnav/kdma) on `arena feature planners add kdma`.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for code provenance.

## License

MIT (inherited from upstream).
