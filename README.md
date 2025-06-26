# MLR Take home

Setup:
```bash
$ git submodule update --init --recursive
$ pip install uv
$ uv sync
$ source .venv/bin/activate
```

The repo contains a simple implementation of the K-norm filter, an eviction policy that uses the norm of the keys. You can run evaluations with the LM eval harness.