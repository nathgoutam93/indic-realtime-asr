# Install deps

CPU environment:
```bash
pip install -r requirements/base.txt
pip install -r requirements/cpu.txt
```
CUDA environment:
```bash
pip install -r requirements/base.txt
pip install -r requirements/gpu.txt
```
# Run 
```
python3 run.py 
```

You can set runtime configuration either in your shell environment or in a local `.env` file.
See [`env.example`](/home/gnrc/ai4bharat/env.example) for the full list of supported variables, defaults, and comments.
