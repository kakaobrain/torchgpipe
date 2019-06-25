---
name: "\U0001F41E Bug report"
about: Submit a bug report to help us improve torchgpipe

---

# 🐞 Bug
A clear and concise description of what the bug is.

## Code that reproduces
```python
# Paste here the code that reproduces the bug.
# Keep the code compact to focus on the actual problem.
```

Paste here output from the above code:
```
```

## Environment
Run the below Python code and paste the output:
```python
import os, platform, torch, torchgpipe

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_properties(0))
    print('Number of GPUs:', torch.cuda.device_count())
    print('CUDA:', torch.version.cuda)
    print('cuDNN:', torch.backends.cudnn.version())
else:
    print('No GPUs')

print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('torchgpipe:', torchgpipe.__version__)

try:
    with open(os.path.join(torchgpipe.__path__[0], '../.git/HEAD')) as f:
        print('torchgpipe.git:', f.read())
except FileNotFoundError:
    pass
```

Paste here:
```
```

## Additional context
Add any other context about the problem here.
