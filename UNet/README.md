```
tensorboard --logdir=UNet/Log
unet
```
If you don't want any plots for checking loss or accuracy, then don't need to run `tensorboard --logdir=UNet/Log`. `unet` is aliased as below:
```
alias unet='PYTHONPATH=/app/project python /app/project/UNet/Src/main.py'
```

