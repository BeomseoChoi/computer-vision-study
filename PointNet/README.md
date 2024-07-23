```
tensorboard --logdir=PointNet/Log
pointnet
```
If you don't want any plots for checking loss or accuracy, then don't need to run `tensorboard --logdir=PointNet/Log`. `pointnet` is aliased as below:
```
alias pointnet='PYTHONPATH=/app/project python /app/project/PointNet/Src/main.py'
```

