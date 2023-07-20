## Training & Evaluation

We train our model with 8 GPUs in 4-5 dayas on Waymo Open Motion Dataset with the following command line:

```
##Use all available GPUs by default

python train.py --n_epoch 30  --batch_size 16 --val_batch_size 128 --ddp_mode True --port 31253 --name hdgt_refine --amp bf16
```

- batch_size=16 is ok for 3090/V100/A100 while the bfloat16 is only available for 3090/A100. You could remove "--amp bf16" when your GPU is not 3090 or A100.
- Our code adopts Pytorch DDP by manually spawning mutliple processes.
- The results on validation set are around **ade6 0.5806, fde6 1.1757, mr6 0.1495**. Better results could be achieved by adopting more data augmentation tricks, heavy regularization, and longer epoch. 
- This codebase is a mix of our TPAMI paper, CoRL paper, and some modifications used during competition.

