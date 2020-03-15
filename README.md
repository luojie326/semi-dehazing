# semi-dehazing


Semi-supervised single image dehazing.

## dependency
* ```pytorch >= 1.0 ```
* ```visdom ```

## Dataset make
Make you dataset by:
1. synthetic datasets: align two images (hazy(HxWxC), clean(HxWxC)) into one image (Hx2WxC). To be noted that H and W should be multiple of 8. Put them in ```./datasets/dehazing/train```
2. real images: put them in ```./datasets/dehazing/unlabeled```
3. test images: align the same as 1. and put them in ```./datasets/dehazing/test```

## Train
The you can train the model by:
```
python  train.py  --dataroot ./datasets/dehazing --name run_id_1 --learn_residual  --display_freq 100 --print_freq 100 --display_port 8097 --which_model_netG EDskipconn --lambda_vgg 10 --lambda_mse 1000 --lambda_ssim 0 --niter 90 --niter_decay 0 --fineSize 256 --no_html --lambda_DC 1e-4 --lambda_TV 1e-4 --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --semi --patch_size 7 --batch_size 2
```
## Test
You can test you model on RESIDE SOTS dataset.
```
python test.py --learn_residual --which_model_netG EDskipconn --gpu_id 0 --no_html --which_epoch latest
```

RESIDE dataset:
B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang. Benchmarking single-image dehazing and beyond. IEEE Transactions on Image Processing, 2019.
