# Semi-supervised single image dehazing
Codes for Semi-supervised single image dehazing.
## dependency
* ```pytorch >= 1.0 ```
* ```visdom ```

## Dataset make
Make you dataset by:
1. synthetic images: align two images (hazy(HxWxC), clean(HxWxC)) into one image (Hx2WxC). To be noted that H and W should be multiple of 8. Put them (~6000 images) in ```./datasets/dehazing/train```. 
2. real hazy images: put them(~1000 images) in ```./datasets/dehazing/unlabeled```
3. test images: align the same as 1. and put them in ```./datasets/dehazing/test```

## Train
The you can train the model by:
```
python  train.py  --dataroot ./datasets/dehazing --name run_id_1 --learn_residual  --display_freq 100 --print_freq 100 --display_port 8097 --which_model_netG EDskipconn --lambda_vgg 10 --lambda_mse 1000 --lambda_ssim 0 --niter 90 --niter_decay 0 --fineSize 256 --no_html --lambda_DC 1e-4 --lambda_TV 1e-4 --gpu_id 0 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --semi --patch_size 7 --batch_size 2
```
Monitor the traning process via visdom by:
```
python -m visdom.server -port 8097
```
## Test
You can test you model on RESIDE SOTS [dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-v0).
```
python test.py --dataset ./datasets/dehazing --name run_id_1  --learn_residual --which_model_netG EDskipconn --gpu_id 0 --no_html --which_epoch latest
```
## Refenrece:
```
@article{li2019semi,
  title={Semi-supervised image dehazing},
  author={Li, Lerenhan and Dong, Yunlong and Ren, Wenqi and Pan, Jinshan and Gao, Changxin and Sang, Nong and Yang, Ming-Hsuan},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={2766--2779},
  year={2019},
  publisher={IEEE}
}
```
