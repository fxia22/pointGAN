# pointGAN
point set generative adversarial nets

# Dependencies

rendering requires cv2, training requires pytorch. 

On Ubuntu:
```
sudo apt-get install python-opencv
pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp27-none-linux_x86_64.whl 
pip install torchvision
```


# Download data and running

```
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
python train_gan.py
python show_gan.py --model gan/modelG_10.pth # choose your own model
```

# Sample results

![result](https://github.com/fxia22/pointGAN/blob/master/misc/output.gif?raw=true)
