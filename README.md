# Supervised-unsupervised CIFAR10
A cifar10 classifying network combining supervised and unsupervised learning.  
Written in pytorch.

## Run
### Test  
For testing, run the following command.  
```python
python src/test.py -w weight/good.pth
```

The dataset is automatically downloaded.

### Train  
For training, run the following command.  
```
python src/train.py
```
The number of the training epochs and the size of minibatch can be specified by adding like `-e 100 -b 100`.

## Accuracy
The result on the test data.  

Overall accuracy: 86.72%  

airplane: 88.1%  
automobile: 94.6%  
bird: 79.6%  
cat: 82.7%  
deer: 83.5%  
dog: 78.1%  
frog: 88.7%  
horse: 89.5%  
ship: 92.6%  
truck: 89.8%  

  
## (My) Environment
- Pytorch (1.6.0)
- Torchvision (0.7.0)
- numpy (1.18.5)

If you have installed docker, you can setup this environment by building the official docker image of pytorch.
```
docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
```
https://hub.docker.com/r/pytorch/pytorch/tags

When you start a container from the docker image, 
it is needed to mount this repository on the staring container.  

For example, I do the following command in the repository.
```
docker run --gpus=all --rm -v $(pwd)/:/workspace -it <IMAGE ID>
```
( Remove `--gpus=all` if your computer does not have a GPU. )