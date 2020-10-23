# rigl-pytorch

An open source implementation of Google Research's paper [Rigging the Lottery: Making All Tickets Winners (RigL)](https://github.com/google-research/rigl) in PyTorch as versatile, simple, and fast as possible.

## Other Implementations:
- View the TensorFlow implementation (also the original) by Utku Evci @ Google Brain [here](https://github.com/google-research/rigl)!
- Additionally, it is also implemented in [vanilla python](https://evcu.github.io/ml/sparse-micrograd/) and [graphcore](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl).

## Usage:
- I have provided an imagenet training script that was slightly modified to add RigL's functionality. It adds a few parser statements, and only 3 required lines of RigL code usage to work! View the original training script [here](https://github.com/pytorch/examples/tree/master/imagenet) and the RigL version [here](https://github.com/McCrearyD/rigl-pytorch/blob/master/train_imagenet_rigl.py). :)

- You can use the pruning power of RigL by adding 3 lines of code to **your already existing training script**! Here is how:

```python
# first, create your model
model = ...

# create your dataset/dataloader
dataset = ...
dataloader = ...

# define your optimizer (ex. SGD)
optimizer = ...


# RigL runs best when you allow RigL's topology modifications to run for 75% of the total training iterations (batches)
# so, let's calculate T_end according to this
epochs = 100
total_iterations = len(datloader) * epochs
T_end = int(0.75 * total_iterations)

# now, create the RigLScheduler object
pruner = RigLScheduler(model,                  # model you created
                       dense_allocation=0.1,   # a float between 0 and 1 that designates how sparse you want the network to be (0.1 dense_allocation = 90% sparse)
                       T_end=T_end,            # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                       delta=args.delta,       # delta hyperparam within the paper (recommended = 100)
                       alpha=args.alpha,       # alpha hyperparam within the paper (recommended = 0.3)
                       static_topo=False)      # if True, the topology will be frozen, in other words RigL will not do it's job (for debugging)
                       
... more code ...

for data in dataloader:
    # do forward pass, calculate loss, etc.
    ...
    
    # instead of calling optimizer.step(), wrap it as such:
    if pruner():
        # this block of code will execute according to the given hyperparameter schedule
        # in other words, optimizer.step() is not called after a RigL step
        optimizer.step()
```
