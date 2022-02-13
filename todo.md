
for saving the trained model:
    use pickle pkg in python
    and Dict data structure


features:
    
    add lr_finder: super-convergence by Lesile Smith: https://arxiv.org/abs/1708.07120 

    add Flatten() layer (or unsqueeze() in pytorch): basically just dimension re-arrangement
      -> can working on complex topo model (with Reshape(), and maybe Embedding())

    add LSTM: without this, this package would be very limited when applying it

    add Conv2D: tho it's slow in pure python, it's a necessity in deep learning

