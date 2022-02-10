import numpy as np

# from nn/module.py
def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col)


class Tensor:
    def __init__(self, h=2, w=2, weight=None):
        if weight is None:
            self.weight = layer_init(h, w)  # layer
        else:
            self.weight = weight
        self.forward = None  # to save forward pass from previous layer
        self.grad = None  # d_layer
        self.trainable = True


class Activation:
    def __init__(self):
        self.grad = None

    def ReLU(self, x):
        fp = np.maximum(x, 0)
        grad = (fp > 0).astype(np.float32)
        return fp, grad

    def Sigmoid(self, xx):  # slow
        S = np.array(list(map(lambda x: 1/(1+np.exp(-x)), xx)))
        return S, np.multiply(S, (1-S))

    def Softmax(self, x):
        fp = np.divide(np.exp(x).T, np.exp(x.sum(axis=1))).T
        return fp, fp


class Sequential:
    def __init__(self, layers):
        self.model = layers

    def add(self, layer):
        self.model.append(layer)

    def forward(self, x):
        """go add graph topo"""
        for layer in self.model:
            if not isinstance(layer, Tensor):
                x, grad = layer(self, x)
                layer.grad = grad
            else:
                layer.forward = x
                x = x @ layer.weight
        return x

    def backward(self, bpass):
        for layer in self.model[::-1]:
            if not isinstance(layer, Tensor):
                bpass = np.multiply(bpass, layer.grad)
            else:
                layer.grad = layer.forward.T @ bpass
                bpass = bpass @ (layer.weight.T)
                if layer.trainable:
                    self.optim(layer)

    def compile(self, lossfn, optim):
        self.lossfn = lossfn
        self.optim = optim

    def fit(self, X, Y, epoch, batch_size=32):
        history = {"loss": [], "accuracy": []}
        for _ in range(epoch):
            samp = np.random.randint(0, len(X), size=batch_size)
            x = X[samp]
            y = Y[samp]
            yhat = self.forward(x)
            loss, gradient = self.lossfn(self, y, yhat)
            self.backward(gradient)
            history["loss"].append(loss.mean())
            history["accuracy"].append(
                (yhat.argmax(axis=1) == y).astype(np.float32).mean())
        return history

    def evaluate(self, X, Y, batch_size=32):
        accuracies = []
        assert len(X) == len(Y)
        for i in range(len(X)//batch_size):
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            yhat = self.forward(x)
            accuracy = (yhat.argmax(axis=1) == y).astype(np.float32).mean()
            accuracies.append(accuracy)
        return sum(accuracies)/len(accuracies)

    def summary(self):
        # print each layer + loss (optional)
        for layer in self.model:
            print("type  ", layer.shape)
        if self.lossfn is not None:
            print("loss: name")  # to be implement name of each function

    def save(self, name="model_0.txt"):
        """
        dict of arrays
        file format needs to be changed, ref: fetch_it.py
        """
        import os
        fp = os.path.join("record", name)  # for instance: model_0
        # rename to avoid collision and overwritten
        print("filename %s" % fp)
        # return
        with open(fp, "x") as f:
            for layer in self.model:
                f.write("\nlayer %d\n\n" % (self.model.index(layer)+1))
                if isinstance(layer, Tensor):
                    for row in layer.weight:
                        for ele in row:
                            f.write(str(ele)+" ")
                        f.write("\n\n")
            f.close()

class Loss:
    def mse(self, y, yhat, supervised=True, num_class=10):
        """read num_class when supervised"""
        if supervised:
            label = np.zeros((len(y), num_class), dtype=np.float32)
            label[range(label.shape[0]), y] = 1
            y = label
        loss = np.square(np.subtract(yhat, y))  # vector form
        diff = 2*np.subtract(yhat, y)/(y.shape[-1])
        return loss, diff

    def crossentropy(self, y, yhat, supervised=True, num_class=10):
        """softmax + cross entropy loss"""
        label = np.zeros((len(y), num_class), dtype=np.float32)
        label[range(label.shape[0]), y] = 1
        los = (-yhat + np.log(np.exp(yhat).sum(axis=1)).reshape((-1, 1)))
        loss = (label*los)  # .mean(axis=1)
        d_out = label/len(y)
        diff = -d_out + np.exp(-los)*d_out.sum(axis=1).reshape((-1, 1))
        return loss, diff


class Optimizer:
    # void func's
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    # call() is from topo.py
    def call(self,layer,opt):
        if layer.child is not None:
            opt(layer.child)

    def SGD(self, layer):
        layer.weight -= self.learning_rate * layer.grad
        # call() usage
        # update the params in model recursively
        self.call(layer, self.SGD)


    def Adam(self, layer, b1=0.9, b2=0.999, eps=1e-8):
        m, v, t = 0, 0, 0
        tmp = 0  # to record weight change
        while np.abs(((tmp-layer.weight).sum())/layer.weight.sum()) > 1e-1:
            t += 1
            g = layer.grad
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            mhat = m/(1-b1**t)
            vhat = v/(1-b2**t)
            # prev weight
            tmp = layer.weight
            # current weight
            layer.weight -= self.learning_rate*mhat/(vhat**0.5+eps)


# from topo.py
class Activations:
    def __init__(self):
        self.child = None
        self.grad = None
        self.trainable = False

    def backwards(self, bpass):
        bpass = np.multiply(self.grad, bpass)
        if self.child is not None:
            self.child.backwards(bpass)

class ReLU(Activations):
    def __call__(self, layer):
        self.child = layer
        return layer

    def forwards(self, x):
        if self.child is not None:
            x = self.child.forwards(x)
        out = np.maximum(x, 0)
        self.grad = (out > 0).astype(np.float32)
        return out

class Linear:
    def __init__(self, h=1, w=1, weight=None):
        if weight is None:
            self.weight = layer_init(h, w)
        else:
            self.weight = weight
        # topo
        self.child = None
        self.forward = None  # save forward pass from previous layer
        self.grad = None  # d_layer
        self.trainable = True

    def __call__(self, layer):
        self.child = layer
        return layer

    def forwards(self, ds):
        if self.child is not None:
            ds = self.child.forwards(ds)
        if self.trainable:
            self.forward = ds
        return ds @ self.weight

    def backwards(self, bpass, optim):
        if self.trainable:
            self.grad = self.forward.T @ bpass
            optim(self)
        bpass = bpass @ (self.weight.T)
        if self.child is not None:
            self.child.backwards(bpass, optim)

class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=False):
      self.filters = filters
      self.ks = kernel_size

      # fast in built-in, consider merge in layer_init()
      weight = np.random.uniform(-1., 1.,size=(filters,kernel_size,kernel_size))/np.sqrt(kernel_size**2)
      self.weight = weight.astype(np.float32)

      self.st = stride
      self.padding = padding  # bool

      # similar to Tensor, can be replaced by inheriting from class Layer
      self.forward = None
      self.grad = np.zeros(weight.shape,dtype=np.float32)
      self.trainable = True

      self.child = None

    def __repr__(self):
      return f"filters: {self.filters}, ks: {self.ks}"

    def __call__(self,layer):
      self.child = layer
      return layer

    def forwards(self, x): 
      ks = self.ks
      st = self.st
      # output[0]: batchsize -> No. of filter
      # not the real conv, which doesn't require padding
      # remove padding when forward, 
      # and add padding when backward
      out = np.zeros((self.filters,x.shape[1],x.shape[2]))
      for r in range(self.filters):
        for k in range(0, (x.shape[1]-ks) + 1, st):
          for m in range(0, (x.shape[2]-ks) + 1, st):
            tmp = x[:, k:k+ks, m:m+ks]
            ret = np.multiply(self.weight[r], tmp)
            out[r, k, m] = ret.sum()

      self.forward = out 
      return out 

    def backwards(self,bpass):
      # d_weight = forward.T @ bpass
      ks = self.ks
      st = self.st
      rk = self.forward.shape[1]
      rm = self.forward.shape[2]

      for r in range(self.filters):
        tmpgrad = self.forward[r].T @ bpass[r] 
        tmpout = np.zeros(self.weight[0].shape)
        for k in range(0, rk, st):
          for m in range(0, rm, st):
            tmpout += tmpgrad[k:ks+k, m:ks+k].sum()
        self.grad[r] += tmpout


