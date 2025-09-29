import numpy as np
from typing import Tuple

def calc_output_size_2d(h_in:int, w_in:int, kh:int, kw:int, pad_h:int=0, pad_w:int=0, stride_h:int=1, stride_w:int=1) -> Tuple[int,int]:
    h_out = (h_in + 2*pad_h - kh)//stride_h + 1
    w_out = (w_in + 2*pad_w - kw)//stride_w + 1
    return h_out, w_out

class Conv2d:
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Tuple[int,int],
                 stride:Tuple[int,int]=(1,1), padding:Tuple[int,int]=(0,0), lr:float=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        kh, kw = kernel_size
        self.kh, self.kw = kh, kw
        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding
        fan_in = in_channels * kh * kw
        fan_out = out_channels * kh * kw
        limit = np.sqrt(6.0/(fan_in+fan_out))
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, kh, kw)).astype(np.float64)
        self.b = np.zeros(out_channels, dtype=np.float64)
        self.lr = lr
        self.cache = {}

    def _pad(self, x):
        if self.pad_h==0 and self.pad_w==0:
            return x
        return np.pad(x, ((0,0),(0,0),(self.pad_h,self.pad_h),(self.pad_w,self.pad_w)), mode='constant')

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, C, H, W = x.shape
        x_p = self._pad(x)
        H_p, W_p = x_p.shape[2], x_p.shape[3]
        H_out, W_out = calc_output_size_2d(H, W, self.kh, self.kw, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        out = np.zeros((batch, self.out_channels, H_out, W_out), dtype=np.float64)
        for n in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        patch = x_p[n, :, si:si+self.kh, sj:sj+self.kw]
                        out[n, oc, i, j] = np.sum(patch * self.W[oc]) + self.b[oc]
        self.cache['x'] = x
        self.cache['x_p'] = x_p
        return out

    def backward(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.cache['x']
        x_p = self.cache['x_p']
        batch, C, H, W = x.shape
        _, _, H_out, W_out = delta.shape
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_x_p = np.zeros_like(x_p)
        for n in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        patch = x_p[n, :, si:si+self.kh, sj:sj+self.kw]
                        d = delta[n, oc, i, j]
                        grad_W[oc] += d * patch
                        grad_b[oc] += d
                        grad_x_p[n, :, si:si+self.kh, sj:sj+self.kw] += d * self.W[oc]
        if self.pad_h==0 and self.pad_w==0:
            grad_x = grad_x_p
        else:
            grad_x = grad_x_p[:, :, self.pad_h:-self.pad_h if self.pad_h>0 else None, self.pad_w:-self.pad_w if self.pad_w>0 else None]
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return grad_W, grad_b, grad_x

class MaxPool2D:
    def __init__(self, pool_size:Tuple[int,int]=(2,2), stride:Tuple[int,int]=None):
        self.ph, self.pw = pool_size
        if stride is None:
            self.stride_h, self.stride_w = pool_size
        else:
            self.stride_h, self.stride_w = stride
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, C, H, W = x.shape
        H_out, W_out = calc_output_size_2d(H, W, self.ph, self.pw, 0, 0, self.stride_h, self.stride_w)
        out = np.zeros((batch, C, H_out, W_out), dtype=x.dtype)
        argmax = {}
        for n in range(batch):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        patch = x[n, c, si:si+self.ph, sj:sj+self.pw]
                        flat_idx = np.argmax(patch)
                        out[n, c, i, j] = patch.flatten()[flat_idx]
                        argmax[(n,c,i,j)] = flat_idx
        self.cache['argmax'] = argmax
        self.cache['x_shape'] = x.shape
        return out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        batch, C, H_out, W_out = delta.shape
        H_in, W_in = self.cache['x_shape'][2], self.cache['x_shape'][3]
        grad_x = np.zeros(self.cache['x_shape'], dtype=delta.dtype)
        for n in range(batch):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        flat_idx = self.cache['argmax'][(n,c,i,j)]
                        local = np.unravel_index(flat_idx, (self.ph, self.pw))
                        grad_x[n, c, si+local[0], sj+local[1]] += delta[n, c, i, j]
        return grad_x

class AveragePool2D:
    def __init__(self, pool_size:Tuple[int,int]=(2,2), stride:Tuple[int,int]=None):
        self.ph, self.pw = pool_size
        if stride is None:
            self.stride_h, self.stride_w = pool_size
        else:
            self.stride_h, self.stride_w = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, C, H, W = x.shape
        H_out, W_out = calc_output_size_2d(H, W, self.ph, self.pw, 0, 0, self.stride_h, self.stride_w)
        out = np.zeros((batch, C, H_out, W_out), dtype=x.dtype)
        for n in range(batch):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        patch = x[n, c, si:si+self.ph, sj:sj+self.pw]
                        out[n, c, i, j] = np.mean(patch)
        return out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        batch, C, H_out, W_out = delta.shape
        H_in = H_out * self.stride_h
        W_in = W_out * self.stride_w
        grad_x = np.zeros((batch, C, H_in, W_in), dtype=delta.dtype)
        area = self.ph * self.pw
        for n in range(batch):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        si = i*self.stride_h
                        sj = j*self.stride_w
                        grad_x[n, c, si:si+self.ph, sj:sj+self.pw] += delta[n, c, i, j] / area
        return grad_x

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.input_shape)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    probs = softmax(logits)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), labels] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(N), labels] -= 1
    grad /= N
    return loss, grad

class SimpleFC:
    def __init__(self, in_dim:int, out_dim:int, lr:float=0.01):
        limit = np.sqrt(6.0/(in_dim+out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        self.lr = lr
        self.cache_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_x = x
        return x.dot(self.W) + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_W = self.cache_x.T.dot(grad_out)
        grad_b = grad_out.sum(axis=0)
        grad_x = grad_out.dot(self.W.T)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return grad_x

def unit_test_conv2d_small():
    x = np.array([[[[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]]]], dtype=np.float64)  # shape (1,1,4,4)
    w = np.array([
        [[ [0.,0.,0.],[0.,1.,0.],[0.,-1.,0.] ]],   # out 0, in=1, 3x3
        [[ [0.,0.,0.],[0.,-1.,1.],[0.,0.,0.] ]]
    ], dtype=np.float64)  # shape (2,1,3,3)
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3), stride=(1,1), padding=(0,0), lr=0.0)
    conv.W = w.copy()
    conv.b = np.zeros(2)
    out = conv.forward(x)  # (1,2,2,2)
    print("Forward output:\n", out[0])
    expected = np.array([[[-4.,-4.],[-4.,-4.]], [[1.,1.],[1.,1.]]])
    assert np.allclose(out[0], expected), "forward does not match expected"
    print("Forward matches expected.")
    delta = np.array([[[[-4.,-4.],[10.,11.]], [[1.,-7.],[1.,-11.]]]], dtype=np.float64)
    grad_W, grad_b, grad_x = conv.backward(delta)
    print("Grad W:\n", grad_W)
    print("Grad b:\n", grad_b)
    print("Grad x:\n", grad_x)
    # assignment example gave some backprop results; we print grads for inspection.

def mnist_demo_conv2d_small(epochs=2, batch_size=64, max_train=2000, max_test=500):
    try:
        from tensorflow.keras.datasets import mnist
    except Exception as e:
        print("TensorFlow not available; MNIST demo skipped. To run it, install tensorflow.")
        return
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train[:max_train].reshape(-1,1,28,28)
    y_train = y_train[:max_train]
    x_test = x_test[:max_test].reshape(-1,1,28,28)
    y_test = y_test[:max_test]
    conv = Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(2,2), lr=0.01)
    pool = MaxPool2D(pool_size=(2,2))
    flatten = Flatten()
    # compute flattened dim
    dummy = conv.forward(x_train[:1])
    pooled = pool.forward(dummy)
    dim = pooled.reshape(1,-1).shape[1]
    fc = SimpleFC(dim, 10, lr=0.01)
    for ep in range(epochs):
        perm = np.random.permutation(len(x_train))
        losses = []
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i+batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            out_c = conv.forward(xb)
            out_p = pool.forward(out_c)
            feat = flatten.forward(out_p)
            logits = fc.forward(feat)
            loss, grad_logits = cross_entropy_loss(logits, yb)
            losses.append(loss)
            grad_feat = fc.backward(grad_logits)
            grad_pool = flatten.backward(grad_feat)
            # backward pooling
            grad_conv_out = pool.backward(grad_pool)
            grad_W, grad_b, grad_x = conv.backward(grad_conv_out)
        print(f"Epoch {ep+1}/{epochs}, loss={np.mean(losses):.4f}")
    out_c = conv.forward(x_test)
    out_p = pool.forward(out_c)
    feat = flatten.forward(out_p)
    logits = fc.forward(feat)
    preds = np.argmax(logits, axis=1)
    acc = (preds==y_test).mean()
    print("Test accuracy (small demo):", acc)

def param_count_conv(layer:Conv2d) -> int:
    return int(np.prod(layer.W.shape) + layer.b.size)

def output_and_param_examples():
    examples = [
        {'in':(144,144,3),'filter':(3,3,6),'stride':(1,1),'pad':(0,0)},
        {'in':(60,60,24),'filter':(3,3,48),'stride':(1,1),'pad':(0,0)},
        {'in':(20,20,10),'filter':(3,3,20),'stride':(2,2),'pad':(0,0)},
    ]
    for ex in examples:
        H,W,C = ex['in']
        kh, kw, out_ch = ex['filter']
        sh, sw = ex['stride']
        ph, pw = ex['pad']
        H_out, W_out = calc_output_size_2d(H,W,kh,kw,ph,pw,sh,sw)
        params = (C*kh*kw)*out_ch + out_ch
        print(f"Input {H}x{W}x{C} -> Out {H_out}x{W_out}x{out_ch}, params: {params}")

def le_net_like():
    layers = []
    layers.append(('conv1', Conv2d(1,6,(5,5), stride=(1,1), padding=(0,0), lr=0.01)))
    layers.append(('relu1', np.tanh))  # using tanh to mimic original; ReLU can be used instead
    layers.append(('pool1', MaxPool2D((2,2))))
    layers.append(('conv2', Conv2d(6,16,(5,5), stride=(1,1), padding=(0,0), lr=0.01)))
    layers.append(('relu2', np.tanh))
    layers.append(('pool2', MaxPool2D((2,2))))
    return layers

def short_model_summaries():
    print("AlexNet (2012): Deep conv net that popularized use of ReLU, dropout, local response normalization, and large-scale training on GPUs. Uses ~5 conv layers + 3 FC layers, large number of filters (e.g., 96,256...).")
    print("VGG16 (2014): Very uniform architecture using many stacked 3x3 conv filters and 2x2 pools. Variants like VGG16/19 show that deep networks with small filters work well. Simpler to implement but parameter heavy.")

def filter_size_notes():
    print("Why 3x3 is common: stacking two 3x3 convs has receptive field 5x5 but with fewer parameters and more non-linearities than a single 5x5. 1x1 convs: act as channel-wise linear combinations (bottleneck, feature mixing) without spatial change.")

if __name__ == "__main__":
    print("Unit test for small 2D conv (forward check)...")
    unit_test_conv2d_small()
    print("\nOutput + parameter examples for three convs:")
    output_and_param_examples()
    print("\nShort model summaries:")
    short_model_summaries()
    print("\nFilter size notes:")
    filter_size_notes()
    # Uncomment to run MNIST demo if tensorflow is available:
    # mnist_demo_conv2d_small(epochs=2)
