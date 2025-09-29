import numpy as np

class SimpleConv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(out_channels, in_channels, kernel_size) * 0.01
        self.b = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        batch_size, _, width = x.shape
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.out = np.zeros((batch_size, self.out_channels, out_width))
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
        for n in range(batch_size):
            for oc in range(self.out_channels):
                for ow in range(out_width):
                    start = ow * self.stride
                    end = start + self.kernel_size
                    self.out[n, oc, ow] = np.sum(x_padded[n, :, start:end] * self.W[oc]) + self.b[oc]
        return self.out

    def backward(self, d_out, lr=0.01):
        batch_size, _, out_width = d_out.shape
        dx = np.zeros_like(self.x)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        x_padded = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding)))
        dx_padded = np.zeros_like(x_padded)
        for n in range(batch_size):
            for oc in range(self.out_channels):
                for ow in range(out_width):
                    start = ow * self.stride
                    end = start + self.kernel_size
                    dW[oc] += d_out[n, oc, ow] * x_padded[n, :, start:end]
                    db[oc] += d_out[n, oc, ow]
                    dx_padded[n, :, start:end] += d_out[n, oc, ow] * self.W[oc]
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding]
        else:
            dx = dx_padded
        self.W -= lr * dW
        self.b -= lr * db
        return dx

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 3, 8)
    conv = SimpleConv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    out = conv.forward(x)
    print("Forward output shape:", out.shape)
    d_out = np.random.randn(*out.shape)
    dx = conv.backward(d_out)
    print("Backward dx shape:", dx.shape)
