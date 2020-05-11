import numpy as np

def cross_entropy(probs,y_true):
    true_vals = np.squeeze(np.eye(10)[y_true])
    entropy = -np.sum(true_vals * np.log(probs))
    return (entropy)

def im2col(input_data, filter_h, filter_w, stride=1):
    N, C, H, W = input_data.shape
    out_h = (H - filter_h)//stride + 1
    out_w = (W - filter_w)//stride + 1

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # convert a w*h matrix into a column vector for dot products
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1):
    N, C, H, W = input_shape
    out_h = (H - filter_h)//stride + 1
    out_w = (W - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + stride - 1, W + stride - 1))

    # convert column back to matrix to represent image
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, :H, :W]