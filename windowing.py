import torch as tr


def windows(length, window_size):
    for j in range(0, length, window_size):
        yield j, j + window_size - 1


if __name__ == "__main__":
    x = tr.randn(1000)
    for i, j in windows(1000, 20):
        print(i, j, tr.mean(x[i:j]))
