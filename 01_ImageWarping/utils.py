import numpy as np

def pixels_in_polygon(corners):
    n = corners.shape[0]
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    grid = np.meshgrid(xs, ys)
    points = np.stack([p.ravel() for p in grid], axis=1).astype(int)
    pixels = []

    for p in points:
        inside = False
        for i in range(n):
            start = corners[i - 1]
            end = corners[i]
            if (((end[1] > p[1]) != (start[1] > p[1])) and
                (p[0] < ((start[0] - end[0]) * (p[1] - end[1]) / (start[1] - end[1]) + end[0]))):
                inside = not inside
        if inside:
            pixels += [p]

    return pixels
