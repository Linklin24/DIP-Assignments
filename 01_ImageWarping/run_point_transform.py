import cv2
import numpy as np
import gradio as gr
from scipy.interpolate import interp2d

from utils import pixels_in_polygon


# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 计算正交补向量
def orth(vec):
    return np.array([-vec[1], vec[0]])

# 对网格进行插值得到变换后的图像
def grid_to_image(image, x, y, grid_x, grid_y):
    rows, cols = image.shape[0:2]
    warped_image = np.zeros_like(image)
    xs, ys = np.meshgrid(x, y)

    nx, ny = len(x), len(y)
    for i in range(nx-1):
        for j in range(ny-1):
            old_quad = np.array([[xs[i, j], ys[i, j]], [xs[i+1, j], ys[i+1, j]],
                                [xs[i+1, j+1], ys[i+1, j+1]], [xs[i, j+1], ys[i, j+1]]])
            new_quad = np.array([[grid_x[i, j], grid_y[i, j]], [grid_x[i+1, j], grid_y[i+1, j]],
                                [grid_x[i+1, j+1], grid_y[i+1, j+1]], [grid_x[i, j+1], grid_y[i, j+1]]])
            pixels = np.array(pixels_in_polygon(new_quad))

            if (pixels.shape[0] == 0):
                continue

            px, py = pixels[:, 0], pixels[:, 1]
            inside_picture = ((px >= 0) & (px < cols) & (py >= 0) & (py < rows))
            pixels = pixels[inside_picture]

            if (pixels.shape[0] == 0):
                continue

            new_x_coords, new_y_coords = new_quad[:, 0], new_quad[:, 1]
            old_x_coords, old_y_coords = old_quad[:, 0], old_quad[:, 1]
            interpx = interp2d(new_x_coords, new_y_coords, old_x_coords)
            interpy = interp2d(new_x_coords, new_y_coords, old_y_coords)
            estimated_old_points = np.array([np.squeeze([interpx(x, y), interpy(x, y)]) for x, y in pixels])
            nearest_old_pixels = np.round(estimated_old_points).astype(int)
            nearest_old_pixels[:, 0] = np.clip(nearest_old_pixels[:, 0], 0, cols-1)
            nearest_old_pixels[:, 1] = np.clip(nearest_old_pixels[:, 1], 0, rows-1)

            old_colors = np.empty((nearest_old_pixels.shape[0], 3))

            valid_x, valid_y = [A.ravel() for A in np.split(nearest_old_pixels, 2, axis=1)]
            old_colors = image[valid_y, valid_x]

            pixels_idx = tuple(np.array([A.ravel() for A in np.split(pixels, 2, axis=1)][::-1]))
            warped_image[pixels_idx] = old_colors
    
    return warped_image

# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, nx=50, ny=50):
    """ 
    Return
    ------
        A deformed image.
    """
    
    # warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping

    warped_image = np.zeros_like(image)
    rows, cols = image.shape[0:2]
    p_num = len(target_pts)
    source_pts = source_pts[0:p_num]

    weight = np.zeros(p_num)

    xs = np.linspace(0, rows-1, nx)
    ys = np.linspace(0, cols-1, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for r in range(len(xs)):
        for c in range(len(ys)):
            v = np.array([xs[r], ys[c]])

            if np.any(np.all(v == source_pts, axis=1)):
                idx = np.where(np.all(v == source_pts, axis=1))[0]
                warped_image[target_pts[idx, 1], target_pts[idx, 0]] = image[xs[r], ys[c]]
                continue

            for i in range(p_num):
                weight[i] = 1 / np.linalg.norm(v - source_pts[i]) ** (2 * alpha)

            p_star = np.zeros(2)
            q_star = np.zeros(2)
            p_star[0] = np.average(source_pts[:, 0], weights=weight)
            p_star[1] = np.average(source_pts[:, 1], weights=weight)
            q_star[0] = np.average(target_pts[:, 0], weights=weight)
            q_star[1] = np.average(target_pts[:, 1], weights=weight)

            p_hat = source_pts - p_star
            q_hat = target_pts - q_star

            f = np.zeros(2)
            for i in range(p_num):
                mat_A = weight[i] * np.vstack((p_hat[i], -orth(p_hat[i]))) @ np.vstack((v-p_star, -orth(v-p_star))).T
                f += q_hat[i] @ mat_A
            
            f = f / np.linalg.norm(f) * np.linalg.norm(v - p_star) + q_star
            if 0 <= f[0] and f[0] < rows and 0 <= f[1] and f[1] < cols:
                grid_x[r, c], grid_y[r, c] = f[1], f[0]

    warped_image = grid_to_image(image, xs, ys, grid_x, grid_y)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
