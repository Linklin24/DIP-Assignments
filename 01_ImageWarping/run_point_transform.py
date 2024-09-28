import cv2
import numpy as np
import gradio as gr


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

# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8, nx=900, ny=300):
    """ 
    Return
    ------
        A deformed image.
    """

    warped_image = np.zeros_like(image)
    rows, cols = image.shape[0:2]
    n = len(target_pts)  # 控制点对个数
    source_pts = source_pts[0:n]

    # 控制点对坐标和实际像素坐标相反
    source_pts = source_pts[:, ::-1]
    target_pts = target_pts[:, ::-1]

    # 交换控制点和目标点, 便于后续插值
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    # 生成矩形网格, nx和ny分别为x和y方向上的网格点数量
    xs = np.linspace(0, rows-1, nx)
    ys = np.linspace(0, cols-1, ny)
    height, width = len(xs), len(ys)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing='ij')

    # 计算网格点坐标v
    v = np.stack((x_grid, y_grid), axis=-1) # [height, weight, 2]

    # 计算权重w_i
    weight = 1 / np.linalg.norm(v.reshape(height, width, 1, 2) - source_pts + eps, axis=-1) ** (2 * alpha)  # [height, width, n]

    # 计算p_star和q_star
    p_star = weight @ source_pts / np.sum(weight, axis=-1, keepdims=True)  # [height, width, 2]
    q_star = weight @ target_pts / np.sum(weight, axis=-1, keepdims=True)  # [height, width, 2]

    # 计算p_hat和q_hat
    p_hat = source_pts - p_star.reshape(height, width, 1, 2)  # [height, width, n, 2]
    q_hat = target_pts - q_star.reshape(height, width, 1, 2)  # [height, width, n, 2]

    # 计算矩阵A_i, A_i的元素可以写为[[a11, a12], [-a12, a11]], 因此只需计算a11和a12
    v_minus_p_star = (v - p_star).reshape(height, width, 1, 2)  # [height, width, 1, 2]
    a11 = p_hat[..., 0] * v_minus_p_star[..., 0] + p_hat[..., 1] * v_minus_p_star[..., 1]  # [height, width, n]
    a12 = p_hat[..., 0] * v_minus_p_star[..., 1] - p_hat[..., 1] * v_minus_p_star[..., 0]  # [height, width, n]
    weight = weight.reshape(height, width, n, 1)  # [height, width, n, 1]
    mat_A = (weight * np.stack((a11, a12, -a12, a11), axis=-1)).reshape(height, width, n, 2, 2)  # [height, width, n, 2, 2]

    # 计算f
    f = np.sum(q_hat.reshape(height, width, n, 1, 2) @ mat_A, axis=(2, 3))  # [height, width, 2]
    v_minus_p_star = v_minus_p_star.reshape(height, width, 2)  # [height, width, 2]
    f = np.linalg.norm(v_minus_p_star, axis=-1, keepdims=True) * f / np.linalg.norm(f, axis=-1, keepdims=True) + q_star  # [height, width, 2]

    # 对f插值得到最终结果
    f = np.float32(f)
    warped_image = cv2.remap(image, f[..., 1], f[..., 0], cv2.INTER_LINEAR)
    warped_image = cv2.resize(warped_image, (cols, rows))

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
