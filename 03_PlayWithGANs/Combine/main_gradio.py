import os
import os.path as osp
import collections
import numpy as np
import torch
import gradio as gr
from PIL import Image
import face_alignment

import dnnlib
from gradio_utils import draw_points_on_image
from viz.renderer import Renderer
from gan_inv.inversion import PTI
from gan_inv.lpips import util

cache_dir = './checkpoints'
device = 'cuda'
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points

def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state

def init_images(global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    state['renderer'].init_network(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        None,  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(np.array(init_image))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state

def update_image_draw(image, points, mask, show_mask, global_state=None):
    # image_draw = draw_points_on_image(image, points)
    image_draw = Image.fromarray(np.array(image))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw

def start_draggan(global_state):
    p_in_pixels = []
    t_in_pixels = []
    valid_points = []

    # Transform the points into torch tensors
    for key_point, point in global_state["points"].items():
        try:
            p_start = point.get("start_temp", point["start"])
            p_end = point["target"]

            if p_start is None or p_end is None:
                continue

        except KeyError:
            continue

        p_in_pixels.append(p_start)
        t_in_pixels.append(p_end)
        valid_points.append(key_point)

    mask = torch.tensor(global_state['mask']).float()
    drag_mask = 1 - mask

    renderer: Renderer = global_state["renderer"]
    global_state['temporal_params']['stop'] = False
    global_state['editing_state'] = 'running'

    # reverse points order
    p_to_opt = reverse_point_pairs(p_in_pixels)
    t_to_opt = reverse_point_pairs(t_in_pixels)
    print('Running with:')
    print(f'    Source: {p_in_pixels}')
    print(f'    Target: {t_in_pixels}')
    step_idx = 0
    while True:
        if global_state["temporal_params"]["stop"] or step_idx > 50:
            break

        # do drage here!
        renderer._render_drag_impl(
            global_state['generator_params'],
            p_to_opt,  # point
            t_to_opt,  # target
            drag_mask,  # mask,
            global_state['params']['motion_lambda'],  # lambda_mask
            reg=0,
            feature_idx=5,  # NOTE: do not support change for now
            r1=global_state['params']['r1_in_pixels'],  # r1
            r2=global_state['params']['r2_in_pixels'],  # r2
            # random_seed     = 0,
            # noise_mode      = 'const',
            trunc_psi=global_state['params']['trunc_psi'],
            # force_fp32      = False,
            # layer_name      = None,
            # sel_channels    = 3,
            # base_channel    = 0,
            # img_scale_db    = 0,
            # img_normalize   = False,
            # untransform     = False,
            is_drag=True,
            to_pil=True)

        if step_idx % global_state['draw_interval'] == 0:
            print('Current Source:')
            for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                            t_to_opt):
                global_state["points"][key_point]["start_temp"] = [
                    p_i[1],
                    p_i[0],
                ]
                global_state["points"][key_point]["target"] = [
                    t_i[1],
                    t_i[0],
                ]
                start_temp = global_state["points"][key_point][
                    "start_temp"]
                print(f'    {start_temp}')

            image_result = global_state['generator_params']['image']
            image_draw = update_image_draw(
                image_result,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )
            global_state['images']['image_raw'] = image_result

        yield (
            global_state,
            step_idx,
            global_state['images']['image_show'],

            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            # enable stop button in loop
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=False),
            gr.UploadButton.update(interactive=False),
            gr.Button.update(interactive=False),

            gr.Number.update(interactive=False),
            gr.Number.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Slider.update(interactive=False),
        )

        # increate step
        step_idx += 1

    image_result = global_state['generator_params']['image']
    global_state['images']['image_raw'] = image_result
    image_draw = update_image_draw(image_result,
                                    global_state['points'],
                                    global_state['mask'],
                                    global_state['show_mask'],
                                    global_state)

    # fp = NamedTemporaryFile(suffix=".png", delete=False)
    # image_result.save(fp, "PNG")

    global_state['editing_state'] = 'add_points'

    yield (
        global_state,
        0,  # reset step to 0 after stop.
        global_state['images']['image_show'],

        gr.Button.update(interactive=True),
        gr.Button.update(interactive=True),
        gr.Button.update(interactive=True),
        gr.Button.update(interactive=True),
        # NOTE: disable stop button with loop finish
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=True),
        gr.UploadButton.update(interactive=True),
        gr.Button.update(interactive=True),

        gr.Number.update(interactive=True),
        gr.Number.update(interactive=True),
        gr.Slider.update(interactive=True),
        gr.Slider.update(interactive=True),
        gr.Slider.update(interactive=True),
    )

valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

init_pkl = 'stylegan2-ffhq-512x512'

with gr.Blocks() as app:

    global_state = gr.State({
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        "facial_landmarks": {
            # facial landmarks
        },
        'mask':
        None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
            "smile": 1.0,
            "slim": 0.9,
            "eyes": 1.5,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl
    })

    # init image
    global_state = init_images(global_state)

    with gr.Row():

        with gr.Row():

            with gr.Column(scale=3):

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='生成图片', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        seed_number = gr.Number(
                            value=global_state.value['params']['seed'],
                            interactive=True,
                            label="Seed",
                        )
                        
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='编辑图片', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                smile_button = gr.Button('微笑')
                            with gr.Column(scale=1, min_width=10):
                                slim_button = gr.Button('瘦脸')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                eyes_button = gr.Button("大眼")
                            with gr.Column(scale=1, min_width=10):
                                close_eyes_button = gr.Button("闭眼")
                        steps_number = gr.Number(value=0,
                                                 label="Steps",
                                                 interactive=False)
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                stop_button = gr.Button("结束编辑")
                            with gr.Column(scale=1, min_width=10):
                                reset_button = gr.Button("重置图片")

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='参数控制', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="迭代步长")
                        smile_parameter = gr.Slider(
                            minimum=0,
                            maximum=2,
                            step=0.1,
                            value=global_state.value['params']['smile'],
                            label="微笑参数")
                        slim_parameter = gr.Slider(
                            minimum=0.8,
                            maximum=1,
                            step=0.01,
                            value=global_state.value['params']['slim'],
                            label="瘦脸参数")
                        eyes_parameter = gr.Slider(
                            minimum=1,
                            maximum=3,
                            step=0.1,
                            value=global_state.value['params']['eyes'],
                            label="大眼参数")
                        
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='自选图片', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                custom_image = gr.UploadButton(label="上传自定义图片",
                                                               file_types=['.png', '.jpg', '.jpeg'])
                            with gr.Column(scale=1, min_width=10):
                                reset_custom_image = gr.Button('重置自定义图片')

            with gr.Column(scale=10):
                image_show = gr.Image(
                    value=global_state.value['images']['image_show']).style(
                        height=768, width=768)
                
    gr.Markdown("""
            ## 使用说明

            1. 使用自定义图片进行编辑需要花费较长时间进行 GAN Inversion.
            2. 大眼参数对应于眼睛的放大系数, 瘦脸参数对应于脸部的缩小系数.
            3. 第一次使用程序对肖像进行编辑时可能花费较长时间(10s).
            """)
        

    def on_change_update_image_seed(seed, global_state):
        """Function to handle generation seed change.
        1. Set seed to global_state
        2. Re-init images and clear all states
        """

        global_state["params"]["seed"] = int(seed)
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    seed_number.change(
        on_change_update_image_seed,
        inputs=[seed_number, global_state],
        outputs=[global_state, image_show],
    )

    def on_click_stop(global_state):
        """Function to handle stop button is clicked.
        1. send a stop signal by set global_state["temporal_params"]["stop"] as True
        2. Disable Stop button
        """
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    stop_button.click(on_click_stop,
                      inputs=[global_state],
                      outputs=[global_state, stop_button])

    def on_click_reset(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    reset_button.click(
        on_click_reset,
        inputs=[global_state],
        outputs=[global_state, image_show],
    )
    
    def on_change_lr(lr, global_state):
        if lr == 0:
            print('lr is 0, do nothing.')
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)
        return global_state

    lr_number.change(
        on_change_lr,
        inputs=[lr_number, global_state],
        outputs=[global_state],
    )

    def on_change_smile_parameter(smile_parameter, global_state):
        global_state["params"]["smile"] = smile_parameter

        return global_state

    smile_parameter.change(
        on_change_smile_parameter,
        inputs=[smile_parameter, global_state],
        outputs=[global_state],
    )

    def on_change_slim_parameter(slim_parameter, global_state):
        global_state["params"]["slim"] = slim_parameter

        return global_state

    slim_parameter.change(
        on_change_slim_parameter,
        inputs=[slim_parameter, global_state],
        outputs=[global_state],
    )

    def on_change_eyes_parameter(eyes_parameter, global_state):
        global_state["params"]["eyes"] = eyes_parameter

        return global_state

    eyes_parameter.change(
        on_change_eyes_parameter,
        inputs=[eyes_parameter, global_state],
        outputs=[global_state],
    )

    def on_click_inverse_custom_image(custom_image,global_state):
        print('inverse GAN')

        if isinstance(global_state, gr.State):
            state = global_state.value
        else:
            state = global_state

        state['renderer'].init_network(
            state['generator_params'],  # res
            valid_checkpoints_dict[state['pretrained_weight']],  # pkl
            state['params']['seed'],  # w0_seed,
            None,  # w_load
            state['params']['latent_space'] == 'w+',  # w_plus
            'const',
            state['params']['trunc_psi'],  # trunc_psi,
            state['params']['trunc_cutoff'],  # trunc_cutoff,
            None,  # input_transform
            state['params']['lr']  # lr,
        )

        percept = util.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True
        )

        image = Image.open(custom_image.name)

        pti = PTI(global_state['renderer'].G,percept)
        inversed_img, w_pivot = pti.train(image,state['params']['latent_space'] == 'w+')
        inversed_img = (inversed_img[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        inversed_img = inversed_img.cpu().numpy()
        inversed_img = Image.fromarray(inversed_img)
        global_state['images']['image_show'] = Image.fromarray(np.array(inversed_img))

        global_state['images']['image_orig'] = inversed_img
        global_state['images']['image_raw'] = inversed_img
            
        global_state['mask'] = np.ones((inversed_img.size[1], inversed_img.size[0]),
                                    dtype=np.uint8)
        global_state['generator_params'].image = inversed_img
        global_state['generator_params'].w = w_pivot.detach().cpu().numpy()
        global_state['renderer'].set_latent(w_pivot,global_state['params']['trunc_psi'],global_state['params']['trunc_cutoff'])

        del percept
        del pti
        print('inverse end')

        return global_state, global_state['images']['image_show'], gr.Button.update(interactive=True)
    
    custom_image.upload(
        on_click_inverse_custom_image,
        inputs=[custom_image, global_state],
        outputs=[global_state, image_show, reset_custom_image])
    
    def on_reset_custom_image(global_state):
        if isinstance(global_state, gr.State):
            state = global_state.value
        else:
            state = global_state
        clear_state(state)
        state['renderer'].w = state['renderer'].w0.detach().clone()
        state['renderer'].w.requires_grad = True
        state['renderer'].w_optim = torch.optim.Adam([state['renderer'].w], lr=state['renderer'].lr)
        state['renderer']._render_drag_impl(state['generator_params'],
                                            is_drag=False,
                                            to_pil=True)

        init_image = state['generator_params'].image
        state['images']['image_orig'] = init_image
        state['images']['image_raw'] = init_image
        state['images']['image_show'] = Image.fromarray(np.array(init_image))
        state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                                dtype=np.uint8)
        return state, state['images']['image_show']
    
    reset_custom_image.click(
        on_reset_custom_image,
        inputs=[global_state],
        outputs=[global_state, image_show])
        
    def on_click_smile(global_state):
        clear_state(global_state)

        image_raw = global_state['images']['image_raw']
        landmarks = fa.get_landmarks((np.array(image_raw)))[-1]

        left_delta = landmarks[49] - landmarks[59]
        right_delta = landmarks[53] - landmarks[55]
        delta = np.vstack([left_delta, right_delta])
        
        points = global_state["points"]
        parameter = global_state["params"]["smile"]
        stay = landmarks[[51, 56, 57]].astype(int)
        start = landmarks[[48, 54]].astype(int)
        target = (start + delta * parameter).astype(int)
        for i in range(2):
            points[i] = {'start': start[i], 'target': target[i]}
        for i in range(2, 5):
            points[i] = {'start': stay[i-2], 'target': stay[i-2]}

        yield from start_draggan(global_state)

    smile_button.click(
        on_click_smile,
        inputs=[global_state],
        outputs=[
            global_state,
            steps_number,
            image_show,
            # form_download_result_file,
            # >>> buttons
            smile_button,
            slim_button,
            eyes_button,
            close_eyes_button,
            stop_button,
            reset_button,
            custom_image,
            reset_custom_image,
            # <<< buttonm
            # >>> inputs comps
            seed_number,
            lr_number,
            smile_parameter,
            slim_parameter,
            eyes_parameter,
        ],
    )

    def on_click_slim(global_state):
        clear_state(global_state)

        image_raw = global_state['images']['image_raw']
        landmarks = fa.get_landmarks((np.array(image_raw)))[-1]
        
        face_center = np.mean(landmarks[0:17], axis=0, keepdims=True).repeat(9, 0)
        
        points = global_state["points"]
        parameter = global_state["params"]["slim"]
        start = landmarks[4:13].astype(int)
        target = (face_center + (start - face_center) * parameter).astype(int)
        for i in range(9):
            points[i] = {'start': start[i], 'target': target[i]}

        yield from start_draggan(global_state)

    slim_button.click(
        on_click_slim,
        inputs=[global_state],
        outputs=[
            global_state,
            steps_number,
            image_show,
            # form_download_result_file,
            # >>> buttons
            smile_button,
            slim_button,
            eyes_button,
            close_eyes_button,
            stop_button,
            reset_button,
            custom_image,
            reset_custom_image,
            # <<< buttonm
            # >>> inputs comps
            seed_number,
            lr_number,
            smile_parameter,
            slim_parameter,
            eyes_parameter,
        ],
    )

    def on_click_eyes(global_state):
        clear_state(global_state)

        image_raw = global_state['images']['image_raw']
        landmarks = fa.get_landmarks((np.array(image_raw)))[-1]
        
        left_eye_center = np.mean(landmarks[36:42], axis=0, keepdims=True).repeat(4, 0)
        right_eye_center = np.mean(landmarks[42:48], axis=0, keepdims=True).repeat(4, 0)
        eye_center = np.vstack([left_eye_center, right_eye_center])
        
        points = global_state["points"]
        parameter = global_state["params"]["eyes"]
        point_slice = np.r_[37:39, 40:42, 43:45, 46:48]
        start = landmarks[point_slice].astype(int)
        target = (eye_center + (start - eye_center) * parameter).astype(int)
        for i in range(8):
            points[i] = {'start': start[i], 'target': target[i]}

        yield from start_draggan(global_state)

    eyes_button.click(
        on_click_eyes,
        inputs=[global_state],
        outputs=[
            global_state,
            steps_number,
            image_show,
            # form_download_result_file,
            # >>> buttons
            smile_button,
            slim_button,
            eyes_button,
            close_eyes_button,
            stop_button,
            reset_button,
            custom_image,
            reset_custom_image,
            # <<< buttonm
            # >>> inputs comps
            seed_number,
            lr_number,
            smile_parameter,
            slim_parameter,
            eyes_parameter,
        ],
    )

    def on_click_close_eyes(global_state):
        clear_state(global_state)

        image_raw = global_state['images']['image_raw']
        landmarks = fa.get_landmarks((np.array(image_raw)))[-1]
        
        left_eye_center = np.mean(landmarks[36:42], axis=0, keepdims=True).repeat(4, 0)
        right_eye_center = np.mean(landmarks[42:48], axis=0, keepdims=True).repeat(4, 0)
        eye_center = np.vstack([left_eye_center, right_eye_center])
        
        points = global_state["points"]
        point_slice = np.r_[37:39, 40:42, 43:45, 46:48]
        start = landmarks[point_slice].astype(int)
        for i in range(8):
            points[i] = {'start': start[i], 'target': eye_center[i]}

        yield from start_draggan(global_state)

    close_eyes_button.click(
        on_click_close_eyes,
        inputs=[global_state],
        outputs=[
            global_state,
            steps_number,
            image_show,
            # form_download_result_file,
            # >>> buttons
            smile_button,
            slim_button,
            eyes_button,
            close_eyes_button,
            stop_button,
            reset_button,
            custom_image,
            reset_custom_image,
            # <<< buttonm
            # >>> inputs comps
            seed_number,
            lr_number,
            smile_parameter,
            slim_parameter,
            eyes_parameter,
        ],
    )

gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch()
