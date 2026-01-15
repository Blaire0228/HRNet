# ****************************************************************************
#  HRNet Pose demo on KNEO Pi (KL730)
#  - 依照 GenericImageInference 範本流程
#  - 讀一張圖 → 推論 → 取出 heatmap → 畫關節點
#
#  一行指令示範：
#  python KL730HRNet.py -img people.jpg -p 0 \
#         -fw res/firmware/KL730/kp_firmware.tar \
#         -m  res/models/KL730/HRNet/models_730.nef
# ****************************************************************************

import os
import sys
import argparse

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
import cv2
import numpy as np
import math
import time

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

FEATURE_WEIGHTS = {
    "left_elbow_angle": 1.0,
    "right_elbow_angle": 1.0,
    "left_knee_angle": 0.8,
    "right_knee_angle": 0.8,
    "left_wrist_height": 1.2,
    "right_wrist_height": 1.2,
    "wrist_dist": 1.0,
    "torso_shift": 0.8,
}


def _get_model_input_hw(model_nef_descriptor):
    """從 NEF 讀模型輸入 HxW，失敗就用 256x192."""
    try:
        # [N, C, H, W]
        shape = model_nef_descriptor.models[0].input_nodes[0].tensor_shape_info.v2.shape
        in_h, in_w = int(shape[2]), int(shape[3])
        return in_h, in_w
    except Exception:
        pass

    try:
        shape = model_nef_descriptor.models[0].input_nodes[0].tensor_shape_info.shape
        in_h, in_w = int(shape[2]), int(shape[3])
        return in_h, in_w
    except Exception:
        pass

    print('[Model Input] auto-detect failed, fallback to 256x192')
    return 256, 192


def _extract_shape(obj):
    """盡量從不同 SDK 欄位拿 shape."""
    for key in ("shape_list", "shape", "output_shape", "dims"):
        if hasattr(obj, key):
            try:
                shp = getattr(obj, key)
                return tuple(int(x) for x in list(shp))
            except Exception:
                pass
    return None


def _to_numpy(inf_node, debug=False):
    """Convert Kneron InferenceFloatNodeOutput (or variants) to np.ndarray(float32)."""
    import numpy as _np

    # 已經是 ndarray 的情況
    if isinstance(inf_node, _np.ndarray):
        return inf_node.astype(_np.float32, copy=False)

    # ✅ 你這個 SDK 版本最重要的一行：結果在 .ndarray 裡
    if hasattr(inf_node, "ndarray"):
        try:
            return _np.asarray(inf_node.ndarray, dtype=_np.float32)
        except Exception:
            # 真的有問題才往下嘗試其他方式
            pass

    if debug:
        try:
            print('[DEBUG] node type:', type(inf_node))
            attrs = [a for a in dir(inf_node) if not a.startswith('__')]
            print('[DEBUG] attrs:', attrs)
        except Exception:
            pass

    # 1) methods that directly give numpy-like
    for meth in ("to_numpy", "numpy", "as_numpy"):
        if hasattr(inf_node, meth) and callable(getattr(inf_node, meth)):
            try:
                arr = getattr(inf_node, meth)()
                return _np.asarray(arr, dtype=_np.float32)
            except Exception:
                pass

    # 2) float_data + shape
    if hasattr(inf_node, 'float_data'):
        data = getattr(inf_node, 'float_data')
        arr = _np.asarray(list(data), dtype=_np.float32)
        shp = _extract_shape(inf_node)
        if shp:
            try:
                arr = arr.reshape(shp)
            except Exception:
                pass
        return arr

    # 3) data list/tuple
    if hasattr(inf_node, 'data') and isinstance(getattr(inf_node, 'data'), (list, tuple)):
        data = getattr(inf_node, 'data')
        arr = _np.asarray(list(data), dtype=_np.float32)
        shp = _extract_shape(inf_node)
        if shp:
            try:
                arr = arr.reshape(shp)
            except Exception:
                pass
        return arr

    # 4) data bytes + shape
    if hasattr(inf_node, 'data') and not isinstance(getattr(inf_node, 'data'), (list, tuple)):
        try:
            buf = _np.frombuffer(getattr(inf_node, 'data'), dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    buf = buf.reshape(shp)
                except Exception:
                    pass
            return buf
        except Exception:
            pass

    # 5) buffer bytes + shape
    if hasattr(inf_node, 'buffer'):
        try:
            buf = _np.frombuffer(getattr(inf_node, 'buffer'), dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    buf = buf.reshape(shp)
                except Exception:
                    pass
            return buf
        except Exception:
            pass

    # 6) iterable fallback
    if hasattr(inf_node, '__len__') and hasattr(inf_node, '__getitem__'):
        try:
            arr = _np.asarray([inf_node[i] for i in range(len(inf_node))], dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    arr = arr.reshape(shp)
                except Exception:
                    pass
            return arr
        except Exception:
            pass

    # 7) 各種 getter
    for getter in ("get_float_data", "get_data", "get_buffer"):
        if hasattr(inf_node, getter) and callable(getattr(inf_node, getter)):
            try:
                raw = getattr(inf_node, getter)()
                if isinstance(raw, (bytes, bytearray)):
                    arr = _np.frombuffer(raw, dtype=_np.float32)
                else:
                    arr = _np.asarray(list(raw), dtype=_np.float32)
                shp = _extract_shape(inf_node)
                if shp:
                    try:
                        arr = arr.reshape(shp)
                    except Exception:
                        pass
                return arr
            except Exception:
                pass

    raise TypeError('Unknown inference node output type; cannot convert to numpy.')

def decode_heatmap_to_keypoints(heatmap, orig_w, orig_h):
    """heatmap: (K,H,W) 或 (1,K,H,W) → [(x,y,score), ...]"""
    if heatmap.ndim == 4:
        heatmap = heatmap[0]

    K, Hh, Wh = heatmap.shape
    flat = heatmap.reshape(K, -1)

    idx = flat.argmax(axis=1)
    ys = (idx // Wh).astype(np.float32)
    xs = (idx % Wh).astype(np.float32)
    scores = flat.max(axis=1).astype(np.float32)

    xs = xs * (orig_w / float(Wh))
    ys = ys * (orig_h / float(Hh))

    kpts = [(float(xs[i]), float(ys[i]), float(scores[i])) for i in range(K)]
    return kpts

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle(a, b, c):
    """
    計算角 ABC（degree）
    a, b, c = (x, y)
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0

    cos = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos))

def kpts_to_dict(kpts):
    return {
        name: (kpts[i][0], kpts[i][1])
        for i, name in enumerate(KEYPOINT_NAMES)
    }

def draw_hp_bar(frame, x, y, w, hp, max_hp, color_mode="player", hit_time=0):
    """
    在角色頭上畫 HP bar

    frame      : 遊戲畫面
    x, y       : 角色圖片左上角座標
    w          : 角色圖片寬度
    hp         : 目前 HP
    max_hp     : 最大 HP
    color_mode : "player" 或 "enemy"
    """

    # 防呆
    hp_ratio = max(0.0, min(1.0, hp / float(max_hp)))

    # 頭部位置
    head_x = x + w // 2
    head_y = y - 25   # 可微調

    # 血條尺寸
    bar_width = 120
    bar_height = 12

    bar_x = head_x - bar_width // 2
    bar_y = head_y

    # 背景
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (60, 60, 60),
        -1
    )

    # 顏色模式
    if color_mode == "player":
        if hp_ratio > 0.5:
            hp_color = (0, 255, 0)
        elif hp_ratio > 0.2:
            hp_color = (0, 255, 255)
        else:
            hp_color = (0, 0, 255)
    else:  # enemy
        if hp_ratio > 0.5:
            hp_color = (0, 0, 255)
        elif hp_ratio > 0.2:
            hp_color = (0, 165, 255)
        else:
            hp_color = (0, 255, 255)

    #被打中時閃爍
    if time.time() - hit_time < 0.2:
        hp_color = (255, 255, 255)  # 白色閃爍

    # 血量本體
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_width * hp_ratio), bar_y + bar_height),
        hp_color,
        -1
    )

    # 外框
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        2
    )

def overlay_png(bg, fg, x, y):
    """
    bg: BGR background image
    fg: BGRA png image
    x, y: top-left corner
    """
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]

    # 若完全在畫面外，直接不畫
    if x >= bw or y >= bh:
        return

    # 計算可畫範圍（裁切，不 return）
    w = min(fw, bw - x)
    h = min(fh, bh - y)

    fg = fg[:h, :w]
    bg_roi = bg[y:y+h, x:x+w]

    fg_rgb = fg[:, :, :3].astype(np.float32)
    alpha = fg[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    blended = fg_rgb * alpha + bg_roi.astype(np.float32) * (1 - alpha)

    bg[y:y+h, x:x+w] = blended.astype(np.uint8)

def extract_features(p):
    """
    p: dict of keypoints
    return: feature dict
    """

    shoulder_width = dist(p["left_shoulder"], p["right_shoulder"]) + 1e-6

    feats = {}

    # 手肘角度
    feats["left_elbow_angle"] = angle(
        p["left_shoulder"], p["left_elbow"], p["left_wrist"]
    )
    feats["right_elbow_angle"] = angle(
        p["right_shoulder"], p["right_elbow"], p["right_wrist"]
    )

    # 膝蓋角度
    feats["left_knee_angle"] = angle(
        p["left_hip"], p["left_knee"], p["left_ankle"]
    )
    feats["right_knee_angle"] = angle(
        p["right_hip"], p["right_knee"], p["right_ankle"]
    )

    # 手腕高度（y 越小越高）
    feats["left_wrist_height"] = (
        p["left_shoulder"][1] - p["left_wrist"][1]
    ) / shoulder_width

    feats["right_wrist_height"] = (
        p["right_shoulder"][1] - p["right_wrist"][1]
    ) / shoulder_width

    # 手腕距離（是否交叉 / 靠近）
    feats["wrist_dist"] = (
        dist(p["left_wrist"], p["right_wrist"]) / shoulder_width
    )

    # 身體傾斜（頭相對髖部中心）
    hip_center_x = (p["left_hip"][0] + p["right_hip"][0]) / 2
    feats["torso_shift"] = (
        p["nose"][0] - hip_center_x
    ) / shoulder_width

    return feats

def pose_similarity(feat_now, feat_ref):
    """
    return score: 0 ~ 100
    """

    total_error = 0.0
    total_weight = 0.0

    for k, w in FEATURE_WEIGHTS.items():
        if k not in feat_now or k not in feat_ref:
            continue

        diff = abs(feat_now[k] - feat_ref[k])

        # 角度誤差 normalize（180° → 1）
        if "angle" in k:
            diff = diff / 180.0

        total_error += diff * w
        total_weight += w

    if total_weight == 0:
        return 0

    norm_error = total_error / total_weight

    # 轉成分數（誤差越小，分數越高）
    score = max(0.0, 100.0 * (1.0 - norm_error))
    return score


# COCO 格式的關節點定義（17個點）

COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 骨架連線定義：哪些關節點要連在一起 [起點索引, 終點索引]
COCO_SKELETON = [
    [15, 13],  # left_ankle -> left_knee
    [13, 11],  # left_knee -> left_hip
    [16, 14],  # right_ankle -> right_knee
    [14, 12],  # right_knee -> right_hip
    [11, 12],  # left_hip -> right_hip
    [5, 11],   # left_shoulder -> left_hip
    [6, 12],   # right_shoulder -> right_hip
    [5, 6],    # left_shoulder -> right_shoulder
    [5, 7],    # left_shoulder -> left_elbow
    [6, 8],    # right_shoulder -> right_elbow
    [7, 9],    # left_elbow -> left_wrist
    [8, 10],   # right_elbow -> right_wrist
    [5, 0],    # left_shoulder -> nose (簡化，實際通過頭部)
    [6, 0],    # right_shoulder -> nose
    [0, 1],    # nose -> left_eye
    [0, 2],    # nose -> right_eye
    [1, 3],    # left_eye -> left_ear
    [2, 4],    # right_eye -> right_ear
]

# 關節點的顏色（BGR格式，OpenCV用）
# 不同身體部位用不同顏色系，比較好看
JOINT_COLORS = {
    # 頭部 (藍綠色系)
    0: (255, 255, 0),    # nose - 青色
    1: (255, 200, 0),    # left_eye - 淺青色
    2: (255, 200, 0),    # right_eye - 淺青色
    3: (200, 255, 0),    # left_ear - 黃綠色
    4: (200, 255, 0),    # right_ear - 黃綠色
    # 上半身左 (藍色系)
    5: (255, 0, 0),      # left_shoulder - 藍色
    7: (200, 0, 0),      # left_elbow - 深藍色
    9: (150, 0, 0),      # left_wrist - 更深藍色
    # 上半身右 (綠色系)
    6: (0, 255, 0),      # right_shoulder - 綠色
    8: (0, 200, 0),      # right_elbow - 深綠色
    10: (0, 150, 0),     # right_wrist - 更深綠色
    # 下半身左 (紅色系)
    11: (0, 0, 255),     # left_hip - 紅色
    13: (0, 0, 200),     # left_knee - 深紅色
    15: (0, 0, 150),     # left_ankle - 更深紅色
    # 下半身右 (紫色系)
    12: (255, 0, 255),   # right_hip - 洋紅色
    14: (200, 0, 200),   # right_knee - 深洋紅色
    16: (150, 0, 150),   # right_ankle - 更深洋紅色
}

# 骨架連線的顏色（BGR格式）
SKELETON_COLORS = {
    'head': (200, 200, 200),      # 頭部連線 - 灰色
    'torso': (128, 128, 255),     # 軀幹 - 淺藍色
    'left_arm': (255, 128, 128),  # 左手 - 淺紅色
    'right_arm': (128, 255, 128), # 右手 - 淺綠色
    'left_leg': (255, 255, 128),  # 左腿 - 淺黃色
    'right_leg': (255, 128, 255), # 右腿 - 淺洋紅色
}


def get_skeleton_color(idx1, idx2):
    """
    根據兩個關節點的索引決定連線顏色
    不同身體部位用不同顏色
    """
    # 頭部 (0-4)
    if idx1 <= 4 or idx2 <= 4:
        return SKELETON_COLORS['head']
    # 軀幹 (5,6,11,12)
    if idx1 in [5, 6, 11, 12] and idx2 in [5, 6, 11, 12]:
        return SKELETON_COLORS['torso']
    # 左手 (5,7,9)
    if idx1 in [5, 7, 9] or idx2 in [5, 7, 9]:
        return SKELETON_COLORS['left_arm']
    # 右手 (6,8,10)
    if idx1 in [6, 8, 10] or idx2 in [6, 8, 10]:
        return SKELETON_COLORS['right_arm']
    # 左腿 (11,13,15)
    if idx1 in [11, 13, 15] or idx2 in [11, 13, 15]:
        return SKELETON_COLORS['left_leg']
    # 右腿 (12,14,16)
    if idx1 in [12, 14, 16] or idx2 in [12, 14, 16]:
        return SKELETON_COLORS['right_leg']
    return (200, 200, 200)  # 預設灰色


def draw_keypoints_enhanced(img, keypoints, threshold=0.2, 
                           show_skeleton=True,  point_radius=5, 
                           line_thickness=3):
    """
    畫關節點和骨架的函數
    
    參數：
        img: 原始圖片（BGR格式）
        keypoints: 17個關節點的座標和分數 [(x, y, score), ...]
        threshold: 分數低於這個值就不畫
        show_skeleton: 要不要畫骨架連線
        show_labels: 要不要顯示關節點名稱（像 nose, left_eye 之類的）
        show_scores: 要不要顯示分數
        point_radius: 關節點的大小
        line_thickness: 骨架線條的粗細
    
    回傳：畫好的圖片
    """
    vis = img.copy()
    
    # 過濾低分數的關節點
    valid_kpts = []
    for i, (x, y, score) in enumerate(keypoints):
        if score >= threshold:
            valid_kpts.append((i, x, y, score))
    
    if not valid_kpts:
        return vis
    
    # 繪製骨架連線
    if show_skeleton:
        for (idx1, idx2) in COCO_SKELETON:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                x1, y1, s1 = keypoints[idx1]
                x2, y2, s2 = keypoints[idx2]
                
                # 只有兩個關節點都超過閾值才繪製連線
                if s1 >= threshold and s2 >= threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    color = get_skeleton_color(idx1, idx2)
                    cv2.line(vis, pt1, pt2, color, line_thickness, lineType=cv2.LINE_AA)
    
    # 繪製關節點
    for i, (x, y, score) in enumerate(keypoints):
        if score < threshold:
            continue
        
        pt = (int(x), int(y))
        color = JOINT_COLORS.get(i, (0, 255, 255))  # 預設黃色
        
        # 根據分數調整節點大小 (分數越高，節點越大)
        radius = int(point_radius * (0.7 + 0.6 * score))
        radius = max(3, min(radius, 12))  # 限制在 3-12 之間
        
        # 繪製外圈 (較亮)
        cv2.circle(vis, pt, radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # 繪製主體
        cv2.circle(vis, pt, radius, color, -1, lineType=cv2.LINE_AA)
        # 繪製內圈 (較暗，增加立體感)
        cv2.circle(vis, pt, max(2, radius // 2), tuple(c // 2 for c in color), -1, lineType=cv2.LINE_AA)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description='HRNet pose demo on KL730 (GenericImageInference).')
    parser.add_argument('-p', '--port_id', type=int, default=0, help='USB port ID (default: 0)')
    parser.add_argument('-fw', '--firmware', default='res/firmware/KL730/kp_firmware.tar',
                        help='Path to kp_firmware.tar')
    parser.add_argument('-m', '--model', default='res/models/KL730/HRNet/models_730.nef',
                        help='Path to HRNet NEF')
    parser.add_argument('-img', '--image', default=None,
                        help='Path to input image')
    parser.add_argument('--webcam', action='store_true',
                    help='Use webcam instead of image')
    parser.add_argument('--thresh', type=float, default=0.2,
                        help='score threshold for drawing keypoints')
    args = parser.parse_args()

    if not args.webcam and args.image is None:
        print("Please provide --image or use --webcam")
        return

    usb_port_id = args.port_id

    # --- Connect device ---
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=usb_port_id)
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != usb_speed and kp.UsbSpeed.KP_USB_SPEED_HIGH != usb_speed:
            print('\033[93m[Warning] Device is not at super/high speed.\033[0m')
    except Exception as e:
        print(f'[Warning] USB speed check failed: {e}')

    print('[Connect Device]')
    device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
    print(' - Success')

    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    # --- Firmware & model ---
    print('[Upload Firmware]')
    kp.core.load_firmware_from_file(device_group=device_group,
                                    scpu_fw_path=args.firmware,
                                    ncpu_fw_path='')
    print(' - Success')

    print('[Upload Model]')
    model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                        file_path=args.model)
    print(' - Success')

    # --- Read & preprocess image ---

    cap = None

    # --- Load UI assets ---
    background = cv2.imread("assets/bg.png")
    player_png = cv2.imread("assets/player.png", cv2.IMREAD_UNCHANGED)
    enemy_png  = cv2.imread("assets/enemy.png", cv2.IMREAD_UNCHANGED)
    magic_png = cv2.imread("assets/magic.png", cv2.IMREAD_UNCHANGED)
    player_png = cv2.resize(player_png, (128, 256))
    enemy_png  = cv2.resize(enemy_png,  (128, 256))
    magic_png = cv2.resize(magic_png, (128, 128))

    if background is None or player_png is None or enemy_png is None:
        print("Error: cannot load img")
        return


    REFERENCE_POSES = {}
    target_pose_key = None

    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: cannot open webcam")
            return
        
    in_h, in_w = _get_model_input_hw(model_nef_descriptor)
    print(f'[Model Input] HxW = {in_h}x{in_w}')

    # 統一背景大小
    cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Game", 1280, 720)
    bg_h, bg_w = 720, 1280
    background = cv2.resize(background, (bg_w, bg_h))

    #倒數時間初始化
    countdown_active = False
    countdown_start_time = 0
    COUNTDOWN_SECONDS = 2

    #戰鬥狀態初始化
    enemy_hp = 100
    player_hp = 100
    player_hit_time = 0
    enemy_hit_time = 0
    game_over = False
    game_result = None   # "WIN" or "LOSE"

    # ===== 姿勢設定模式 =====
    pose_setup_mode = True          # 一開始先設定姿勢
    pose_setup_key = None           # 目前要存的是哪個數字

    #姿勢捕捉變數初始化
    CAPTURE_SECONDS = 2.0
    capture_active = False
    capture_start_time = 0
    capture_scores = []
    
    #普通攻擊變數初始化
    last_attack_score = None
    last_damage = None
    
    #大招變數初始化
    combo_active = False
    combo_index = 0
    combo_pose_keys = []
    combo_scores = []

    COMBO_COUNT = 3
    COMBO_DAMAGE_MULTIPLIER = 2.5

    # ===== 魔法彈動畫 =====
    magic_active = False
    magic_start_time = 0
    MAGIC_DURATION = 0.4   # 飛行時間（秒）

    magic_start_pos = (0, 0)
    magic_end_pos   = (0, 0)
    pending_damage = 0

    #敵人攻擊變數初始化
    enemy_attack_active = False
    enemy_attack_start_time = 0
    ENEMY_ATTACK_SECONDS = 2.5
    attack_resolved = False

    enemy_defense_pose_key = None   # 敵人要求的防禦姿勢
    enemy_damage = 15
    # 姿勢 "0" 是防禦
    DEFENSE_POSE_KEY = "0"

    POSE_ICON_IMAGES = {}

    for i in range(10):
        img = cv2.imread(f"assets/{i}.png", cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] pose icon {i}.png not found")
            continue

        # 統一大小
        img = cv2.resize(img, (160, 160))
        POSE_ICON_IMAGES[str(i)] = img



    while True:

        # ===== GAME OVER 狀態鎖定 =====
        if game_over:
            frame = background.copy()
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (bg_w, bg_h),
                (0, 0, 0),
                -1
            )
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            result_text = "YOU WIN!" if game_result == "WIN" else "YOU LOSE"

            cv2.putText(
                frame,
                result_text,
                (bg_w // 2 - 220, bg_h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0) if game_result == "WIN" else (0, 0, 255),
                6
            )

            cv2.putText(
                frame,
                "Press R to Restart | Q to Quit",
                (bg_w // 2 - 280, bg_h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

            cv2.imshow("Pose Game", frame)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('r'):
                # reset
                enemy_hp = 100
                player_hp = 100
                game_over = False
                game_result = None

                countdown_active = False
                capture_active = False
                combo_active = False
                enemy_attack_active = False
                magic_active = False
                enemy_attack_start_time = 0
                attack_resolved = False
                pending_damage = 0
                target_pose_key = None
                last_attack_score = None
                last_damage = None

                print("[Game] Restart!")
                continue

            elif key == ord('q') or key == 27:
                break

            else:
                continue

        #讀取webcam
        if args.webcam:
            ret, img_bgr = cap.read()
            if not ret:
                break
        else:
            img_bgr = cv2.imread(args.image)
            if img_bgr is None:
                print(f'Error: cannot read image: {args.image}')
                break

        orig_h, orig_w = img_bgr.shape[:2]

        resized = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        img_bgr565 = cv2.cvtColor(resized, cv2.COLOR_BGR2BGR565)

        generic_desc = kp.GenericImageInferenceDescriptor(
            model_id=model_nef_descriptor.models[0].id,
            inference_number=0,
            input_node_image_list=[
                kp.GenericInputNodeImage(
                    image=img_bgr565,
                    resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
                    padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                    normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
                    image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565,
                )
            ]
        )


        # --- Inference ---
        kp.inference.generic_image_inference_send(
            device_group=device_group,
            generic_inference_input_descriptor=generic_desc
        )
        raw = kp.inference.generic_image_inference_receive(device_group=device_group)

        out0 = kp.inference.generic_inference_retrieve_float_node(
            node_idx=0,
            generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
        )
        heatmap = _to_numpy(out0)

        kpts = decode_heatmap_to_keypoints(heatmap, orig_w, orig_h)

        pose_dict = kpts_to_dict(kpts)
        current_feat = extract_features(pose_dict)
        
        #在攝影機上畫骨架
        vis = draw_keypoints_enhanced(
            img_bgr,
            kpts,
            threshold=args.thresh,
            show_skeleton=True,     # 顯示骨架
            point_radius=4,
            line_thickness=3
        )


        # 倒數邏輯
        now = time.time()

        if countdown_active:
            elapsed = now - countdown_start_time
            remaining = COUNTDOWN_SECONDS - int(elapsed)

            if remaining > 0:
                countdown_text = str(remaining)
            else:
                countdown_text = "GO"

            # 倒數結束
            if elapsed >= COUNTDOWN_SECONDS + 0.5:
                countdown_active = False
                countdown_text = None

                # 進入姿勢鎖定階段
                capture_active = True
                capture_start_time = time.time()
                capture_scores = []

                if pose_setup_mode:
                    print("[Pose Setup] Capturing pose...")
                else:
                    print("[Battle] Capturing attack pose")

        else:
            countdown_text = None

        #敵人攻擊邏輯
        enemy_text = None

        if enemy_attack_active:
            elapsed = time.time() - enemy_attack_start_time
            remaining = ENEMY_ATTACK_SECONDS - elapsed

            if remaining > 0:
                enemy_text = f"DEFEND! {remaining:.1f}"
            else:
                # 倒數結束 → 判斷是否成功防禦
                enemy_attack_active = False

                if enemy_defense_pose_key in REFERENCE_POSES:
                    ref_feat = REFERENCE_POSES[enemy_defense_pose_key]
                    defend_score = pose_similarity(current_feat, ref_feat)

                    if defend_score >= 70:
                        print("[Enemy Attack] Blocked!")
                    else:
                        player_hp = max(0, player_hp - enemy_damage)
                        player_hit_time = time.time()
                        if player_hp == 0 and not game_over:
                            game_over = True
                            magic_active = False
                            enemy_attack_active = False
                            attack_resolved = False
                            pending_damage = 0
                            game_result = "LOSE"

                        print("[Enemy Attack] Hit! HP -", enemy_damage)

                else:
                    # 沒有防禦模板，直接受傷
                    player_hp = max(0, player_hp - enemy_damage)
                    player_hit_time = time.time()
                    if player_hp == 0 and not game_over:
                        game_over = True
                        magic_active = False
                        enemy_attack_active = False
                        attack_resolved = False
                        pending_damage = 0
                        game_result = "LOSE"
                            
                    print("[Enemy Attack] No defense pose! HP -", enemy_damage)

        # ===== 是否顯示姿勢示意圖 =====
        show_pose_hint = (
            target_pose_key is not None
            and (countdown_active or capture_active)
        )
     
        #建立遊戲畫面
        frame = background.copy()# 先用背景當畫布
        # 疊玩家 / 敵人
        player_x, player_y = 350, 350
        enemy_x, enemy_y   = 900, 350

        overlay_png(frame, player_png, x=player_x, y=player_y)
        overlay_png(frame, enemy_png,  x=enemy_x,  y=enemy_y)

        draw_hp_bar(
        frame,
        x=player_x,
        y=player_y,
        w=player_png.shape[1],
        hp=player_hp,
        max_hp=100,
        color_mode="player",
        hit_time=player_hit_time
        )

        draw_hp_bar(
            frame,
            x=enemy_x,
            y=enemy_y,
            w=enemy_png.shape[1],
            hp=enemy_hp,
            max_hp=100,
            color_mode="enemy",
            hit_time=enemy_hit_time
        )

        # ===== 魔法彈動畫 =====
        if magic_active:
            t = (time.time() - magic_start_time) / MAGIC_DURATION
            if t >= 1.0:
                magic_active = False

                # ===== 命中結算（這裡才扣血）=====
                if pending_damage > 0:
                    enemy_hp = max(0, enemy_hp - pending_damage)
                    enemy_hit_time = time.time()
                    if enemy_hp == 0 and not game_over:
                        game_over = True
                        magic_active = False
                        enemy_attack_active = False
                        attack_resolved = False
                        pending_damage = 0
                        game_result = "WIN"

                    pending_damage = 0

            else:
                # 線性插值（從玩家飛到敵人）
                mx = int(magic_start_pos[0] * (1 - t) + magic_end_pos[0] * t)
                my = int(magic_start_pos[1] * (1 - t) + magic_end_pos[1] * t)

                overlay_png(
                    frame,
                    magic_png,
                    x=mx - magic_png.shape[1] // 2,
                    y=my - magic_png.shape[0] // 2
                )

        
        # 縮小攝影機畫面
        # 小視窗位置
        x0, y0 = 10, 350
        w, h = 320, 240

        # 黑色邊框
        cv2.rectangle(frame,
                    (x0-2, y0-2),
                    (x0+w+2, y0+h+2),
                    (0, 0, 0), 2)

        cam_small = cv2.resize(vis, (w, h))
        frame[y0:y0+h, x0:x0+w] = cam_small

        # ===== 姿勢示意圖（顯示在 webcam 上方）=====
        if show_pose_hint and target_pose_key in POSE_ICON_IMAGES:
            hint_img = POSE_ICON_IMAGES[target_pose_key]

            hint_w, hint_h = 160, 160
            hint_img = cv2.resize(hint_img, (hint_w, hint_h))

            hint_x = x0 + (w // 2) - (hint_w // 2)
            hint_y = y0 - hint_h - 15

            overlay_png(frame, hint_img, hint_x, hint_y)

            cv2.putText(
                frame,
                "POSE",
                (hint_x + 40, hint_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # 在小視窗上加入倒數數字
        if countdown_text:
            cv2.putText(
                frame,
                countdown_text,
                (x0 + w//2 - 30, y0 + h//2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                6
            )

        # 敵人攻擊提示
        if enemy_text:
            cv2.putText(
                frame,
                enemy_text,
                (bg_w//2 - 200, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                5
            )

        # --- Pose feature demo ---
        score = 0.0
        if not pose_setup_mode and target_pose_key in REFERENCE_POSES:
            ref_feat = REFERENCE_POSES[target_pose_key]
            score = pose_similarity(current_feat, ref_feat)

        # --- During capture window ---
        if capture_active:
            capture_scores.append(score)

            elapsed_cap = time.time() - capture_start_time
            if elapsed_cap >= CAPTURE_SECONDS:
                capture_active = False

                # ===== 姿勢設定完成 =====
                if pose_setup_mode and pose_setup_key is not None:
                    REFERENCE_POSES[pose_setup_key] = current_feat.copy()

                    print(f"[Pose Saved] key={pose_setup_key}")

                    # 清除設定狀態
                    pose_setup_key = None
                    target_pose_key = None

                    continue   # 跳過後面的攻擊邏輯

                if len(capture_scores) > 0:
                    avg_score = sum(capture_scores) / len(capture_scores)
                else:
                    avg_score = 0.0

                # ===== 普通攻擊 =====
                if not combo_active:
                    damage = int(avg_score / 8)
                    pending_damage = damage
                    last_attack_score = avg_score
                    last_damage = damage

                    attack_resolved = True

                    # 啟動魔法彈動畫（普通攻擊）
                    magic_active = True
                    magic_start_time = time.time()

                    magic_start_pos = (
                        player_x + player_png.shape[1] // 2,
                        player_y + 40
                    )

                    magic_end_pos = (
                        enemy_x + enemy_png.shape[1] // 2,
                        enemy_y + 40
                    )

                    print(f"[Normal Attack] score={avg_score:.1f}, dmg={damage}")

                # ===== Combo 攻擊 =====
                else:
                    combo_scores.append(avg_score)
                    print(f"[Combo] Step {combo_index+1} score={avg_score:.1f}")

                    combo_index += 1

                    if combo_index < COMBO_COUNT:
                        # 下一段
                        target_pose_key = combo_pose_keys[combo_index]
                        countdown_active = True
                        countdown_start_time = time.time()

                        print(f"[Combo] Next pose = {target_pose_key}")

                    else:
                        # Combo 結束 → 結算
                        avg_combo_score = sum(combo_scores) / len(combo_scores)
                        damage = int((avg_combo_score / 5) * COMBO_DAMAGE_MULTIPLIER)
                        pending_damage = damage

                        last_attack_score = avg_combo_score
                        last_damage = damage

                        combo_active = False
                        combo_index = 0
                        combo_scores = []
                        target_pose_key = None

                        attack_resolved = True

                        # 啟動魔法彈動畫
                        magic_active = True
                        magic_start_time = time.time()

                        # 起點：玩家頭部
                        magic_start_pos = (
                            player_x + player_png.shape[1] // 2,
                            player_y + 40
                        )

                        # 終點：敵人頭部
                        magic_end_pos = (
                            enemy_x + enemy_png.shape[1] // 2,
                            enemy_y + 40
                        )

                        print(f"[Combo Finished] avg={avg_combo_score:.1f}, dmg={damage}")

        # 啟動敵人反擊
        if (not game_over) and attack_resolved and not enemy_attack_active:
            enemy_attack_active = True
            enemy_attack_start_time = time.time()
            enemy_defense_pose_key = DEFENSE_POSE_KEY

            attack_resolved = False 
            target_pose_key = None
            print("[Enemy] Counter attack!")     

        #顯示分數
        cv2.putText(vis, f'Score: {score:.1f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
                
        if target_pose_key:
            cv2.putText(frame,
                        f'Target Pose: {target_pose_key}',
                        (30, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 200, 0), 2)
        #顯示本次攻擊結果
        if last_attack_score is not None:
            cv2.putText(
                frame,
                f'Attack Score: {last_attack_score:.1f}',
                (450, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 0), 3
            )

        if last_damage is not None:
            cv2.putText(
                frame,
                f'Damage: -{last_damage}',
                (500, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3
            )

        cv2.imshow("Pose Game", frame)

        key = cv2.waitKey(1) & 0xFF

        # 離開
        if key == 27 or key == ord('q'):
            break

        # 姿勢設定 0~9
        if pose_setup_mode and key in [ord(str(i)) for i in range(10)]:
            pose_setup_key = chr(key)
            target_pose_key = pose_setup_key   # 顯示示意圖
            countdown_active = True
            countdown_start_time = time.time()

            print(f"[Pose Setup] Start setting pose {pose_setup_key}")
        
        #完成姿勢設定
        if pose_setup_mode and key == 13:  # Enter
            if len(REFERENCE_POSES) >= 1:
                pose_setup_mode = False
                print("[Pose Setup] Finished! Enter battle mode.")
            else:
                print("[Pose Setup] Please set at least one pose.")


        # 普通攻擊
        if key == ord(' ') and not countdown_active and not capture_active and not combo_active:
            if len(REFERENCE_POSES) == 0:
                print("[Normal] No pose templates")
            else:
                import random

                target_pose_key = random.choice(list(REFERENCE_POSES.keys()))
                countdown_active = True
                countdown_start_time = time.time()

                print(f"[Normal Attack] Target pose = {target_pose_key}")

        # 大招 Combo
        if (not game_over) and key == ord('r') and not countdown_active and not capture_active and not combo_active:
            if len(REFERENCE_POSES) < COMBO_COUNT:
                print("[Combo] Not enough templates")
            else:
                import random

                combo_active = True
                combo_index = 0
                combo_scores = []

                combo_pose_keys = random.sample(
                    list(REFERENCE_POSES.keys()),
                    COMBO_COUNT
                )

                target_pose_key = combo_pose_keys[0]
                countdown_active = True
                countdown_start_time = time.time()

                print(f"[Combo Start] {combo_pose_keys}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
