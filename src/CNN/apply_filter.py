import cv2
import numpy as np

def overlay_image_alpha(img, overlay, x, y, overlay_size=None, angle=0, scale_x=1.0, flip_y=False):
    """
    Chồng ảnh PNG có alpha channel lên khung hình.

    Parameters
    ----------
    img : np.ndarray
        Ảnh gốc (frame BGR) để vẽ filter lên.
    overlay : np.ndarray
        Ảnh filter có alpha channel (BGRA).
    x, y : int
        Tọa độ góc trên bên trái vị trí đặt overlay trên img.
    overlay_size : tuple(int, int), optional
        Kích thước (width, height) mới của overlay. Nếu None thì giữ nguyên.
    angle : float
        Góc xoay (deg) áp dụng cho overlay.
    scale_x : float
        Hệ số scale theo trục X (để mô phỏng xoay mặt theo yaw).
    flip_y : bool
        Nếu True thì lật overlay theo trục Y (flip top–bottom).

    Returns
    -------
    np.ndarray
        Ảnh sau khi chồng overlay.
    """
    if overlay is None:
        return img

    # Resize overlay nếu có yêu cầu
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    # Lật filter theo trục Y để phù hợp transform của khuôn mặt
    if flip_y:
        overlay = cv2.flip(overlay, 0)

    h, w = overlay.shape[:2]

    # Ma trận xoay 2D; scale_x áp dụng chủ yếu để mô phỏng chiều sâu (yaw head tilt)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    M[0,0] *= scale_x
    M[0,1] *= scale_x

    # Áp dụng warpAffine lên overlay
    overlay = cv2.warpAffine(overlay, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0,0,0,0))  # giữ alpha = 0 ở vùng rỗng

    # Lấy alpha channel và làm mượt (blur) giúp filter blend mềm hơn
    alpha = overlay[:,:,3] / 255.0 if overlay.shape[2] == 4 else np.ones((h, w))
    alpha = cv2.GaussianBlur(alpha, (7,7), 0)

    # Tính vùng hợp lệ còn nằm trong frame
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img.shape[1])
    y2 = min(y + h, img.shape[0])

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Nếu overlay nằm ngoài frame → bỏ qua
    if overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1:
        return img

    # Alpha blending từng kênh BGR
    for c in range(3):
        img[y1:y2, x1:x2, c] = \
            (1 - alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]) \
            * img[y1:y2, x1:x2, c] + \
            alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2] \
            * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]

    return img


def get_head_angle_and_scale(landmarks):
    """
    Tính góc xoay của đầu theo hướng nghiêng (roll) và hệ số scale theo trục X (yaw).

    Parameters
    ----------
    landmarks : np.ndarray
        Mảng 68 điểm landmark dạng (N,2).

    Returns
    -------
    angle : float
        Góc nghiêng đầu (roll), tính từ vector nối 2 mắt.
    scale_x : float
        Scale theo trục X giúp hiệu ứng nghiêng đầu tự nhiên hơn.
    """
    # Trung bình 6 điểm mắt trái/phải → giảm nhiễu
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    dx = left_eye[0] - right_eye[0]
    dy = right_eye[1] - left_eye[1]

    # arctan2 → tính roll head tilt (nghiêng trái/phải)
    angle = np.degrees(np.arctan2(dy, dx))

    # scale_x mô phỏng độ nghiêng mặt theo yaw (nghiêng trước–sau)
    scale_x = 1.0 - 0.3 * np.tanh(dy / dx)

    return angle, scale_x


def draw_glasses(frame, landmarks, glasses_img):
    """
    Vẽ kính lên vị trí mắt dựa theo landmarks và góc đầu.

    Logic:
    - Tính angle + scale_x từ hướng mắt.
    - Lấy bounding box hai mắt → xác định kích thước kính.
    - Scale kính theo 1.8 để bao phủ vùng mắt rộng hơn.
    """
    angle, scale_x = get_head_angle_and_scale(landmarks)

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Lấy vùng bao phủ cả hai mắt
    min_x = int(min(left_eye[:,0].min(), right_eye[:,0].min()))
    max_x = int(max(left_eye[:,0].max(), right_eye[:,0].max()))
    min_y = int(min(left_eye[:,1].min(), right_eye[:,1].min()))
    max_y = int(max(left_eye[:,1].max(), right_eye[:,1].max()))

    glasses_w = int((max_x - min_x) * 1.8)
    glasses_h = int(glasses_w * glasses_img.shape[0] / glasses_img.shape[1])

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    x = center_x - glasses_w // 2
    y = center_y - glasses_h // 2

    return overlay_image_alpha(frame, glasses_img, x, y,
                               (glasses_w, glasses_h),
                               angle=angle, scale_x=scale_x, flip_y=True)


def draw_mustache(frame, landmarks, mustache_img):
    """
    Vẽ ria mép vào vị trí giữa mũi và miệng.
    """
    nose = landmarks[33].astype(int)
    mouth = landmarks[51].astype(int)

    mx = (nose[0] + mouth[0]) // 2
    my = (nose[1] + mouth[1]) // 2

    mouth_left = landmarks[48]
    mouth_right = landmarks[54]

    # Độ rộng miệng quyết định độ rộng ria mép
    mouth_w = int(np.linalg.norm(mouth_right - mouth_left) * 1.0)
    mustache_h = int(mouth_w * 0.4)

    # Tính angle theo hướng miệng → để ria xoay theo khuôn mặt
    dx = mouth_right[0] - mouth_left[0]
    dy = mouth_right[1] - mouth_left[1]
    angle = -np.degrees(np.arctan2(dy, dx))

    x = mx - mouth_w // 2
    y = my - mustache_h // 2

    return overlay_image_alpha(frame, mustache_img, x, y,
                               (mouth_w, mustache_h),
                               angle=angle)


def draw_pignose(frame, landmarks, pignose_img):
    """
    Vẽ mũi heo dựa vào landmark mũi và khoảng cách giữa 2 mắt.
    """
    nose = landmarks[33].astype(int)

    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    eye_dist = np.linalg.norm(right_eye - left_eye)

    # Scale mũi theo khoảng cách 2 mắt
    scale = eye_dist / 60

    w = int(pignose_img.shape[1] * 0.15 * scale)
    h = int(pignose_img.shape[0] * 0.15 * scale)

    x = nose[0] - w // 2
    y = nose[1] - h // 2 - 10  # nâng filter lên trên mũi

    return overlay_image_alpha(frame, pignose_img, x, y, (w, h), flip_y=False)


def draw_blush(frame, landmarks, blush_img):
    """
    Vẽ má hồng vào vị trí dưới mắt trái/phải.
    """
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_pos = np.mean(left_eye, axis=0).astype(int)
    right_pos = np.mean(right_eye, axis=0).astype(int)

    eye_dist = np.linalg.norm(right_pos - left_pos)
    scale = eye_dist / 60

    w = int(blush_img.shape[1] * 0.15 * scale)
    h = int(blush_img.shape[0] * 0.15 * scale)

    offset_y = -15      # chỉnh vị trí cao/thấp
    offset_x = 11     # tách ra khỏi mũi

    frame = overlay_image_alpha(frame, blush_img,
                                left_pos[0] - w//2 - offset_x,
                                left_pos[1] - offset_y,
                                (w, h), flip_y=True)

    frame = overlay_image_alpha(frame, blush_img,
                                right_pos[0] - w//2 + offset_x,
                                right_pos[1] - offset_y,
                                (w, h), flip_y=True)

    return frame

