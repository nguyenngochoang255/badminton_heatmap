import cv2 as cv
import numpy as np
import sys


VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else 'tran_dau.mp4'


def create_csrt_tracker():
    if hasattr(cv, 'TrackerCSRT_create'):
        return cv.TrackerCSRT_create()
    if hasattr(cv, 'legacy') and hasattr(cv.legacy, 'TrackerCSRT_create'):
        return cv.legacy.TrackerCSRT_create()
    raise SystemExit('Khong tim thay CSRT tracker. Hay cai opencv-contrib-python.')


def read_frame_or_exit(cap, message):
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        cv.destroyAllWindows()
        raise SystemExit(message)
    return frame


cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f'can not open video clip/camera: {VIDEO_PATH}')
    exit()

# Tao Tracking van dong vien
frame = read_frame_or_exit(cap, 'can not read first frame')

roi = cv.selectROI('chon VDV de theo doi', frame, showCrosshair=False, fromCenter=False)
cv.destroyWindow('chon VDV de theo doi')
x_roi, y_roi, w_roi, h_roi = (int(v) for v in roi)
if w_roi <= 0 or h_roi <= 0:
    cap.release()
    cv.destroyAllWindows()
    raise SystemExit('ROI khong hop le')

tracker = create_csrt_tracker()
tracker.init(frame, (x_roi, y_roi, w_roi, h_roi))

# Chon 4 goc
frame = read_frame_or_exit(cap, 'can not read frame for court corner selection')
points = []


def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print('goc', len(points), ':', x, y)


cv.namedWindow('chon 4 goc san')
cv.setMouseCallback('chon 4 goc san', click)

while len(points) < 4:
    display = frame.copy()
    for idx, p in enumerate(points, start=1):
        cv.circle(display, (p[0], p[1]), 5, (0, 0, 255), -1)
        cv.putText(display, str(idx), (p[0] + 8, p[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow('chon 4 goc san', display)
    key = cv.waitKey(20) & 0xFF
    if key in (27, ord('q')):
        cap.release()
        cv.destroyAllWindows()
        raise SystemExit('Da huy chon goc san')
    if key == ord('r'):
        points.clear()
cv.destroyWindow('chon 4 goc san')

# Dung anh san nghieng thanh toa do san that
CHIEU_RONG_SAN = 6.1
CHIEU_DAI_SAN = 13.40
CELL_SIZE = 0.1

goc_anh = np.float32(points)
goc_that = np.float32([
    [0, 0],
    [CHIEU_RONG_SAN, 0],
    [CHIEU_RONG_SAN, CHIEU_DAI_SAN],
    [0, CHIEU_DAI_SAN],
])
H = cv.getPerspectiveTransform(goc_anh, goc_that)

# Tao luoi heatmap, 1 o = 0.1 m
nx = round(CHIEU_RONG_SAN / CELL_SIZE)
ny = round(CHIEU_DAI_SAN / CELL_SIZE)
heat = np.zeros((ny, nx), dtype=np.float32)

# Phat hien ban chan cua van dong vien
bg = cv.createBackgroundSubtractorMOG2(history=700, varThreshold=35, detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

start_pos = cap.get(cv.CAP_PROP_POS_FRAMES)
for _ in range(60):
    ret, f = cap.read()
    if not ret:
        break
    bg.apply(f, learningRate=0.1)
cap.set(cv.CAP_PROP_POS_FRAMES, start_pos)

# ---------- tham so loc nhieu ----------
MAX_JUMP = 1.5
LOST_RESET = 5

prev_feet = []
lost = 0


def splat(rx, ry):
    if not (0 <= rx < CHIEU_RONG_SAN and 0 <= ry < CHIEU_DAI_SAN):
        return
    gx, gy = rx / CELL_SIZE, ry / CELL_SIZE
    ix, iy = int(gx), int(gy)
    fxp, fyp = gx - ix, gy - iy
    for dy in (0, 1):
        for dx in (0, 1):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny:
                wgt = (fxp if dx else 1 - fxp) * (fyp if dy else 1 - fyp)
                heat[jy, jx] += wgt


def split_wide_component(labels, label_id, x1, y1, band_top, min_gap):
    mask = labels == label_id
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    cols = np.unique(xs)
    gaps = np.diff(cols)
    if len(gaps) == 0 or gaps.max() <= min_gap:
        return []

    cut = int(gaps.argmax())
    groups = [cols[:cut + 1], cols[cut + 1:]]
    feet = []
    for group in groups:
        keep = np.isin(xs, group)
        if not np.any(keep):
            continue
        feet.append((x1 + float(xs[keep].mean()), y1 + band_top + int(ys[keep].max())))
    return feet


def find_feet_points(fg, box):
    x, y, w, h = (int(v) for v in box)
    frame_h, frame_w = fg.shape[:2]

    # Tracker CSRT hay bam than nguoi, nen mo rong vung tim xuong duoi de bat giay.
    pad_x = max(18, int(w * 0.35))
    extra_down = max(30, int(h * 0.60))
    x1 = max(x - pad_x, 0)
    y1 = max(y + int(h * 0.42), 0)
    x2 = min(x + w + pad_x, frame_w)
    y2 = min(y + h + extra_down, frame_h)
    search_rect = (x1, y1, x2, y2)
    fallback = [(x + w / 2, min(y + h, frame_h - 1))]

    if x2 <= x1 or y2 <= y1:
        return fallback, search_rect

    sub = fg[y1:y2, x1:x2]
    ys, xs = np.where(sub > 0)
    if len(ys) == 0:
        return fallback, search_rect

    y_max = int(ys.max())
    band_h = max(18, int(h * 0.24))
    band_top = max(0, y_max - band_h)
    foot_mask = sub[band_top:, :]

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(foot_mask, 8)
    min_area = max(10, int(w * h * 0.0015))
    near_bottom = max(12, int(h * 0.18))
    candidates = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv.CC_STAT_AREA])
        if area < min_area:
            continue

        top = int(stats[label_id, cv.CC_STAT_TOP])
        width = int(stats[label_id, cv.CC_STAT_WIDTH])
        height = int(stats[label_id, cv.CC_STAT_HEIGHT])
        bottom = top + height - 1
        if bottom < (y_max - band_top) - near_bottom:
            continue

        cx = float(centroids[label_id][0])
        candidates.append({
            'label': label_id,
            'area': area,
            'width': width,
            'bottom': bottom,
            'cx': cx,
            'point': (x1 + cx, y1 + band_top + bottom),
        })

    if not candidates:
        return fallback, search_rect

    candidates.sort(key=lambda c: (c['bottom'], c['area']), reverse=True)

    selected = []
    min_sep = max(12, int(w * 0.18))
    for item in candidates:
        if all(abs(item['cx'] - prev['cx']) >= min_sep for prev in selected):
            selected.append(item)
        if len(selected) == 2:
            break

    if len(selected) == 1 and selected[0]['width'] > max(24, int(w * 0.50)):
        split_feet = split_wide_component(
            labels,
            selected[0]['label'],
            x1,
            y1,
            band_top,
            min_gap=max(8, int(w * 0.12)),
        )
        if len(split_feet) >= 2:
            return sorted(split_feet[:2], key=lambda p: p[0]), search_rect

    feet_px = [item['point'] for item in selected]
    return sorted(feet_px, key=lambda p: p[0]), search_rect


while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg = bg.apply(frame)
    fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)[1]
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel, iterations=2)

    ok, box = tracker.update(frame)
    if ok:
        lost = 0
        x, y, w, h = (int(v) for v in box)
        feet_px, foot_rect = find_feet_points(fg, box)

        cur_feet = []
        for i, (foot_x, foot_y) in enumerate(feet_px):
            pt = np.float32([[[foot_x, foot_y]]])
            real = cv.perspectiveTransform(pt, H)
            rx, ry = float(real[0][0][0]), float(real[0][0][1])

            if prev_feet:
                # feet_px va prev_feet deu da sort theo x, nen khi cung so chan
                # thi khop trai-trai/phai-phai theo chi so de tranh gop 2 chan ve 1 diem.
                if len(feet_px) == len(prev_feet):
                    near = prev_feet[i]
                else:
                    near = min(prev_feet, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2)
                if ((rx - near[0]) ** 2 + (ry - near[1]) ** 2) ** 0.5 > MAX_JUMP:
                    rx, ry = near
            cur_feet.append((rx, ry))

            splat(rx, ry)
            if len(feet_px) == 2:
                mau = (0, 0, 255) if i == 0 else (255, 0, 0)
            else:
                mau = (255, 255, 255)
            cv.circle(frame, (int(foot_x), int(foot_y)), 4, mau, -1)

        prev_feet = cur_feet
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        fx1, fy1, fx2, fy2 = foot_rect
        cv.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), 1)
    else:
        lost += 1
        if lost >= LOST_RESET:
            prev_feet = []

    cv.imshow('video', frame)
    key = cv.waitKey(1) & 0xFF
    if key in (ord('p'), ord('q'), 27):
        break

cap.release()

# Ve heatmap
heat = cv.GaussianBlur(heat, (0, 0), 2)
if heat.max() > 0:
    heat = heat / heat.max()
heat8 = np.uint8(heat * 255)
heat_color = cv.applyColorMap(heat8, cv.COLORMAP_JET)
heat_color[heat8 < 10] = (94, 140, 74)

SCALE_X = 90
SCALE_Y = 60
W = int(CHIEU_RONG_SAN * SCALE_X)
L = int(CHIEU_DAI_SAN * SCALE_Y)
heat_color = cv.resize(heat_color, (W, L))


def px(x_m, y_m):
    return (int(x_m * SCALE_X), int(y_m * SCALE_Y))


KHOANG_CACH_BIEN_DOC = 0.46
AREA_1 = 1.98
KHOANG_CACH_BIEN_NGANG = 0.76
GIUA_SAN = CHIEU_DAI_SAN / 2

cv.rectangle(heat_color, px(0, 0), px(CHIEU_RONG_SAN, CHIEU_DAI_SAN), (255, 255, 255), 2)
cv.line(heat_color, px(KHOANG_CACH_BIEN_DOC, 0), px(KHOANG_CACH_BIEN_DOC, CHIEU_DAI_SAN), (255, 255, 255), 2)
cv.line(heat_color, px(CHIEU_RONG_SAN - KHOANG_CACH_BIEN_DOC, 0), px(CHIEU_RONG_SAN - KHOANG_CACH_BIEN_DOC, CHIEU_DAI_SAN), (255, 255, 255), 2)
cv.line(heat_color, px(0, GIUA_SAN), px(CHIEU_RONG_SAN, GIUA_SAN), (255, 255, 255), 2)
cv.line(heat_color, px(0, GIUA_SAN - AREA_1), px(CHIEU_RONG_SAN, GIUA_SAN - AREA_1), (255, 255, 255), 2)
cv.line(heat_color, px(0, GIUA_SAN + AREA_1), px(CHIEU_RONG_SAN, GIUA_SAN + AREA_1), (255, 255, 255), 2)
cv.line(heat_color, px(0, KHOANG_CACH_BIEN_NGANG), px(CHIEU_RONG_SAN, KHOANG_CACH_BIEN_NGANG), (255, 255, 255), 2)
cv.line(heat_color, px(0, CHIEU_DAI_SAN - KHOANG_CACH_BIEN_NGANG), px(CHIEU_RONG_SAN, CHIEU_DAI_SAN - KHOANG_CACH_BIEN_NGANG), (255, 255, 255), 2)
cv.line(heat_color, px(CHIEU_RONG_SAN / 2, 0), px(CHIEU_RONG_SAN / 2, GIUA_SAN - AREA_1), (255, 255, 255), 2)
cv.line(heat_color, px(CHIEU_RONG_SAN / 2, GIUA_SAN + AREA_1), px(CHIEU_RONG_SAN / 2, CHIEU_DAI_SAN), (255, 255, 255), 2)

y_cat = int(GIUA_SAN * SCALE_Y)
heat_color = heat_color[y_cat:, :]

cv.imwrite('heatmap.png', heat_color)
cv.imshow('heatmap', heat_color)
cv.waitKey(0)
cv.destroyAllWindows()
