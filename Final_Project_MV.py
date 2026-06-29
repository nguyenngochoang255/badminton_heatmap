# Họ và tên: Nguyễn Ngọc Hoàng
# MSSV: 16146020

import cv2 as cv
import numpy as np

cap = cv.VideoCapture('badminton_match_3.mp4')
if not cap.isOpened():
    print('Khong mo duoc video')
    exit()
 
# Tạo Tracking vận động viên
ret, frame = cap.read()              

roi = cv.selectROI('chon VDV de theo doi', frame, showCrosshair=False)
cv.destroyWindow('chon VDV de theo doi')
tracker = cv.TrackerCSRT_create()    
tracker.init(frame, tuple(int(v) for v in roi))

# Chọn 4 góc
ret, frame = cap.read()              
points = []                          

def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print("goc", len(points), ":", x, y)

cv.namedWindow('chon 4 goc san')
cv.setMouseCallback('chon 4 goc san', click)

while len(points) < 4:               
    for p in points:                 
        cv.circle(frame, (p[0], p[1]), 5, (0, 0, 255), -1)      # Vẽ 4 chấm đỏ để chọn góc (theo thứ tự trên trái-trên phải-dưới phải-dưới trái)
    cv.imshow('chon 4 goc san', frame)
    cv.waitKey(1)
cv.destroyWindow('chon 4 goc san')

# Dựng ảnh sân nghiêng thành ảnh thẳng
CHIEU_RONG_SAN      = 6.1           # Chiều rộng sân là 6.1 m
CHIEU_DAI_SAN       = 13.40         # Chiều dài sân là 13.4 m

goc_anh     = np.float32(points)
goc_that    = np.float32([[0, 0], [CHIEU_RONG_SAN, 0], [CHIEU_RONG_SAN, CHIEU_DAI_SAN], [0, CHIEU_DAI_SAN]])
H = cv.getPerspectiveTransform(goc_anh, goc_that)

# Tạo lưới để đếm điểm ảnh trong heatmap
nx = round(CHIEU_RONG_SAN / 0.1)        # Làm tròn thành 61 ô (1 ô = 10x10)
ny = round(CHIEU_DAI_SAN / 0.1)         # Làm tròn thành 134 ô (1 ô = 10x10)
heat = np.zeros((ny, nx))  

# Khởi tạo Mixture of Gaussians
bg = cv.createBackgroundSubtractorMOG2(history=700, varThreshold=35, detectShadows=True)

start_pos = cap.get(cv.CAP_PROP_POS_FRAMES)   
for _ in range(60):                          
    ret, f = cap.read()
    if not ret:
        break
    bg.apply(f, learningRate=0.1)
cap.set(cv.CAP_PROP_POS_FRAMES, start_pos)


def add_to_heatmap(rx, ry):
    if not (0 <= rx < CHIEU_RONG_SAN and 0 <= ry < CHIEU_DAI_SAN):
        return
    gx, gy = rx / 0.1, ry / 0.1
    ix, iy = int(gx), int(gy)
    fxp, fyp = gx - ix, gy - iy          
    for dy in (0, 1):
        for dx in (0, 1):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny:
                wgt = (fxp if dx else 1 - fxp) * (fyp if dy else 1 - fyp)
                heat[jy, jx] += wgt

while True:
    ret, frame = cap.read()
    if not ret:                     
        break

    fg = bg.apply(frame)             
    fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)[1]   

    ok, box = tracker.update(frame) 
    if ok:
        x, y, w, h = (int(v) for v in box)

        sub = fg[max(y, 0):y + h, max(x, 0):x + w]   
        ww = sub.shape[1]
        ys, xs = np.where(sub > 0)
        feet_px = []                                
        if len(ys) > 0:
            bottom = np.full(ww, -1)
            np.maximum.at(bottom, xs, ys)            
            y_max = int(bottom.max())              
            foot_lv = bottom >= y_max - max(30, int(h * 0.3))
            cols = np.where(foot_lv)[0]              
            gaps = np.diff(cols)                     
            if len(gaps) > 0 and gaps.max() > 20:   
                cut = int(gaps.argmax())
                groups = [cols[:cut + 1], cols[cut + 1:]]   
            else:                                    
                groups = [cols]
            ox, oy = max(x, 0), max(y, 0)
            for g in groups:                         
                feet_px.append((ox + float(g.mean()), oy + int(bottom[g].max())))
        else:                                        
            feet_px.append((x + w / 2, y + h))

        for i, (foot_x, foot_y) in enumerate(feet_px):
            pt = np.float32([[[foot_x, foot_y]]])
            real = cv.perspectiveTransform(pt, H)
            rx, ry = float(real[0][0][0]), float(real[0][0][1])

            add_to_heatmap(rx, ry)                   # cong chan nay vao heatmap
            if len(feet_px) == 2:                    # tach duoc 2 chan: trai DO, phai XANH
                mau = (0, 0, 255) if i == 0 else (255, 0, 0)
            else:                                    # 2 chan sat nhau: 1 cham TRANG
                mau = (255, 255, 255)
            cv.circle(frame, (int(foot_x), int(foot_y)), 4, mau, -1)

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow('video', frame)
    cv.imshow('MOG2 mask', fg)       
    if cv.waitKey(1) == ord('p'):
        break

cap.release()

# Vẽ heatmap
heat = cv.GaussianBlur(heat, (0, 0), 2)     
if heat.max() > 0:
    heat = heat / heat.max()              
heat8 = np.uint8(heat * 255)
heat_color = cv.applyColorMap(heat8, cv.COLORMAP_JET)  
heat_color[heat8 < 10] = (94, 140, 74)  

# Phóng to hình
SCALE_X = 90      
SCALE_Y = 60      
W = int(CHIEU_RONG_SAN  * SCALE_X)
L = int(CHIEU_DAI_SAN   * SCALE_Y)
heat_color = cv.resize(heat_color, (W, L))

# ham doi toa do that (met) -> pixel tren anh
def px(x_m, y_m):
    return (int(x_m * SCALE_X), int(y_m * SCALE_Y))

# Vẽ lại sân trong heatmap
KHOANG_CACH_BIEN_DOC    = 0.46      # Biên dọc giữa đánh đôi và đơn rộng 0.46 m
AREA_1 = 1.98                       # Khu vực giao cầu ngắn dài 1.98 m
KHOANG_CACH_BIEN_NGANG  = 0.76      # Biên ngang giữa đánh đôi và đơn dài 0.76 m
GIUA_SAN = CHIEU_DAI_SAN / 2        # Vị trí lưới
WHITE = (255, 255, 255)
# Khung bao ngoài: toàn sân đánh đôi, từ (0,0) đến (6.1, 13.4)
cv.rectangle(heat_color, px(0, 0), px(CHIEU_RONG_SAN, CHIEU_DAI_SAN), WHITE, 2)

# Hai biên dọc đánh ĐƠN (chừa hành lang đôi 0.46m mỗi bên)
cv.line(heat_color, px(KHOANG_CACH_BIEN_DOC, 0), px(KHOANG_CACH_BIEN_DOC, CHIEU_DAI_SAN), WHITE, 2)                                    
cv.line(heat_color, px(CHIEU_RONG_SAN - KHOANG_CACH_BIEN_DOC, 0), px(CHIEU_RONG_SAN - KHOANG_CACH_BIEN_DOC, CHIEU_DAI_SAN), WHITE, 2)  

# Vạch LƯỚI: chia sân làm 2 nửa (y = 6.7)
cv.line(heat_color, px(0, GIUA_SAN), px(CHIEU_RONG_SAN, GIUA_SAN), WHITE, 2)

# Hai vạch GIAO CẦU NGẮN: cách lưới 1.98m mỗi bên
cv.line(heat_color, px(0, GIUA_SAN - AREA_1), px(CHIEU_RONG_SAN, GIUA_SAN - AREA_1), WHITE, 2)   # nửa TRÊN  (y = 4.72)
cv.line(heat_color, px(0, GIUA_SAN + AREA_1), px(CHIEU_RONG_SAN, GIUA_SAN + AREA_1), WHITE, 2)   # nửa DƯỚI  (y = 8.68)

# Hai vạch GIAO CẦU DÀI đánh đôi: cách biên cuối 0.76m
cv.line(heat_color, px(0, KHOANG_CACH_BIEN_NGANG), px(CHIEU_RONG_SAN, KHOANG_CACH_BIEN_NGANG), WHITE, 2)                                  
cv.line(heat_color, px(0, CHIEU_DAI_SAN - KHOANG_CACH_BIEN_NGANG), px(CHIEU_RONG_SAN, CHIEU_DAI_SAN - KHOANG_CACH_BIEN_NGANG), WHITE, 2)  

# Vạch GIỮA (dọc) chia ô giao cầu trái/phải; vẽ 2 đoạn, chừa khoảng gần lưới
cv.line(heat_color, px(CHIEU_RONG_SAN / 2, 0), px(CHIEU_RONG_SAN / 2, GIUA_SAN - AREA_1), WHITE, 2)              
cv.line(heat_color, px(CHIEU_RONG_SAN / 2, GIUA_SAN + AREA_1), px(CHIEU_RONG_SAN / 2, CHIEU_DAI_SAN), WHITE, 2)  


y_cat = int(GIUA_SAN * SCALE_Y)          
heat_color = heat_color[y_cat:, :]   

cv.imwrite('heatmap.png', heat_color)
cv.imshow('heatmap', heat_color)
cv.waitKey(0)
cv.destroyAllWindows()
