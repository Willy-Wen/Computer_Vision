import numpy as np
import cv2
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    Il_pad = cv2.copyMakeBorder(Il, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    Ir_pad = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    Il_patch = np.zeros((*Il_pad.shape, 9))
    Ir_patch = np.zeros((*Ir_pad.shape, 9))
    idx = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            Il_patch[:, :, :, idx] = (Il_pad > np.roll(Il_pad, [y, x], axis=[0, 1])).astype(int)
            Ir_patch[:, :, :, idx] = (Ir_pad > np.roll(Ir_pad, [y, x], axis=[0, 1])).astype(int)
            idx += 1
    Il_patch = Il_patch[1:-1, 1:-1]
    Ir_patch = Ir_patch[1:-1, 1:-1]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    sigma_r, sigma_s = 4, 10
    Il_disp_costs = np.zeros((h, w, max_disp+1))
    Ir_disp_costs = np.zeros((h, w, max_disp+1))
    for d in range(max_disp+1):
        Il_patch_shift = Il_patch[:, d:].astype(np.uint32)
        Ir_patch_shift = Ir_patch[:, :w-d].astype(np.uint32)
        cost = np.sum(Il_patch_shift ^ Ir_patch_shift, axis=3)
        cost = np.sum(cost, axis=2).astype(np.float32)
        Il_cost = cv2.copyMakeBorder(cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        Il_disp_costs[:, :, d] = xip.jointBilateralFilter(Il, Il_cost, -1, sigma_r, sigma_s)
        Ir_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        Ir_disp_costs[:, :, d] = xip.jointBilateralFilter(Ir, Ir_cost, -1, sigma_r, sigma_s)
    Il_disp = np.argmin(Il_disp_costs, axis=2)
    Ir_disp = np.argmin(Ir_disp_costs, axis=2)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disp = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w), range(h))
    rx = x - Il_disp
    mask1 = rx >= 0
    l_disp = Il_disp[mask1]
    r_disp = Ir_disp[y[mask1], rx[mask1]]
    mask2 = (l_disp == r_disp)
    disp[y[mask1][mask2], x[mask1][mask2]] = Il_disp[mask1][mask2]

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    disp_pad = cv2.copyMakeBorder(disp, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=max_disp)
    label_l = np.zeros((h, w), dtype=np.float32)
    label_r = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            while disp_pad[y, x+1-idx_L] == 0:
                idx_L += 1
            label_l[y, x] = disp_pad[y, x+1-idx_L]
            while disp_pad[y, x+1+idx_R] == 0:
                idx_R += 1
            label_r[y, x] = disp_pad[y, x+1+idx_R]
    labels = np.min((label_l, label_r), axis=0)
    r = 11
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, r)

    return labels.astype(np.uint8)

