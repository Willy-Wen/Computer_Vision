import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # TODO: 1.forming A
    A = np.zeros((2*N, 9))
    for i in range(N):
        A[i*2, :] = [u[i][0], u[i][1], 1, 0, 0, 0, -1*u[i][0]*v[i][0], -1*u[i][1]*v[i][0], -1*v[i][0]]
        A[i*2+1, :] = [0, 0, 0, u[i][0], u[i][1], 1, -1*u[i][0]*v[i][1], -1*u[i][1]*v[i][1], -1*v[i][1]]
    # TODO: 2.solve H with A
    U, S, V = np.linalg.svd(A)
    # print('U:', U.shape, 'S:', S.shape, 'V:', V.shape)
    H = V.T[:, -1].reshape((3, 3))
    # print(H)
    return H

# def back_mask(U, dst_corns):
#     if dst_corns[0][1]==dst_corns[1][1]:
#         up_bound_mask = U[1] >= dst_corns[0][1]
#     else:
#         if dst_corns[0][1]>dst_corns[1][1]:
#             max_id, min_id = 0, 1
#         else:
#             max_id, min_id = 1, 0
#         up_bound_mask = ((U[1]-dst_corns[min_id][1])/(dst_corns[max_id][1]-dst_corns[min_id][1])-(U[0]-dst_corns[min_id][0])/(dst_corns[max_id][0]-dst_corns[min_id][0])) >= 0
#
#     if dst_corns[2][1]==dst_corns[3][1]:
#         bottom_bound_mask = U[1] <= dst_corns[2][1]
#     else:
#         if dst_corns[2][1]>dst_corns[3][1]:
#             max_id, min_id = 2, 3
#         else:
#             max_id, min_id = 3, 2
#         bottom_bound_mask = ((U[1]-dst_corns[min_id][1])/(dst_corns[max_id][1]-dst_corns[min_id][1])-(U[0]-dst_corns[min_id][0])/(dst_corns[max_id][0]-dst_corns[min_id][0])) <= 0
#
#     if dst_corns[0][0]==dst_corns[3][0]:
#         left_bound_mask = U[0] >= dst_corns[0][0]
#     else:
#         if dst_corns[0][0]>dst_corns[3][0]:
#             max_id, min_id = 0, 3
#         else:
#             max_id, min_id = 3, 0
#         left_bound_mask = ((U[1]-dst_corns[min_id][1])/(dst_corns[max_id][1]-dst_corns[min_id][1])-(U[0]-dst_corns[min_id][0])/(dst_corns[max_id][0]-dst_corns[min_id][0])) <= 0
#
#     if dst_corns[1][0]==dst_corns[2][0]:
#         right_bound_mask = U[0] <= dst_corns[1][0]
#     else:
#         if dst_corns[1][0]>dst_corns[2][0]:
#             max_id, min_id = 1, 2
#         else:
#             max_id, min_id = 2, 1
#         right_bound_mask = ((U[1]-dst_corns[min_id][1])/(dst_corns[max_id][1]-dst_corns[min_id][1])-(U[0]-dst_corns[min_id][0])/(dst_corns[max_id][0]-dst_corns[min_id][0])) >= 0
#     mask = up_bound_mask & bottom_bound_mask & left_bound_mask & right_bound_mask
#     return mask

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b', blending=False):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)
    # TODO: 1.meshgrid the (x,y) coordinate pairs
    Ux, Uy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    # TODO: 2.reshape the destination pixels as N x 3(x) 3 x N(o) homogeneous coordinate
    U = np.concatenate(([Ux.reshape(-1)], [Uy.reshape(-1)], [np.ones((xmax - xmin) * (ymax - ymin))]), axis=0)
    # print('\nU.shape: ', U.shape, '\nU:\n', U.astype(int), '\n')
    if direction == 'b':
        # # ========================== method 1 ==========================
        # src_corns = np.array([[0, w_src, w_src, 0], [0, 0, h_src, h_src], [1, 1, 1, 1]])
        # dst_corns = np.dot(H, src_corns)
        # dst_corns = (np.around(dst_corns/dst_corns[2]).astype(int))[0:2][0:4].T
        # # print('\ndst_corns:\n', dst_corns)
        # mask = back_mask(U, dst_corns)
        # # print(U.shape, mask.shape)
        # mUx, mUy = U[0][mask], U[1][mask]
        # # print(Ux.shape, Uy.shape)
        # mU = np.concatenate(([mUx], [mUy], [np.ones(mUx.shape)]), axis=0)
        # V = np.dot(H_inv, mU)
        # mU = np.around(mU).astype(int)
        # V = (V/V[2]).astype(int)
        # dst[mU[1], mU[0]] = src[np.clip(V[1], 0, h_src-1), np.clip(V[0], 0, w_src-1)]

        # ========================== method 2 ==========================
        V = np.dot(H_inv, U)
        V = np.around(V/V[2])
        # print('\nV.shape: ', V.shape, '\nV:\n', V, '\n')
        mask = (V[0] >= 0) & (V[0] <= w_src-1) & (V[1] >= 0) & (V[1] <= h_src-1)
        mUx, mUy = U[0][mask], U[1][mask]
        mU = np.concatenate(([mUx], [mUy], [np.ones(mUx.shape)]), axis=0).astype(int)
        mV = np.dot(H_inv, mU)
        mV = np.around(mV/mV[2]).astype(int)
        if blending:
            # am = np.sum(dst[mU[1], mU[0]], axis=1) != 0
            # u_max, u_min = max(mU[0][am]), min(mU[0][am])
            # a = (mU[0][am]-u_min)/(u_max-u_min)
            # a = np.array([a, a, a]).T
            # dst[mU[1][am], mU[0][am]] = dst[mU[1][am], mU[0][am]]*(1-a) + src[mV[1][am], mV[0][am]]*a
            # dst[mU[1][~am], mU[0][~am]] = src[mV[1][~am], mV[0][~am]]
            for y in range(ymax):
                ym = mU[1] == y
                if True in ym:
                    am = np.sum(dst[mU[1][ym], mU[0][ym]], axis=1) != 0
                    if True in am:
                        u_max, u_min = max(mU[0][ym][am]), min(mU[0][ym][am])
                        if u_max == u_min:
                            a = 1 - (mU[0][ym][am]-u_min)
                        else:
                            a = 1 - (mU[0][ym][am]-u_min)/(u_max-u_min)
                        a = np.array([a, a, a]).T
                        dst[mU[1][ym][am], mU[0][ym][am]] = dst[mU[1][ym][am], mU[0][ym][am]]*a + \
                                                            src[mV[1][ym][am], mV[0][ym][am]]*(1-a)
                    dst[mU[1][ym][~am], mU[0][ym][~am]] = src[mV[1][ym][~am], mV[0][ym][~am]]
        else:
            dst[mU[1], mU[0]] = src[mV[1], mV[0]]
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # TODO: 6. assign to destination image with proper masking
    elif direction == 'f':
        # ========================== method 1 ==========================
        # # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # V = np.dot(H, U)
        # V = (V/V[2]).astype(int)
        # Vx = V[0].reshape(ymax-ymin, xmax-xmin)
        # Vy = V[1].reshape(ymax-ymin, xmax-xmin)
        # # print('Ux.shape: ', Ux.shape)
        # # print('Ux:\n', Ux)
        # # print('Uy.shape: ', Uy.shape)
        # # print('Uy:\n', Uy)
        # # print('Vx.shape: ', Vx.shape)
        # # print('Vx:\n', Vx)
        # # print('Vy.shape: ', Vy.shape)
        # # print('Vy:\n', Vy)
        # # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # mask = ((Vx >= 0) & (Vx < w_dst)) & ((Vy >= 0) & (Vy < h_dst))
        # # print('mask:\n', mask)
        # # TODO: 5.filter the valid coordinates using previous obtained mask
        # mVx, mVy = Vx[mask], Vy[mask]
        # # TODO: 6. assign to destination image using advanced array indicing
        # dst[mVy, mVx, :] = src[mask]

        # ========================== method 2 ==========================
        V = np.dot(H, U)
        V = np.around(V/V[2]).astype(int)
        mask = ((V[0] >= 0) & (V[0] < w_dst)) & ((V[1] >= 0) & (V[1] < h_dst))
        mVx, mVy = V[0][mask], V[1][mask]
        dst[mVy, mVx] = src[mask.reshape(ymax-ymin, xmax-xmin)]
    return dst







if __name__ == "__main__":
    pass
