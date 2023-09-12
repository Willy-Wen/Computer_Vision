import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]

    last_best_H = np.eye(3)
    threshold = 0.3
    n_iter = 3000
    n_sample = 10
    n_best_kps = 100

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        img1 = imgs[idx]    # query_img
        img2 = imgs[idx+1]  # train_img

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:n_best_kps]

        # print('\n', matches[0].queryIdx, matches[0].trainIdx, matches[0].distance)
        # print(kp1[matches[0].queryIdx].pt, kp2[matches[0].trainIdx].pt)
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('img3', img3), cv2.waitKey()

        query_idx = [match.queryIdx for match in matches]
        train_idx = [match.trainIdx for match in matches]
        # print('\n', 'query_idx', '\n', query_idx[:3], '\n', 'train_idx', '\n', train_idx[:3])

        dst_points = np.array([kp1[i].pt for i in query_idx])
        src_points = np.array([kp2[i].pt for i in train_idx])
        # print('\n', 'dst_points', '\n', dst_points[:3], '\n', 'src_points', '\n', src_points[:3])

        # TODO: 2. apply RANSAC to choose best H
        max_inlier = 0
        best_H = np.eye(3)
        for _ in range(n_iter):
            # print(dst_points.shape[0])
            rand_idx = random.sample(range(dst_points.shape[0]), n_sample)
            p1, p2 = dst_points[rand_idx], src_points[rand_idx]
            H = solve_homography(p2, p1)
            U = np.concatenate((src_points.T, [np.ones(src_points.shape[0])]))
            # print('\n', U[:, :5], '\n')
            pred = np.dot(H, U)
            pred = pred/pred[2]
            # print('\n', pred[:, :5], '\n')
            error = np.linalg.norm(pred[:2, :].T-dst_points, axis=1)
            # print('\n', error, '\n')
            inlier = (error<threshold).sum()
            # print('\n', inlier, '\n')
            if inlier > max_inlier:
                max_inlier = inlier
                best_H = H.copy()

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        dst = warping(img2, dst, last_best_H, 0, h_max, 0, w_max, direction='b', blending=True)
        # dst = warping(img2, dst, last_best_H, 0, h_max, 0, w_max, direction='b', blending=False)
    out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)