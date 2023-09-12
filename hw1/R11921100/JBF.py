import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

    ############### Method_1 ###############
    # def joint_bilateral_filter(self, img, guidance):
    #     p = self.pad_w
    #     BORDER_TYPE = cv2.BORDER_REFLECT
    #     padded_img = cv2.copyMakeBorder(img, p, p, p, p, BORDER_TYPE).astype(np.int32)
    #     padded_guidance = cv2.copyMakeBorder(guidance, p, p, p, p, BORDER_TYPE).astype(np.int32)
    #
    #     ### TODO ###
    #     output = np.zeros(img.shape)
    #
    #     grid = np.mgrid[-p:p+1, -p:p+1]
    #     gs_table = grid[0]**2 + grid[1]**2
    #     gs = np.exp(1)**(-gs_table/(2*(self.sigma_s**2)))
    #
    #     h, w = padded_guidance.shape[0], padded_guidance.shape[1]
    #     for z in range(3):
    #         for x in range(p, h-p):
    #             for y in range(p, w-p):
    #                 if len(padded_guidance.shape) == 2:
    #                     gr_table = ((padded_guidance[x, y] - padded_guidance[x-p:x+p+1, y-p:y+p+1])/255.)**2
    #                     gr = np.exp(1)**(-gr_table/(2*(self.sigma_r**2)))
    #                 else:
    #                     gr_table = ((padded_guidance[x, y, :] - padded_guidance[x-p:x+p+1, y-p:y+p+1, :])/255.)**2
    #                     gr = np.exp(1)**(-np.sum(gr_table, axis=2)/(2*(self.sigma_r**2)))
    #                 gs_gr = gs*gr
    #                 output[x-p, y-p, z] = np.sum(gs_gr*padded_img[x-p:x+p+1, y-p:y+p+1, z])/np.sum(gs*gr)
    #
    #     return np.clip(output, 0, 255).astype(np.uint8)



    ############### Method_2 ###############
    # reference: https://github.com/Spheluo/Joint-Bilateral-Filter/blob/main/JBF.py
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w,
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w,
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        gs_table = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)

        gr_table = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)

        sum = np.zeros(padded_img.shape)
        result = np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w + 1):
            for y in range(-self.pad_w, self.pad_w + 1):
                dT = gr_table[np.abs(np.roll(padded_guidance, [y, x], axis=[0, 1])-padded_guidance)]
                gr = dT if dT.ndim == 2 else np.prod(dT, axis=2)
                gs = gs_table[np.abs(x)] * gs_table[np.abs(y)]
                gs_gr = gs * gr
                padded_img_roll = np.roll(padded_img, [y, x], axis=[0, 1])
                for c in range(padded_img.ndim):
                    result[:, :, c] += padded_img_roll[:, :, c]*gs_gr
                    sum[:, :, c] += gs_gr
        output = (result/sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]

        return np.clip(output, 0, 255).astype(np.uint8)




# if __name__ == '__main__':
#     image_path = 'D:/NTU/CV/hw1/hw1_material/part2/testdata/ex.png'
#     gt_bf_path = 'D:/NTU/CV/hw1/hw1_material/part2/testdata/ex_gt_bf.png'
#     gt_jbf_path = 'D:/NTU/CV/hw1/hw1_material/part2/testdata/ex_gt_jbf.png'
#
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     bf_gt = cv2.cvtColor(cv2.imread(gt_bf_path), cv2.COLOR_BGR2RGB)
#     jbf_gt = cv2.cvtColor(cv2.imread(gt_jbf_path), cv2.COLOR_BGR2RGB)
#
#     JBF = Joint_bilateral_filter(3, 0.1)
#     output = JBF.joint_bilateral_filter(img_rgb, img_rgb)
#     print(False in (output == bf_gt))
