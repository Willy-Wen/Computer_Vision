import numpy as np
import cv2

def normolized(img):
    min = np.min(img)
    max = np.max(img)
    img = (img-min)/(max-min)*255
    return img.astype(np.uint8)


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        image1_0 = image
        image1_1 = cv2.GaussianBlur(image1_0, ksize=(0, 0), sigmaX=self.sigma)
        image1_2 = cv2.GaussianBlur(image1_0, ksize=(0, 0), sigmaX=self.sigma**2)
        image1_3 = cv2.GaussianBlur(image1_0, ksize=(0, 0), sigmaX=self.sigma**3)
        image1_4 = cv2.GaussianBlur(image1_0, ksize=(0, 0), sigmaX=self.sigma**4)

        image2_0 = cv2.resize(image1_4, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        image2_1 = cv2.GaussianBlur(image2_0, ksize=(0, 0), sigmaX=self.sigma)
        image2_2 = cv2.GaussianBlur(image2_0, ksize=(0, 0), sigmaX=self.sigma**2)
        image2_3 = cv2.GaussianBlur(image2_0, ksize=(0, 0), sigmaX=self.sigma**3)
        image2_4 = cv2.GaussianBlur(image2_0, ksize=(0, 0), sigmaX=self.sigma**4)

        # cv2.imshow('0', image1_0)
        # cv2.imshow('1', image1_1)
        # cv2.imshow('2', image1_2)
        # cv2.imshow('3', image1_3)
        # cv2.imshow('4', image1_4)
        # cv2.waitKey()

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)

        sub1_10 = cv2.subtract(image1_1, image1_0)
        sub1_21 = cv2.subtract(image1_2, image1_1)
        sub1_32 = cv2.subtract(image1_3, image1_2)
        sub1_43 = cv2.subtract(image1_4, image1_3)

        sub2_10 = cv2.subtract(image2_1, image2_0)
        sub2_21 = cv2.subtract(image2_2, image2_1)
        sub2_32 = cv2.subtract(image2_3, image2_2)
        sub2_43 = cv2.subtract(image2_4, image2_3)

        dog_images1 = np.array([sub1_10, sub1_21, sub1_32, sub1_43])
        dog_images2 = np.array([sub2_10, sub2_21, sub2_32, sub2_43])

        # cv2.imshow('1-1', normolized(sub1_10))
        # cv2.imshow('1-2', normolized(sub1_21))
        # cv2.imshow('1-3', normolized(sub1_32))
        # cv2.imshow('1-4', normolized(sub1_43))
        # cv2.imshow('2-1', normolized(sub2_10))
        # cv2.imshow('2-2', normolized(sub2_21))
        # cv2.imshow('2-3', normolized(sub2_32))
        # cv2.imshow('2-4', normolized(sub2_43))
        # cv2.waitKey()
        # print(dog_images1.shape)
        # print(dog_images2.shape)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = []

        for z in range(1, dog_images1.shape[0]-1):
            for y in range(1, dog_images1.shape[1]-1):
                for x in range(1, dog_images1.shape[2]-1):
                    point = dog_images1[z, y, x]
                    around_points = dog_images1[z-1:z+2, y-1:y+2, x-1:x+2]
                    local_min = np.min(around_points)
                    local_max = np.max(around_points)
                    if (point == local_min or point == local_max) and np.abs(point) >= self.threshold:
                        keypoints.append([y, x])

        for z in range(1, dog_images2.shape[0]-1):
            for y in range(1, dog_images2.shape[1]-1):
                for x in range(1, dog_images2.shape[2]-1):
                    point = dog_images2[z, y, x]
                    around_points = dog_images2[z-1:z+2, y-1:y+2, x-1:x+2]
                    local_min = np.min(around_points)
                    local_max = np.max(around_points)
                    if (point == local_min or point == local_max) and np.abs(point) >= self.threshold:
                        keypoints.append([y*2, x*2])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]

        # print(len(keypoints))
        return keypoints


# if __name__ == '__main__':
#     image_path = 'D:/NTU/CV/hw1/hw1_material/part1/testdata/1.png'
#     img = cv2.imread(image_path, 0).astype(np.float64)
#     DoG = Difference_of_Gaussian(3.0)
#     DoG.get_keypoints(img)
#     # gt = np.load('D:/NTU/CV/hw1/hw1_material/part1/testdata/1_gt.npy')
#     # print(len(gt))