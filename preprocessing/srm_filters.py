import numpy as np
import cv2

class SRMFilter:

    def __init__(self):
        self.kernels = 12
        self.normalisationRange = 255.0
        self.kernel_bank = self.get_srm_kernels() # build kernels once

    # method to get the SRM HPF kernels
    def get_srm_kernels(self):

        kernels = [] # filter bank

    # Detect Gradients, i.e L/R differences
        # 1st filter = horizonatl 1st order filter
        k1 = np.array( [
          [0, 0, 0 ,0 ,0],
          [0, 0, 0 ,0 ,0],
          [0, 0.5, -1 ,0.5 ,0],
          [0, 0, 0 ,0 ,0],
          [0, 0, 0 ,0 ,0],
        ], dtype=np.float32)
        kernels.append(k1)

        # 2nd filter = vertical 1st order filter
        k2 = k1.T.copy()
        kernels.append(k2)

    # Detect change of change
        # 3rd filter = 2nd order horizontal filter
        k3 = np.array([
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, -0.5, 1, -0.5, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
          ], dtype=np.float32)
        kernels.append(k3)

        # 4th filter = vertical 2nd order filter
        k4 = k3.T.copy()
        kernels.append(k4)

    # Detect directional noise patterns & diagonal inconsistencies
        # 5th filter = diagonal filter
        k5 = np.array([
                          [0, 0, 0, 0, 0],
                          [0, 0.5, 0, -0.5, 0],
                          [0, 0, 0, 0, 0],
                          [0, -0.5, 0, 0.5, 0],
                          [0, 0, 0, 0, 0],
                      ], dtype=np.float32)
        kernels.append(k5)

    # Detect transitions, sharper boundaries
        # 6th filter = edge filter
        k6 = np.array([
                          [0, 0, -1, 0, 0],
                          [0, 0, 2, 0, 0],
                          [0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                      ], dtype=np.float32)
        kernels.append(k6)

    # Detect neighbor inconsistencies
        # 7th filter = Laplacian filter
        k7 = np.array([
                          [0, 0, -1, 0, 0],
                          [0, -1, -2, -1, 0],
                          [-1, -2, 12, -2, -1],
                          [0, -1, -2, -1, 0],
                          [0, 0, -1, 0, 0],
                      ], dtype=np.float32)
        kernels.append(k7)

    # Detect inconsistencies in smooth regions
        # 8th filter = low-frequency smooth regions
        k8 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0.25, 0.25, 0.25, 0],
            [0, 0.25, -1.0, 0.25, 0],
            [0, 0.25, 0.25, 0.25, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        kernels.append(k8)

    # Detect cross diagonal inconsistencies
        # 9th filter = cross diagonal filter
        k9 = np.array([
            [0, 0, -0.5, 0, 0],
            [0, 0.5, 0, 0.5, 0],
            [-0.5, 0, 1, 0, -0.5],
            [0, 0.5, 0, 0.5, 0],
            [0, 0, -0.5, 0, 0],
        ], dtype=np.float32)
        kernels.append(k9)

    # Amplify high frequency chnages,
        # 10th filter = high-frequency noise amplification
        k10 = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1],
        ], dtype=np.float32)
        kernels.append(k10)

        # Horizontal daubechies-inspired wavelet filter
        k11 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-0.0545, 0.3129, -0.6756, 0.5854, -0.0158],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        kernels.append(k11)

        k12 = k11.T.copy()
        kernels.append(k12)

        return kernels

    """
    method to apply filters and get the residual map. SRM feature extraction
    Params: stego image and normalise = true
    Return: noise residual map
    """
    def extract_noise_residuals(self, image : np.ndarray, normalise: bool = True) -> np.ndarray:

    # convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        imgFloat = image.astype(np.float32)


        results = [] # residual map storage array

    # apply filters
        for k in self.kernel_bank:
            result = cv2.filter2D(imgFloat, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
            results.append(result)
        return results



