
import numpy as np
from scipy.stats import skew, kurtosis
import pywt

# extracts stats features from SRM residual maps : histograms + co-occurrence features
class FeatureVector:
    def __init__(self):
        self.coOccurrenceMatrix = None
        self.mean =0.0
        self.variance = 0.0
        self.skewness = 0.0
        self.fvector = None

#Computation

    def computeHistogramFeatures(self,residual:np.ndarray) -> np.ndarray:
        # computes and returns an array of 7 1st order stats descriptors from residual map

        flat = residual.flatten()
        features = np.array([
            np.mean(flat), # mean = avg residual value
            np.var(flat), # variance= spread of values
            float(skew(flat)), # skewness=L/R asymmetry of distribution
            float(kurtosis(flat)), # kurtosis = tail heaviness
            np.min(flat), # min residual value
            np.max(flat), # max res value
            np.max(flat) - np.min(flat), #range= total spread of values
        ], dtype = np.float32)

        return features

    def computeCooccurrenceFeatures(self, residual:np.ndarray, quantisationStep: int=4, max_val:int=8) -> np.ndarray:
        # computes co-occurence matrix features from the residual map

        #quantise reduces the rabge of values from large floats to small ints range, to make matrix manageable
        q = np.round(residual / quantisationStep).astype(np.int32)

        # clip the range
        q=np.clip(q,-max_val,max_val)
        size = 2* max_val + 1 # matrix size

        def build_matrix(p, r):
            idx = (p + max_val) * size + (r + max_val)
            counts = np.bincount(idx.ravel(), minlength=size * size)
            return counts.reshape(size, size).astype(np.float64)


        # horizontal co-occurrence
        h_matrix = build_matrix(q[:, :-1],q[:, 1:])
        v_matrix = build_matrix(q[:-1, :],q[1:, :])

        # Normalization
        if h_matrix.sum() > 0:
            h_matrix = h_matrix / h_matrix.sum()
        if v_matrix.sum() > 0:
            v_matrix = v_matrix / v_matrix.sum()

        # symmetrise
        h_sym = (h_matrix + h_matrix.T) / 2
        v_sym = (v_matrix + v_matrix.T) / 2

        # get upper triangle
        idx = np.triu_indices(size)
        self.coOccurrenceMatrix = h_sym # store horizontal matrix for debugging

        return np.concatenate([h_sym[idx], v_sym[idx]]).astype(np.float32)

    def computeSPAMFeatures(self,residual:np.ndarray, quantisationStep: int=4, max_val:int = 4 ) -> np.ndarray:
    #     Subtractive Pixel Adjacency Matrix (SPAM) features - computes pixel differences in 4 directions

        h_diff = residual[:, 1:] - residual[:, :-1] # horizontal
        v_diff = residual[1:, :] - residual[:-1, :] # vertical
        d_diff = residual[1:, 1:] - residual[:-1, 1:] # diagonal
        ad_diff = residual[1:, :-1] - residual[:-1, 1:] # anti-diagonal

        size = 2* max_val + 1
        features = []


        for diff in [h_diff, v_diff, d_diff, ad_diff]:
            # quantisation and clipping
            q = np.round(diff / quantisationStep).astype(np.int32)
            q = np.clip(q,-max_val,max_val)

            # builds the co-occurrence matrix
            def build_matrix(p, r):
                idx = (p + max_val) * size + (r + max_val)
                counts = np.bincount(idx.ravel(), minlength=size * size)
                return counts.reshape(size, size).astype(np.float64)

    #         H co-occurrence of the difference map
            if q.shape[1] > 1:
                matrix = build_matrix(q[:, :-1], q[:, 1:])
            else:
                matrix = build_matrix(q[:, :-1], q[:, 1:])

    #         normalisation and symmetrisation
            if matrix.sum() > 0:
                matrix = matrix / matrix.sum()
            matrix = (matrix + matrix.T) / 2

    #         get upper triangle
            triangle_idx = np.triu_indices(size)
            features.append(matrix[triangle_idx])

        return np.concatenate(features).astype(np.float32)

    def computeWaveletFeatures(self,residual:np.ndarray) -> np.ndarray:
    #     computes wavelet domain features - useful for WOW  S-UNIWARD

        cA, (cH, cD, cV) = pywt.dwt2(residual, 'db8')

        features = []
        for subband in [cA, cH, cD, cV]:
            flat = subband.flatten()
            features.extend([
                float(np.mean(flat)),
                float(np.var(flat)),
                float(skew(flat)),
                float(np.sum(flat ** 2)),
            ])

        return np.array(features, dtype=np.float32)

    def computeLBPFeatures(self, residual: np.ndarray) -> np.ndarray:
    #     Local Binary Pattern (LBP)

        try:
            from skimage.feature import local_binary_pattern
        except ImportError:
            return np.zeros(59, dtype=np.float32)

        # normalisation
        norm = np.clip(residual, 0, 255).astype(np.uint8)

        lbp = local_binary_pattern(norm,  P=8, R=1, method='uniform')

    #     histogram of pattern frequencies
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)

        return hist.astype(np.float32)

    # interface

    def compute(self, residuals: list[np.ndarray]) -> 'FeatureVector':
        # computes and stores full feature vector from a residual map
        # histogram + co-occurrence features

        all_features = []

        for res in residuals:

            hist_features = self.computeHistogramFeatures(res)
            cooc_features = self.computeCooccurrenceFeatures(res)

            spam_features = self.computeSPAMFeatures(res)
            wavelet_features = self.computeWaveletFeatures(res)
            lbp_features = self.computeLBPFeatures(res)

            all_features.append(hist_features)
            all_features.append(cooc_features)
            all_features.append(spam_features)
            all_features.append(wavelet_features)
            all_features.append(lbp_features)

            # store stats for UI
            self.mean = float(np.mean(res))
            self.variance = float(np.var(res))
            self.skewness = float(skew(res.flatten()))

        # combine into 1 vector
        self.fvector = np.concatenate(all_features)

        return self


    def getFv(self) -> np.ndarray:
        # returns the fv
        if self.fvector is None:
            raise ValueError("Feature vector is empty.")
        return self.fvector

    def validate(self) -> bool:
        #validates creation of fv

        if self.fvector is None:
            return False
        if np.any(np.isnan(self.fvector)) or np.any(np.isinf(self.fvector)):
            return False

        return True





















