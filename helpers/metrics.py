import torch
import torch.nn.functional as F
import torchmetrics

'''
HOW TO USE:

model = RDSR().cuda()

dataLoader, eval_dataLoader = get_sr20_dataLoaders('', batch_size)

eval_fns = [
    rootMeanSquaredError,
    signalToReconstructionRatioError,
    torchmetrics.SpectralAngleMapper(),
    torchmetrics.PeakSignalNoiseRatio(),
    torchmetrics.StructuralSimilarityIndexMeasure(),
    torchmetrics.ErrorRelativeGlobalDimensionlessSynthesis(ratio=2), # for SR20
    torchmetrics.UniversalImageQualityIndex()
]

eval_v = model_eval(model, eval_dataLoader, eval_fns)
'''

class Metrics:
    @staticmethod
    def rootMeanSquaredError(x, y):
        assert x.size() == y.size(), 'metrics inputs must have same dimension'

        return torch.sqrt(torch.mean((x - y)**2, dim=(0, 2, 3)))

    @staticmethod
    def signalToReconstructionRatioError(x, y):
        assert x.size() == y.size(), 'metrics inputs must have same dimension'

        sq_mean_x = torch.mean(x, dim=(2, 3))**2
        sq_norm_y_x = torch.linalg.matrix_norm(y - x, dim=(2, 3), ord=1)**2
        size = x.size(dim=2) * x.size(dim=3)
        log10 = 10 * torch.log10(sq_mean_x / (sq_norm_y_x / size))

        return torch.mean(log10, dim=0)
    
    @staticmethod
    def spectralAngleMapper(x, y):
        assert x.size() == y.size(), 'metrics inputs must have same dimension'
        
        sum_m = (x * y).sum(dim=(1, 2, 3))
        sq_sum_x = x.sum(dim=(1, 2, 3))**2
        sq_sum_y = y.sum(dim=(1, 2, 3))**2
        div = sum_m / (torch.sqrt(sq_sum_x * sq_sum_y))
            
        return torch.mean(torch.arccos(torch.clamp(div, min=-1, max=1)), dim=0)