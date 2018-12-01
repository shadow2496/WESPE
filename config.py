import easydict
import torch

config = easydict.EasyDict()

config.tv_weight = 10
config.data_path = './dataset/dped/'
config.model_type = {'0': 'blackberry', '1':'iphone', '2':'sony'}
config.batch_size = 30
config.height = 100
config.width = 100
config.channels = 3
config.kernel_size = 21
config.sigma = 3
config.dtype = torch.FloatTensor
config.gpu_dtype = torch.cuda.FloatTensor
