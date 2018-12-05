import easydict
import torch

config = easydict.EasyDict()

config.data_path = './dataset/dped/'
config.model_type = {'0': 'blackberry', '1': 'iphone', '2': 'sony'}

config.batch_size = 30
config.height = 100
config.width = 100
config.channels = 3

config.feature_id = 35  # VGG19 ReLU5_4
config.kernel_size = 21
config.sigma = 3

config.g_lr = 1e-3
config.d_lr = 1e-3
config.lambda_color = 5 * 1e-3
config.lambda_texture = 5 * 1e-3
config.lambda_tv = 10

config.train_iters = 1  # 200000
# config.train = True
config.use_cuda = True

# DEPRECATED
config.dtype = torch.FloatTensor
config.gpu_dtype = torch.cuda.FloatTensor
