import easydict
import torch

config = easydict.EasyDict()

config.data_path = './dataset/dped/'
config.sample_path = './samples/'
config.checkpoint_path = './checkpoints/'
config.model_type = 'blackberry'  # ['blackberry', 'iphone', 'sony']

config.batch_size = 30  # DEFAULT: 30
config.height = 100
config.width = 100
config.channels = 3

config.feature_id = 35  # VGG19 ReLU5_4
config.kernel_size = 21
config.sigma = 3

config.g_lr = 1e-3
config.d_lr = 1e-3
config.lambda_c = 5 * 1e-3
config.lambda_t = 5 * 1e-3
config.lambda_tv = 10

config.resume_iter = 0
config.train_iters = 1  # DEFAULT: 20000
config.train = False
config.use_cuda = True

# DEPRECATED
config.dtype = torch.FloatTensor
config.gpu_dtype = torch.cuda.FloatTensor
