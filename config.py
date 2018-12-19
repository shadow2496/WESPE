import easydict

config = easydict.EasyDict()

config.data_path = './dataset/dped/'
config.sample_path = './samples/'
config.checkpoint_path = './checkpoints/'
config.model_type = 'blackberry'  # ['blackberry', 'iphone', 'sony']

config.batch_size = 35  # DEFAULT: 30
config.height = 100
config.width = 100
config.channels = 3

config.feature_id = 35  # VGG19 ReLU5_4
config.kernel_size = 21
config.sigma = 3

config.g_lr = 2e-4
config.d_lr = 2e-4
config.lambda_c = 5 * 1e-3
config.lambda_t = 5 * 1e-3
config.lambda_tv = 10

config.resume_iter = 0
config.train_iters = 20000  # DEFAULT: 20000
config.train = True
config.use_cuda = True
