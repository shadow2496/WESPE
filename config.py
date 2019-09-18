from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/dped/'
config.checkpoint_dir = './checkpoints/'
config.sample_dir = './samples/'
config.phone = 'blackberry'  # ['blackberry', 'iphone', 'sony']

config.batch_size = 30
config.width = 100
config.height = 100
config.channels = 3

config.content_id = 36  # VGG19 ReLU5_4
config.kernel_size = 21
config.sigma = 3

config.gen_lr = 5e-4
config.dis_lr = 5e-4
config.w_content = 1
config.w_color = 5e-3
config.w_texture = 5e-3
config.w_tv = 10

config.resume_iter = 0
config.train_iters = 20000
config.train = True
config.use_cuda = True
