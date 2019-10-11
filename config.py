from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/dped/'
config.checkpoint_dir = './checkpoints/'
config.tensorboard_dir = './tensorboard/'
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

config.print_step = 10
config.tensorboard_step = 50
config.checkpoint_step = 1000
config.load_iter = 0
config.train_iters = 20000
config.is_train = True
config.use_cuda = True
