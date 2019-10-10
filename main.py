import os

import torch
import torchvision.models
from torchvision import utils
from skimage.measure import compare_ssim

from config import config
from load_dataset import load_train_data, load_test_data
from models import WESPE
from utils import *  #?


def load_checkpoints(model):
    print('Loading the model checkpoints from iter {}...'.format(config.resume_iter))
    checkpoint_path = os.path.join(config.checkpoint_dir, config.phone)

    gen_g_path = os.path.join(checkpoint_path, '{}-Gen_g.ckpt'.format(config.resume_iter))
    gen_f_path = os.path.join(checkpoint_path, '{}-Gen_f.ckpt'.format(config.resume_iter))
    model.gen_g.load_state_dict(torch.load(gen_g_path, map_location=lambda storage, loc: storage))
    model.gen_f.load_state_dict(torch.load(gen_f_path, map_location=lambda storage, loc: storage))

    if config.train:
        dis_c_path = os.path.join(checkpoint_path, '{}-Dis_c.ckpt'.format(config.resume_iter))
        dis_t_path = os.path.join(checkpoint_path, '{}-Dis_t.ckpt'.format(config.resume_iter))
        model.dis_c.load_state_dict(torch.load(dis_c_path, map_location=lambda storage, loc: storage))
        model.dis_t.load_state_dict(torch.load(dis_t_path, map_location=lambda storage, loc: storage))


def train(models, device):
    vgg19 = torchvision.models.vgg19(pretrained=True)
    for param in vgg19.parameters():
        param.required_grad = False
    vgg19.to(device)

    real_labels = torch.ones((config.batch_size, 1), device=device)
    fake_labels = torch.zeros((config.batch_size, 1), device=device)
    for idx in range(config.resume_iter, config.train_iters):
        train_phone, train_dslr = load_train_data(config.dataset_dir, config.phone, config.batch_size,
                                                  (config.channels, config.height, config.width))
        train_phone = torch.as_tensor(train_phone, device=device)
        train_dslr = torch.as_tensor(train_dslr, device=device)

        # --------------------------------------------------------------------------------------------------------------
        #                                              Train discriminators
        # --------------------------------------------------------------------------------------------------------------
        enhanced, phone_rec = models(train_phone)

        # 1) Adversarial color loss
        dslr_blur = gaussian_blur(train_dslr, config.kernel_size, config.sigma, config.channels, device)
        dslr_blur_logits = models.dis_c(dslr_blur)
        enhanced_blur = gaussian_blur(enhanced.detach(), config.kernel_size, config.sigma, config.channels, device)
        enhanced_blur_logits = models.dis_c(enhanced_blur)
        dis_loss_color = models.bce_criterion(dslr_blur_logits, real_labels) \
                         + models.bce_criterion(enhanced_blur_logits, fake_labels)

        # 2) Adversarial texture loss
        dslr_gray = rgb_to_gray(train_dslr, device)
        dslr_gray_logits = models.dis_t(dslr_gray)
        enhanced_gray = rgb_to_gray(enhanced.detach(), device)
        enhanced_gray_logits = models.dis_t(enhanced_gray)
        dis_loss_texture = models.bce_criterion(dslr_gray_logits, real_labels) \
                           + models.bce_criterion(enhanced_gray_logits, fake_labels)

        # Sum of losses
        dis_loss = dis_loss_color + dis_loss_texture

        models.dis_optimizer.zero_grad()
        dis_loss.backward()
        models.dis_optimizer.step()

        # --------------------------------------------------------------------------------------------------------------
        #                                                Train generators
        # --------------------------------------------------------------------------------------------------------------
        # 1) Content consistency loss
        phone_vgg = get_content(vgg19, train_phone, config.content_id, device)
        phone_rec_vgg = get_content(vgg19, phone_rec, config.content_id, device)
        gen_loss_content = models.mse_criterion(phone_vgg, phone_rec_vgg)

        # 2) Adversarial color loss
        enhanced_blur = gaussian_blur(enhanced, config.kernel_size, config.sigma, config.channels, device)
        enhanced_blur_logits = models.dis_c(enhanced_blur)
        gen_loss_color = models.bce_criterion(enhanced_blur_logits, real_labels)

        # 3) Adversarial texture loss
        enhanced_gray = rgb_to_gray(enhanced, device)
        enhanced_gray_logits = models.dis_t(enhanced_gray)
        gen_loss_texture = models.bce_criterion(enhanced_gray_logits, real_labels)

        # 4) TV loss
        y_tv = models.mse_criterion(enhanced[:, :, 1:, :], enhanced[:, :, :-1, :])
        x_tv = models.mse_criterion(enhanced[:, :, :, 1:], enhanced[:, :, :, :-1])
        gen_loss_tv = y_tv + x_tv

        # Sum of losses
        gen_loss = config.w_content * gen_loss_content + config.w_color* gen_loss_color \
                   + config.w_texture * gen_loss_texture + config.w_tv * gen_loss_tv

        models.gen_optimizer.zero_grad()
        gen_loss.backward()
        models.gen_optimizer.step()

        if (idx + 1) % config.print_step == 0:
            print("Iteration: {}/{}, gen_loss: {:.4f}, dis_loss: {:.4f}".format(
                idx + 1, config.train_iters, gen_loss.item(), dis_loss.item()))
            print("gen_loss_content: {:.4f}, gen_loss_color: {:.4f}, gen_loss_texture: {:.4f}, gen_loss_tv: {:.4f}".format(
                gen_loss_content.item(), gen_loss_color.item(), gen_loss_texture.item(), gen_loss_tv.item()))
            print("dis_loss_color: {:.4f}, dis_loss_texture: {:.4f}".format(dis_loss_color.item(), dis_loss_texture.item()))

        if (idx + 1) % config.checkpoint_step == 0:
            sample_path = os.path.join(config.sample_dir, config.phone)
            checkpoint_path = os.path.join(config.checkpoint_dir, config.phone)

            utils.save_image(train_phone, os.path.join(sample_path, '{:05d}-phone.jpg'.format(idx + 1)))
            utils.save_image(phone_rec, os.path.join(sample_path, '{:05d}-phone_rec.jpg'.format(idx + 1)))
            utils.save_image(enhanced, os.path.join(sample_path, '{:05d}-enhanced.jpg'.format(idx + 1)))
            utils.save_image(train_dslr, os.path.join(sample_path, '{:05d}-dslr.jpg'.format(idx + 1)))
            utils.save_image(enhanced_blur, os.path.join(sample_path, '{:05d}-enhanced_blur.jpg'.format(idx + 1)))
            utils.save_image(dslr_blur, os.path.join(sample_path, '{:05d}-dslr_blur.jpg'.format(idx + 1)))
            utils.save_image(enhanced_gray, os.path.join(sample_path, '{:05d}-enhanced_gray.jpg'.format(idx + 1)))
            utils.save_image(dslr_gray, os.path.join(sample_path, '{:05d}-dslr_gray.jpg'.format(idx + 1)))

            torch.save(models.gen_g.state_dict(), os.path.join(checkpoint_path, '{:05d}-Gen_g.ckpt'.format(idx + 1)))
            torch.save(models.gen_f.state_dict(), os.path.join(checkpoint_path, '{:05d}-Gen_f.ckpt'.format(idx + 1)))
            torch.save(models.dis_c.state_dict(), os.path.join(checkpoint_path, '{:05d}-Dis_c.ckpt'.format(idx + 1)))
            torch.save(models.dis_t.state_dict(), os.path.join(checkpoint_path, '{:05d}-Dis_t.ckpt'.format(idx + 1)))
            print("Saved intermediate images and model checkpoints.")


def test(model, device):
    test_path = config.dataset_dir + config.phone + '/test_data/patches/canon/'
    test_image_num = len([name for name in os.listdir(test_path)
                         if os.path.isfile(os.path.join(test_path, name))]) // config.batch_size * config.batch_size

    score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar = 0.0, 0.0, 0.0, 0.0
    for start in range(0, test_image_num, config.batch_size):
        end = min(start + config.batch_size, test_image_num)
        test_phone, test_dslr = load_test_data(config.phone, config.dataset_dir, start, end,
                                               config.height * config.width * config.channels)
        x = torch.from_numpy(test_phone).float()
        y_real = torch.from_numpy(test_dslr).float()
        x = x.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)
        y_real = y_real.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)

        y_fake = model.gen_g(x)

        # Calculate PSNR & SSIM scores
        score_psnr += psnr(y_fake, y_real) * config.batch_size

        y_fake_np = y_fake.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_real_np = y_real.cpu().numpy().transpose(0, 2, 3, 1)
        temp_ssim, _ = compare_ssim(y_fake_np, y_real_np, multichannel=True, gaussian_weights=True, full=True)
        score_ssim_skimage += (temp_ssim * config.batch_size)

        temp_ssim, _ = ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5)
        score_ssim_minstar += temp_ssim * config.batch_size

        score_msssim_minstar += multi_scale_ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5) * config.batch_size
        print('PSNR & SSIM scores of {} images are calculated.'.format(end))

    score_psnr /= test_image_num
    score_ssim_skimage /= test_image_num
    score_ssim_minstar /= test_image_num
    score_msssim_minstar /= test_image_num
    print('PSNR : {:.4f}, SSIM_skimage : {:.4f}, SSIM_minstar : {:.4f}, SSIM_msssim: {:.4f}'.format(
        score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar))


def main():
    if not os.path.exists(os.path.join(config.sample_dir, config.phone)):
        os.makedirs(os.path.join(config.sample_dir, config.phone))
    if not os.path.exists(os.path.join(config.checkpoint_dir, config.phone)):
        os.makedirs(os.path.join(config.checkpoint_dir, config.phone))

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    models = WESPE(config).to(device)
    if config.resume_iter != 0:
        load_checkpoints(models)

    if config.is_train:
        models.train()
        train(models, device)
    else:
        models.eval()
        test(models, device)


if __name__ == '__main__':
    main()
