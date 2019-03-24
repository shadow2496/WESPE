import os

import torch
import torchvision.models
from torchvision import utils
from skimage.measure import compare_ssim

from config import config
from load_dataset import *
from utils import *
from models import *


def get_feature(model, img_tensor, content_id, device):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, config.channels, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, config.channels, 1, 1)
    img_normalized = (img_tensor - mean) / std
    # print('img_normalized mean : ', img_normalized.permute(1, 0, 2, 3).reshape(config.channels, -1).mean(1))
    # print('img_normalized std : ', img_normalized.permute(1, 0, 2, 3).reshape(config.channels, -1).std(1))
    feature = model.features[:content_id](img_normalized)
    # feature = feature.data.squeeze().cpu().numpy().transpose(1, 2, 0)
    return feature


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
    vgg19 = torchvision.models.vgg19(pretrained=True).to(device)

    true_labels = torch.ones(config.batch_size, dtype=torch.long).to(device)
    false_labels = torch.zeros(config.batch_size, dtype=torch.long).to(device)
    for idx in range(config.resume_iter, config.train_iters):
        train_phone, train_dslr = load_train_data(config.dataset_dir, config.phone, config.batch_size,
                                                  config.width * config.height * config.channels)
        x = torch.from_numpy(train_phone).float()
        y_real = torch.from_numpy(train_dslr).float()
        x = x.view(-1, config.channels, config.height, config.width).to(device)
        y_real = y_real.view(-1, config.channels, config.height, config.width).to(device)

        # --------------------------------------------------------------------------------------------------------------
        #                                                Train generators
        # --------------------------------------------------------------------------------------------------------------
        y_fake = models.gen_g(x)
        x_rec = models.gen_f(y_fake)
        # print('y_fake shape : ', y_fake.size())
        # print('x_rec shape : ', x_rec.size())

        # content loss
        feat_x = get_feature(vgg19, x, config.content_id, device)
        feat_x_rec = get_feature(vgg19, x_rec, config.content_id, device)
        # print('feat_x shape : ', feat_x.size())
        # print('feat_x_rec : ', feat_x_rec.size())
        loss_content = torch.pow(feat_x.detach() - feat_x_rec, 2).mean()

        # color loss
        # gaussian blur image for discriminator_c
        fake_blur = gaussian_blur(y_fake, config.kernel_size, config.sigma, config.channels, device)
        # print('fake blur image shape : ', fake_blur.size())
        # print('real blur image shape : ', real_blur.size())
        logits_fake_blur = models.dis_c(fake_blur)
        loss_c = models.criterion(logits_fake_blur, true_labels)

        # texture loss
        # gray-scale image for discriminator_t
        fake_gray = gray_scale(y_fake)
        # print('fake grayscale image shape : ', fake_gray.size())
        # print('real grayscale image shape : ', real_gray.size())
        logits_fake_gray = models.dis_t(fake_gray)
        loss_t = models.criterion(logits_fake_gray, true_labels)

        # total variation loss
        # need to know why it is calculated this way
        height_tv = torch.pow(y_fake[:, :, 1:, :] - y_fake[:, :, :config.height - 1, :], 2).mean()
        width_tv = torch.pow(y_fake[:, :, :, 1:] - y_fake[:, :, :, :config.width - 1], 2).mean()
        loss_tv = height_tv + width_tv

        # all loss sum
        gen_loss = loss_content + config.lambda_c * loss_c + config.lambda_t * loss_t + config.lambda_tv * loss_tv

        models.g_optimizer.zero_grad()
        models.f_optimizer.zero_grad()
        gen_loss.backward()
        models.g_optimizer.step()
        models.f_optimizer.step()

        # --------------------------------------------------------------------------------------------------------------
        #                                              Train discriminators
        # --------------------------------------------------------------------------------------------------------------
        y_fake = models.gen_g(x)

        fake_blur = gaussian_blur(y_fake, config.kernel_size, config.sigma, config.channels, device)
        real_blur = gaussian_blur(y_real, config.kernel_size, config.sigma, config.channels, device)
        logits_fake_blur = models.dis_c(fake_blur.detach())
        logits_real_blur = models.dis_c(real_blur.detach())
        loss_dc = models.criterion(logits_real_blur, true_labels) + models.criterion(logits_fake_blur, false_labels)

        fake_gray = gray_scale(y_fake)
        real_gray = gray_scale(y_real)
        logits_fake_gray = models.dis_t(fake_gray.detach())
        logits_real_gray = models.dis_t(real_gray.detach())
        loss_dt = models.criterion(logits_real_gray, true_labels) + models.criterion(logits_fake_gray, false_labels)

        # dis_loss = config.lambda_c * loss_dc + config.lambda_t * loss_dt
        dis_loss = loss_dc + loss_dt

        models.c_optimizer.zero_grad()
        models.t_optimizer.zero_grad()
        dis_loss.backward()
        models.c_optimizer.step()
        models.t_optimizer.step()

        print('Iteration : {}/{}, Gen_loss : {:.4f}, Dis_loss : {:.4f}'.format(
            idx + 1, config.train_iters, gen_loss.data, dis_loss.data))
        print('Loss_content : {:.4f}, Loss_c : {:.4f}, Loss_t : {:.4f}, Loss_tv: {:.4f}'.format(
            loss_content.data, loss_c.data, loss_t.data, loss_tv.data))
        print('Loss_dc : {:.4f}, Loss_dt : {:.4f}'.format(loss_dc.data, loss_dt.data))

        if (idx + 1) % 1000 == 0:
            sample_path = os.path.join(config.sample_dir, config.phone)
            checkpoint_path = os.path.join(config.checkpoint_dir, config.phone)

            utils.save_image(x, os.path.join(sample_path, '{}-x.jpg'.format(idx + 1)))
            utils.save_image(x_rec, os.path.join(sample_path, '{}-x_rec.jpg'.format(idx + 1)))
            utils.save_image(y_fake, os.path.join(sample_path, '{}-y_fake.jpg'.format(idx + 1)))
            utils.save_image(y_real, os.path.join(sample_path, '{}-y_real.jpg'.format(idx + 1)))
            utils.save_image(fake_blur, os.path.join(sample_path, '{}-fake_blur.jpg'.format(idx + 1)))
            utils.save_image(real_blur, os.path.join(sample_path, '{}-real_blur.jpg'.format(idx + 1)))
            utils.save_image(fake_gray, os.path.join(sample_path, '{}-fake_gray.jpg'.format(idx + 1)))
            utils.save_image(real_gray, os.path.join(sample_path, '{}-real_gray.jpg'.format(idx + 1)))

            torch.save(models.gen_g.state_dict(), os.path.join(checkpoint_path, '{}-Gen_g.ckpt'.format(idx + 1)))
            torch.save(models.gen_f.state_dict(), os.path.join(checkpoint_path, '{}-Gen_f.ckpt'.format(idx + 1)))
            torch.save(models.dis_c.state_dict(), os.path.join(checkpoint_path, '{}-Dis_c.ckpt'.format(idx + 1)))
            torch.save(models.dis_t.state_dict(), os.path.join(checkpoint_path, '{}-Dis_t.ckpt'.format(idx + 1)))
            print('Saved intermediate images and model checkpoints.')


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
    models = WESPE(config, device)
    if config.resume_iter != 0:
        load_checkpoints(models)

    if config.train:
        train(models, device)
    else:
        test(models, device)


if __name__ == '__main__':
    main()
