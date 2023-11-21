"""General-purpose training script for images-to-images translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the data ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, data, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test_total tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import datetime
import time, os
import copy

# import test_total
import torch

from data.cataract_test_dataset import CataractTestDataset
from model_eval.eval_public import eval_public
from model_eval.evaluate import model_eval, model_eval_ultrasound, fiq_evaluation
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from test_all import set_A2A_opt
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
# from model_eval.evaluate import model_eval
# from model_eval.eval_public import eval_public
from data.EyeQ_segment_pair_dataset import EyeQSegmentPairDataset
from util.metric_logger import MetricLogger
from models.backbone.densenet_mcf import dense121_mcs

def model_test(testOpt, testDataset, model, web_dir, guide, eval_test=True):
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (testOpt.name, testOpt.phase, testOpt.epoch))
    if eval_test:
        model.eval()
    print("hhhhhhhhhhhhhhh",len(testDataset))
    for i, data in enumerate(testDataset):

        # 跑的时候只跑前面的几张图片
        # if i >= testOpt.num_test:  # only apply our model to opt.num_test images.
        #     print('process finish:', i)
        #     break
        model.set_input(data, isTrain=False)  # unpack images from images loader

        model.test()  # run inference
        visuals = model.get_current_visuals()  # get images results
        img_path = model.get_image_paths()  # get images paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th images... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                    guide=guide)
    model.train()
    webpage.save()  # save the HTML

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training optionsq
    dataset = create_dataset(opt)  # create a data given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the data.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    meters = MetricLogger(delimiter="  ")

    # 载入MCF模型
    model_mcf = dense121_mcs(n_class=3)
    # 加载预训练模型,将之前训练好的参数张量填入现在的模型中
    if opt.pre_model is not None:
        loaded_model = torch.load("./pretrain/DenseNet121_v3_v1.tar")
        model_mcf.load_state_dict(loaded_model['state_dict'])

    # 将模型放入gpu
    model_mcf.cuda()
    total_iters = 0                # the total number of training iterations
    max_ssim = max_ssim2 = max_ssim_iter = max_ssim_iter2= 0

    # ----------为了测试的初始化-------------
    # 先拿到训练阶段的通用参数  之后再修改其他参数
    # TODO
    testOpt = copy.deepcopy(opt)
    # testOpt = TestOptions().parse()  # get test_total options
    # testOpt.dataroot = testOpt.test_dataroot_when_training
    # testOpt.dataset_mode = 'cataract_guide_padding' if opt.model in ['cataract_dehaze', 'still_gan'] else opt.dataset_mode
    testOpt.dataroot = opt.test_dataroot_when_training
    testOpt.dataset_mode = opt.test_dataset_mode
    testOpt.phase = 'test_total'
    testOpt.isTrain = False
    testOpt.num_test = 360
    testOpt.target_gt_dir = "gt"
    testOpt.num_threads = 4  # test_total code only supports num_threads = 1
    testOpt.batch_size = 1  # test_total code only supports batch_size = 1
    testOpt.serial_batches = True  # disable images shuffling; comment this line if results on randomly chosen images are needed.
    testOpt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    testOpt.display_id = -1  # no visdom display; the test_total code saves the results to a HTML file.
    testOpt.load_size = testOpt.crop_size = testOpt.test_crop_size
    testDataset = create_dataset(testOpt)  # create a data given opt.dataset_mode and other options

    # define the website directory
    guide = True if 'guide' in opt.dataset_mode else False
    # ------------------------------------
    # # ------------测试模型-----------------
    # # if opt.test_when_train:
    # if opt.test_when_train:
    #     test_web_dir = os.path.join(testOpt.results_dir, testOpt.name,
    #                                 '{}_{}'.format(testOpt.phase, 'latest'))
    #     test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, 0)
    #     print('creating web directory', test_web_dir)
    #     model_test(testOpt, testDataset, model, test_web_dir, guide)
    #     if opt.is_fid_score:
    #         cataractTestDataset = CataractTestDataset(opt, test_web_dir)
    #         ssim = model_eval(testOpt, test_web_dir, wrap=False)
    #         eval_public(opt, cataractTestDataset)
    #     else:
    #         ssim = model_eval(testOpt, test_web_dir)
    #     if ssim > max_ssim:
    #         max_ssim = ssim
    #         # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
    #         # model.save_networks(epoch)
    # # ------------测试模型-----------------
    start_time = datetime.datetime.now()
    print(start_time)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for images loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.在每个epoch开始前更新
        # for i, images in enumerate(dataset_S):  # inner loop within one epoch
        # for i, images in enumerate(zip(dataset_S, dataset_T)):
        # -----可能会对训练产生影响，修改G_DD的学习率------
        # if epoch == int(opt.n_epochs * 0.5) and 'lambda_G_DD' in opt:
        #     opt.lambda_G_DD = opt.lambda_G_DD * 0.6
        # elif epoch == int(opt.n_epochs * 0.9) and 'lambda_G_DD' in opt:
        #     opt.lambda_G_DD = opt.lambda_G_DD * 0.6

        for i, data_source in enumerate(dataset):
            data = data_source
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data,isTrain=True,model=model_mcf)         # unpack images from data and apply preprocessing  数据流

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # 直接将网络初始化时希望可视化的参数送到visualizer
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印loss和可视化loss
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # 第二个epoch重新设置loss
                if epoch == 2:
                    meters = MetricLogger(delimiter="  ")
                meters.update(**losses)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_meters_losses(epoch, epoch_iter, losses, t_comp, t_data, meters)
                # TODO:debug visdom
                if opt.display_id > 0:
                    # TODO:可视化时适当修改dataset_size
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # TODO
            # # 保存网络，根据图像的的iter来，而不是epoch，基本不用，因为iter不是累加的
            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)
            # iter_data_time = time.time()

        # ------------测试模型-----------------
        if opt.test_when_train:
            test_web_dir = os.path.join(testOpt.results_dir, testOpt.name,
                                        '{}_{}'.format(testOpt.phase, 'latest'))
            test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, 0)
            print('creating web directory', test_web_dir)
            model_test(testOpt, testDataset, model, test_web_dir, guide)
            # ssim = model_eval(testOpt, test_web_dir, wrap=False)
            # TODO
            # ssim = fiq_evaluation(testOpt, test_web_dir)
            #ssim = fiq_evaluation(testOpt, test_web_dir)
            ssim = model_eval(testOpt, test_web_dir)
            if ssim[0] > max_ssim:
                max_ssim = ssim[0]
        #         max_ssim_iter = epoch
        #         print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #         model.save_networks(epoch)
        # if opt.test_A2A:
        #
        #     test_web_dir2 = os.path.join(testOpt.results_dir, testOpt.name,
        #                                 '{}_{}'.format(testOpt.phase, 'latest'))
        #     test_web_dir2 = '{:s}_iter{:d}'.format(test_web_dir2, 1)
        #     print('creating web directory', test_web_dir2)
        #     A2ATestOpt, A2ATestDataset = set_A2A_opt(opt)
        #     model_test(A2ATestOpt, A2ATestDataset, model, test_web_dir2, guide)
        #     ssim = model_eval_ultrasound(A2ATestOpt, test_web_dir2, wrap=False)
        #     if ssim[0] > max_ssim2:
        #         max_ssim2 = ssim[0]
        #         max_ssim_iter2 = epoch
        #         print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #         model.save_networks(epoch)
        # ------------测试模型-----------------
        # TODO:测试模型
        # # ------------测试模型-----------------
        # if (opt.test_when_train and epoch % opt.test_freq == 0) or (opt.test_when_train and epoch == 2):
        #     test_web_dir = os.path.join(testOpt.results_dir, testOpt.name,
        #                                 '{}_{}'.format(testOpt.phase, 'latest'))
        #     test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, epoch)
        #     print('creating web directory', test_web_dir)
        #     model_test(testOpt, testDataset, model, test_web_dir, guide, testOpt.eval_test)
        #     if not opt.is_public_test_dataset and (opt.test_rcf or opt.test_fiq):
        #         if opt.is_fid_score:
        #             ssim = model_eval(testOpt, test_web_dir, meters=meters, wrap=False)
        #
        #         else:
        #             ssim = model_eval(testOpt, test_web_dir, meters=meters)
        #     # else:
        #     #     if opt.is_fid_score:
        #     #         cataractTestDataset = CataractTestDataset(opt, test_web_dir)
        #     #         eval_public(opt, cataractTestDataset)
        #     if opt.is_fid_score:
        #         cataractTestDataset = CataractTestDataset(opt, test_web_dir)
        #         eval_public(opt, cataractTestDataset)
        #         print('eval finished:', len(cataractTestDataset))
        #             # if ssim > max_ssim:
        #             #     max_ssim = ssim
        #             #     max_ssim_iter = epoch
        #             # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #             # model.save_networks(epoch)
        #
        # elif epoch % 5 == 0 or epoch == 2:
        #     test_web_dir = os.path.join(testOpt.results_dir, testOpt.name,
        #                                 '{}_{}'.format(testOpt.phase, 'latest'))
        #     test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, epoch)
        #     print('creating web directory', test_web_dir)
        #     model_test(testOpt, testDataset, model, test_web_dir, guide)
        #
        # # ------------测试模型-----------------

        # 保存网络
        if epoch % opt.save_epoch_freq == 0 or epoch == 100:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)




        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    end_time = datetime.datetime.now()
    print(end_time)
    elapsed_time = end_time - start_time
    elapsed_hours = elapsed_time.total_seconds() / 3600  # 1 hour = 3600 seconds

    print(f"The program ran for {elapsed_hours} hours")
    model.save_networks('latest')
    print("max_ssim", max_ssim,"max_epoch",max_ssim_iter)
    print("max_ssim2", max_ssim2, "max_epoch2", max_ssim_iter2)

