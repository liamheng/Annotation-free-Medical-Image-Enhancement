import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_batch_images
from util import html
import copy
import time
from model_eval.segmentation.eval_seg import eval_seg_when_train
from model_eval.EyeQ.eval_FIQA import eval_FIQA
from model_eval.evaluate import model_eval, fiq_evaluation
from util.find_gpu import find_gpu


def model_test(testOpt, testDataset, model, web_dir, eval_test=False, only_fake_TB=False):

    if testOpt.test_fiqa:
        test_good, test_usable, test_reject = eval_FIQA(testOpt, os.path.join(web_dir, 'images'), model.fiqa_model, save_log=False)
        return [test_good, 2*test_good + test_usable]

    if not testOpt.no_ssim_psnr:
        if testOpt.dataset_mode == 'fiq_basic':
            ssim, psnr = fiq_evaluation(testOpt, web_dir)
        else:
            ssim, psnr = model_eval(testOpt, web_dir, write_res=False)
        return [ssim, psnr]

def set_test_opt(opt):
    # hard-code some parameters for test_total
    opt.test_seg = True
    opt.num_threads = 4  # test_total code only supports num_threads = 1
    # batch_size 在尺度大的，可以设置比较小
    opt.batch_size = 1  # test_total code only supports batch_size = 1
    opt.serial_batches = True  # disable images shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test_total code saves the results to a HTML file.
    opt.test_fiqa = True

def get_out_put_web_dir_name(opt):
    if opt.phase == None:
        web_dir_name = os.path.join(opt.results_dir, opt.name,
                                          '{:s}_latest'.format(opt.phase))
    else:
        if opt.load_iter != 0:
            web_dir_name = os.path.join(opt.results_dir, opt.name,
                                        '{:s}_latest_iter{}'.format(opt.phase, opt.load_iter))
        else:
            web_dir_name = os.path.join(opt.results_dir, opt.name,
                                        '{:s}_latest'.format(opt.phase))
    return web_dir_name

def set_eyeq_opt(opt):
    eyeqOpt = copy.deepcopy(opt)
    set_test_opt(eyeqOpt)
    eyeqOpt.isTrain = False
    eyeqOpt.dataroot = '/images/liuhaofeng/Project/pixDRP/datasets/eyeq'
    eyeqOpt.need_mask = True
    eyeqOpt.dataset_mode = 'fiq_inference'
    eyeqOpt.num_test = 100000
    eyeqOpt.phase = 'test_eyeq'
    eyeqOpt.no_ssim_psnr = True
    eyeqOpt.only_fake_TB = True
    eyeqOpt.test_name = 'test_fiqa'
    eyeqOpt.test_fiqa = True
    eyeqOpt.load_size = eyeqOpt.crop_size = eyeqOpt.test_crop_size
    eyeqTestDataset = create_dataset(eyeqOpt)
    return eyeqOpt, eyeqTestDataset, get_out_put_web_dir_name(eyeqOpt)

def set_fiq_opt(opt):
    fiqTestOpt = copy.deepcopy(opt)
    set_test_opt(fiqTestOpt)
    fiqTestOpt.isTrain = False
    # fiqTestOpt.dataroot = '/images/liuhaofeng/Project/pixDRP/datasets/fiq_1229'
    fiqTestOpt.dataroot = './datasets/fiq_1229'
    fiqTestOpt.seg_gt_dir = '/images/liuhaofeng/Project/pixDRP/datasets/fiq_seg/target_gt'
    fiqTestOpt.dataset_mode = 'fiq_basic'
    fiqTestOpt.only_fake_TB = True
    fiqTestOpt.need_mask = True
    fiqTestOpt.num_test = 100000
    fiqTestOpt.phase = 'test_fiq'
    fiqTestOpt.target_gt_dir = 'target_gt_sat'
    fiqTestOpt.seg_gt_mask_dir = '/images/liuhaofeng/Project/pixDRP/datasets/fiq_seg/target_gt_mask'
    fiqTestOpt.seg_mask_dir = '/images/liuhaofeng/Project/pixDRP/datasets/fiq_seg/target_mask'
    fiqTestOpt.load_size = fiqTestOpt.crop_size = fiqTestOpt.test_crop_size
    fiqTestOpt.test_name = get_out_put_web_dir_name(fiqTestOpt)
    # fiqTestOpt.test_name = 'test_fiq'
    fiqTestOpt.test_fiqa = False
    fiqTestOpt.no_ssim_psnr = False
    fiqTestDataset = create_dataset(fiqTestOpt)  # create a dataset given opt.dataset_mode and other options
    return fiqTestOpt, fiqTestDataset, get_out_put_web_dir_name(fiqTestOpt)

def set_drive_opt(opt):
    driveTestOpt = copy.deepcopy(opt)
    set_test_opt(driveTestOpt)
    driveTestOpt.isTrain = False
    driveTestOpt.dataroot = '/images/liuhaofeng/Project/pixDRP/datasets/drive_seg'
    driveTestOpt.dataset_mode = 'fiq_basic'
    driveTestOpt.only_fake_TB = True
    driveTestOpt.need_mask = True
    driveTestOpt.num_test = 100000
    driveTestOpt.phase = 'test_drive'
    driveTestOpt.target_gt_dir = 'drive_test_gt'
    driveTestOpt.seg_gt_dir = '/images/liuhaofeng/Project/pixDRP/datasets/drive_seg/drive_seg_gt'
    driveTestOpt.seg_gt_mask_dir = '/images/liuhaofeng/Project/pixDRP/datasets/drive_seg/drive_seg_gt_mask'
    driveTestOpt.seg_mask_dir = '/images/liuhaofeng/Project/pixDRP/datasets/drive_seg/drive_seg_mask'
    driveTestOpt.load_size = driveTestOpt.crop_size = driveTestOpt.test_crop_size
    driveTestOpt.test_name = get_out_put_web_dir_name(driveTestOpt)
    driveTestOpt.test_fiqa = False
    driveTestOpt.no_ssim_psnr = False
    driveTestDataset = create_dataset(driveTestOpt)  # create a dataset given opt.dataset_mode and other options
    return driveTestOpt, driveTestDataset, get_out_put_web_dir_name(driveTestOpt)

def set_cataract_opt(opt):
    cataractTestOpt = copy.deepcopy(opt)
    set_test_opt(cataractTestOpt)
    cataractTestOpt.isTrain = False
    cataractTestOpt.dataroot = './datasets/drive_DG_temp_0911'
    cataractTestOpt.dataset_mode = 'cataract_guide_padding'
    cataractTestOpt.only_fake_TB = True
    cataractTestOpt.need_mask = True
    cataractTestOpt.num_test = 126 / opt.batch_size + 1
    cataractTestOpt.phase = 'test_cataract'
    cataractTestOpt.target_gt_dir = 'target_gt'
    # cataractTestOpt.seg_gt_dir = '/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_seg_s3'
    cataractTestOpt.seg_gt_dir = '/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_seg_s3'
    cataractTestOpt.seg_gt_mask_dir = '/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_gt_mask'
    cataractTestOpt.seg_mask_dir = '/images/liuhaofeng/Project/pixDA_GM/datasets/cataract_0830/target_mask'
    cataractTestOpt.load_size = cataractTestOpt.crop_size = cataractTestOpt.test_crop_size
    cataractTestOpt.test_name = get_out_put_web_dir_name(cataractTestOpt)
    cataractTestOpt.test_fiqa = False
    cataractTestOpt.no_ssim_psnr = False
    cataractTestDataset = create_dataset(cataractTestOpt)  # create a dataset given opt.dataset_mode and other options
    return cataractTestOpt, cataractTestDataset, get_out_put_web_dir_name(cataractTestOpt)

def set_public_cataract_opt(opt):
    publicCataractTestOpt = copy.deepcopy(opt)
    set_test_opt(publicCataractTestOpt)
    publicCataractTestOpt.isTrain = False
    publicCataractTestOpt.dataroot = 'datasets/TMI_public_circle_crop_1129'
    publicCataractTestOpt.dataset_mode = 'cataract_guide_padding'
    publicCataractTestOpt.only_fake_TB = True
    publicCataractTestOpt.need_mask = True
    publicCataractTestOpt.num_test = 100 / opt.batch_size + 1
    publicCataractTestOpt.phase = 'test_public_cataract'
    publicCataractTestOpt.target_gt_dir = 'target'
    publicCataractTestOpt.load_size = publicCataractTestOpt.crop_size = publicCataractTestOpt.test_crop_size
    publicCataractTestOpt.test_name = get_out_put_web_dir_name(publicCataractTestOpt)
    publicCataractTestOpt.test_fiqa = True
    publicCataractTestOpt.no_ssim_psnr = True
    publicCataractTestDataset = create_dataset(publicCataractTestOpt)  # create a dataset given opt.dataset_mode and other options
    return publicCataractTestOpt, publicCataractTestDataset, get_out_put_web_dir_name(publicCataractTestOpt)

def set_isee_opt(opt):
    iseeTestOpt = copy.deepcopy(opt)
    set_test_opt(iseeTestOpt)
    iseeTestOpt.isTrain = False
    iseeTestOpt.dataroot = 'datasets/isee_DR_1124'
    iseeTestOpt.dataset_mode = 'cataract_guide_padding'
    iseeTestOpt.only_fake_TB = True
    iseeTestOpt.need_mask = True
    iseeTestOpt.num_test = 784 / opt.batch_size + 1
    iseeTestOpt.phase = 'test_isee'
    iseeTestOpt.target_gt_dir = 'target'
    iseeTestOpt.load_size = iseeTestOpt.crop_size = iseeTestOpt.test_crop_size
    iseeTestOpt.test_name = get_out_put_web_dir_name(iseeTestOpt)
    iseeTestOpt.test_fiqa = True
    iseeTestOpt.no_ssim_psnr = True
    publicCataractTestDataset = create_dataset(iseeTestOpt)  # create a dataset given opt.dataset_mode and other options
    return iseeTestOpt, publicCataractTestDataset, get_out_put_web_dir_name(iseeTestOpt)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    set_test_opt(opt)
    if opt.find_gpu:
        opt.gpu_ids = [find_gpu(opt.gpu_ids[0], need_memory=opt.need_memory)]
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.test_rcf:
        cataractTestOpt, cataractTestDataset, cataract_dir_name = set_cataract_opt(opt)
        model_test(cataractTestOpt, cataractTestDataset, model, cataract_dir_name, eval_test=cataractTestOpt.eval_test,
                   only_fake_TB=cataractTestOpt.only_fake_TB)
        metrics = eval_seg_when_train(cataractTestOpt, cataract_dir_name, model.seg_model, is_fake_TB=True)
        print('cataract result: ', metrics)
    if opt.test_public_cataract_fiqa:
        publicCataractTestOpt, publicCataractTestDataset, public_cataract_dir_name = set_public_cataract_opt(opt)
        model_test(publicCataractTestOpt, publicCataractTestDataset, model, public_cataract_dir_name,
                   eval_test=publicCataractTestOpt.eval_test, only_fake_TB=publicCataractTestOpt.only_fake_TB)
        # print('public cataract result: ', metrics)
    if opt.test_fiq:
        fiqTestOpt, fiqTestDataset, fiq_dir_name = set_fiq_opt(opt)
        model_test(fiqTestOpt, fiqTestDataset, model, fiq_dir_name, eval_test=fiqTestOpt.eval_test,
                   only_fake_TB=fiqTestOpt.only_fake_TB)
        metrics = eval_seg_when_train(fiqTestOpt, fiq_dir_name, model.seg_model, is_fake_TB=True)
        print('fiq result: ', metrics)

    # if opt.test_drive:
    #     driveTestOpt, driveTestDataset, drive_dir_name = set_drive_opt(opt)
    #     model_test(driveTestOpt, driveTestDataset, model, drive_dir_name, eval_test=driveTestOpt.eval_test,
    #                only_fake_TB=driveTestOpt.only_fake_TB)
    #     metrics = eval_seg_when_train(driveTestOpt, drive_dir_name, model.seg_model, is_fake_TB=True)
    #     print('drive result: ', metrics)

    if opt.test_public_fiqa:
        eyeqOpt, eyeqTestDataset, eyeq_dir_name = set_eyeq_opt(opt)
        model_test(eyeqOpt, eyeqTestDataset, model, eyeq_dir_name, eval_test=eyeqOpt.eval_test,
                   only_fake_TB=eyeqOpt.only_fake_TB)
    if opt.test_isee:
        iseeOpt, iseeTestDataset, isee_dir_name = set_isee_opt(opt)
        model_test(iseeOpt, iseeTestDataset, model, isee_dir_name, eval_test=iseeOpt.eval_test,
                   only_fake_TB=iseeOpt.only_fake_TB)


