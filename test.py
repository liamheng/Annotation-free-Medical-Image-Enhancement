import ntpath
from util.util import tensor2im, save_image
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_batch_images
from util import html
from model_eval.segmentation.eval_seg import eval_seg_when_train
from model_eval.EyeQ.eval_FIQA import eval_FIQA
from model_eval.evaluate import model_eval, fiq_evaluation
# def inference(opt, model, dataset, output_dir):
#     model.eval()
#     for i, images in enumerate(dataset):
#         if i >= opt.num_test:  # only apply our model to opt.num_test images.
#             break
#         model.set_input(images, isTrain=False)  # unpack images from images loader
#         model.test_total()           # run inference
#         visuals = model.get_current_visuals()  # get images results
#         img_paths = model.get_image_paths()     # get images paths
#         if i % 5 == 0:  # save images to an HTML file
#             print('processing (%04d)-th images... %s' % (i, img_paths))
#         save_inference_images(output_dir, visuals, img_paths, aspect_ratio=opt.aspect_ratio)
def test_net(opt, model, dataset, only_fake_TB=False):
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.test_name != '':
        web_dir = os.path.join(opt.results_dir, opt.name, opt.test_name)
    elif opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval_test:
        model.eval()
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data, isTrain=False)  # unpack images from images loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get images results
        img_path = model.get_image_paths()     # get images paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th images... %s' % (i, img_path))
        save_batch_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, only_fake_TB=only_fake_TB)
    webpage.save()  # save the HTML
    # if opt.test_fiqa:
    #     test_good, test_usable, test_reject = eval_FIQA(opt, os.path.join(web_dir, 'images'), model.fiqa_model)

    if opt.test_fiqa:
        test_good, test_usable, test_reject = eval_FIQA(opt, os.path.join(web_dir, 'images'), model.fiqa_model,
                                                        save_log=False)
        return [test_good, 2 * test_good + test_usable]

    if not opt.no_ssim_psnr:
        if opt.dataset_mode == 'fiq_basic':
            ssim, psnr = fiq_evaluation(opt, web_dir)
        elif opt.dataset_mode == 'FIQ':
            ssim, psnr = fiq_evaluation(opt, web_dir)
        else:
            ssim, psnr = model_eval(opt, web_dir, write_res=False)
        return [ssim, psnr]

# def save_inference_images(output_dir, visuals, image_paths, aspect_ratio=1.0):
#     for i, image_path in enumerate(image_paths):
#         short_path = ntpath.basename(image_path)
#         name = os.path.splitext(short_path)[0]
#         for label, im_data in visuals.items():
#             # only_fake_B:
#             if label != 'fake_B':
#                 continue
#             im = tensor2im(im_data[i:i+1])
#             image_name = '%s.png' % (name)
#             save_path = os.path.join(output_dir, image_name)
#             save_image(im, save_path, aspect_ratio=aspect_ratio)


def set_test_opt(opt):
    # hard-code some parameters for test_total
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.need_mask = True
    opt.phase = 'test_total'


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    set_test_opt(opt)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # if not os.path.isdir(opt.inference_dir):
    #     os.mkdir(opt.inference_dir)
    test_net(opt, model, dataset)
