from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test_total options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # parser.add_argument('--phase', type=str, default='test_total', help='train, val, test_total, etc')
        # Dropout and Batchnorm has different behavioir during training and test_total.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test_total time.')
        parser.add_argument('--is_fiq', action='store_true', help='is fiq dataset.')
        # rewrite devalue values
        parser.set_defaults(model='test_total')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        # 修改这两个属性，测试时不再crop
        # parser.set_defaults(load_source_size=parser.get_default('crop_size'))
        # parser.set_defaults(load_target_size=parser.get_default('crop_size'))
        self.isTrain = False
        # parser.add_argument('--target_gt_dir', type=str, default='high_quality', help='saves ground truth here')
        parser.add_argument('--postname', type=str, default='breast', help='saves ground truth here')

        return parser
