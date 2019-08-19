import os
import argparse

from functools import partial


class Config(argparse.Namespace):
    @staticmethod
    def get_parser(name):
        """ make default formatted parser """
        parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # print default value always
        parser.add_argument = partial(parser.add_argument, help=' ')
        return parser

    def build_parser(self):
        parser = self.get_parser("Config for neural style transfer")
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--size', type=int, default=512, help='image height/width')
        parser.add_argument('--n_iter', type=int, default=300, help='number of training iterations')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency of training status')
        parser.add_argument('--content_weight', type=float, default=1., help='weights for content loss')
        parser.add_argument('--style_weight', type=float, default=1e6, help='weights for style loss')
        parser.add_argument('--path', default='image/', help='path where images store')
        parser.add_argument('--content_name', default='dancing.jpg', help='filename of content image')
        parser.add_argument('--style_name', default='picasso.jpg', help='filename of style image')
        parser.add_argument('--generated_name', default='generated.jpg', help='filename of generated image')
        parser.add_argument('--content_layers', default='4', help='desired conv layers to compute content losses')
        parser.add_argument('--style_layers', default='1,2,3,4,5', help='desired conv layers to compute style losses')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args, unparsed = parser.parse_known_args()
        super().__init__(**vars(args))

        self.content_path = os.path.join(self.path, self.content_name)
        self.style_path = os.path.join(self.path, self.style_name)
        self.generated_path = os.path.join(self.path, self.generated_name)
        self.content_layers = [int(s) for s in self.content_layers.split(',')]
        self.style_layers = [int(s) for s in self.style_layers.split(',')]

