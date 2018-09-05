import argparse
from collections import OrderedDict
import os
import sys

import caffe
import numpy as np

caffe.set_mode_gpu()
      
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True,
        help='Deploy prototxt of SSD-VGG16')
    parser.add_argument('--model', type=str, required=True,
        help='Deploy caffemodel of SSD-VGG16')
    return parser.parse_args()

def load_net(proto, model):
    net = caffe.Net(proto, caffe.TEST)
    net.copy_from(model)
    return net

def model_stats(proto, model):
    net = load_net(proto, model)
    stats = OrderedDict()
    max_name_len = 0
    max_shape_len = 0
    for i, (name, params) in enumerate(net.params.iteritems()):
        sys.stdout.flush()
        sys.stdout.write('\rProcessing layer {}/{} ... '.format(i+1, len(net.params))) 
        for p in params:
            key = '{}({})'.format(name, p.data.shape)
            nzeros = np.count_nonzero(p.data.flatten())
            zeros = p.data.size - nzeros
            assert p.data.size == (zeros + nzeros)
            stats[key] = {'zeros': zeros, 'nzeros': nzeros}
            if len(name) > max_name_len:
                max_name_len = len(name)
            if len(str(p.data.shape)) > max_shape_len:
                max_shape_len = len(str(p.data.shape))
    print 'done!'
    return stats, max_name_len, max_shape_len

def main(args):
    # get stats of each layer in the model
    stats, name_len, shape_len = model_stats(args.proto, args.model)

    # print stats
    total_zs = 0
    total_nzs = 0
    for i, (k, v) in enumerate(stats.iteritems()):
        total_zs += v['zeros']
        total_nzs += v['nzeros']
        size = v['zeros'] + v['nzeros']
        name, shape = k.split('((')
        name = name.strip()
        shape = shape.replace(')', '').strip()
        name = '{}{}'.format(name, ' '*(name_len - len(name)))
        shape = '({}){}'.format(shape, ' '*(shape_len - len(shape)))
        print '{:3d}. {}: shape={}\tparams={:.2f}k\tzeros={:.2f}k\t({:.2f}%)\tnon_zeros={:.2f}k\t({:.2f}%)'.format(
            i+1, name, shape, size/1000., v['zeros']/1000., (v['zeros']*100.)/size, v['nzeros']/1000.,
            (v['nzeros']*100.)/size)
    total_params = total_zs + total_nzs
    print '\n-----------------------'
    print 'Total parameters = {:.2f} Million'.format(total_params/1e6)
    print 'Total non zero parameters = {:.2f} Million ({:.2f}%)'.format(total_nzs/1e6, (total_nzs*100.)/total_params)
    print 'Total zero parameters = {:.2f} Million ({:.2f}%)\n'.format(total_zs/1e6, (total_zs*100.)/total_params)

 
if __name__=='__main__':
    main(parse_args())
