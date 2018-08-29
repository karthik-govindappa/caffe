import argparse
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
    parser.add_argument('--prune_percent', type=float, required=True,
        help='Percentage of params to be pruned [0.0, 1.0)')
    return parser.parse_args()

def save_sorted_indices(num_params, prune_percent):
    with open('ssd_vgg16.params', 'r') as f:
        params = []
        sys.stdout.write('Loading params from file ... ')
        sys.stdout.flush()
        for param in f:
            param = param.strip('\n')
            params.append(abs(float(param)))
        print 'done!'

    sys.stdout.write('Sorting params ... ')
    sys.stdout.flush()
    sorted_ids = np.argsort(params)
    del params
    print 'done!'
        
    prune_count = int(num_params * prune_percent)
    prune_ids = sorted_ids[:prune_count]
    del sorted_ids
    
    sys.stdout.write('Loading param ids from file ... ')
    sys.stdout.flush()
    with open('ssd_vgg16_params.ids', 'r') as f:
        params_ids = []
        for idx in f:
            idx = idx.strip('\n')
            params_ids.append(idx)
        print 'done!'
        
    sys.stdout.write('Saving sorted param ids to file ... ')
    sys.stdout.flush()
    with open('sdd_vgg16_sorted_params.ids', 'w') as f:
        for idx in prune_ids:
            f.write('{}\n'.format(params_ids[idx]))
        del params_ids
        print 'done!'

def load_net(proto, model):
    net = caffe.Net(proto, caffe.TEST)
    net.copy_from(model)
    return net

def save_indexed_params(proto, model):
    net = load_net(proto, model)
    fparams = open('ssd_vgg16.params', 'w')
    fparams_ids = open('ssd_vgg16_params.ids', 'w')
    count = 0
    for i, (name, params) in enumerate(net.params.iteritems()):
        sys.stdout.flush()
        sys.stdout.write('\rSaving indexed params of layer {} [{}/{}] ... '.format(name, i+1, len(net.params))) 
        for p in params:
            if len(p.data.shape) == 4:
                for n in xrange(p.data.shape[0]):
                    for c in xrange(p.data.shape[1]):
                        for h in xrange(p.data.shape[2]):
                            for w in xrange(p.data.shape[3]):
                                key = '{}-{}-{}-{}-{}'.format(name, n, c, h, w)
                                fparams.write('{}\n'.format(p.data[n,c,h,w]))
                                fparams_ids.write('{}\n'.format(key))
                                count += 1
            elif len(p.data.shape) == 1:
                for n in xrange(p.data.shape[0]):
                    key = '{}-{}'.format(name, n)
                    fparams.write('{}\n'.format(p.data[n]))
                    fparams_ids.write('{}\n'.format(key))
                    count += 1
    print 'done!'
    return count

def main(args):
    # save indexed params and their indices
    num_params = save_indexed_params(args.proto, args.model)
    print 'Total number of parameters: {} Million'.format(float(num_params)/1e6)
    
    # saved subset of sorted indices
    save_sorted_indices(num_params, args.prune_percent)
    
if __name__=='__main__':
    main(parse_args())
