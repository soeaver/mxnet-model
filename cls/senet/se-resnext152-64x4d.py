import mxnet as mx


def Conv(feat, name, num_filter, kernel, stride, pad, num_group=1):
    conv = mx.symbol.Convolution(name=name, data=feat, num_filter=num_filter, pad=pad,
                                 kernel=kernel,
                                 stride=stride,  num_group=num_group, no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, use_global_stats=True,
                                  fix_gamma=False, eps=0.0001)
    conv_relu = mx.symbol.Activation(name=name + '_relu', data=conv_bn, act_type='relu')
    return conv_relu


def senet_block(feat, name, num_filter, num_output, stride, num_group, match=False):
    reduce = Conv(feat=feat, name=name + '_conv1', num_filter=num_filter, kernel=(1, 1), stride=(1, 1),
                  pad=(0, 0))
    conv_3x3 = Conv(feat=reduce, name=name + '_conv2', num_filter=num_output, kernel=(3, 3), stride=stride,
                    pad=(1, 1), num_group=num_group)
    increase = mx.symbol.Convolution(name=name + '_conv3', data=conv_3x3, num_filter=num_output,
                                     pad=(0, 0),
                                     kernel=(1, 1),
                                     stride=(1, 1), no_bias=True)
    increase_bn = mx.symbol.BatchNorm(name=name + '_conv3_bn', data=increase, use_global_stats=True,
                                      fix_gamma=False, eps=0.00001)
    global_pool = mx.symbol.Pooling(name=name + '_global_pool', data=increase_bn, kernel=(7, 7), pool_type = 'avg',
                                    pooling_convention='full', global_pool=True)
    down = mx.symbol.Convolution(name=name + '_down', data=global_pool, num_filter= num_output / 16,
                                 pad=(0, 0), kernel=(1, 1),
                                 stride=(1, 1), no_bias=False)
    down_relu = mx.symbol.Activation(name=name + '_down_relu', data=down, act_type='relu')
    up = mx.symbol.Convolution(name=name + '_up', data=down_relu, num_filter=num_output, pad=(0, 0),
                               kernel=(1, 1),
                               stride=(1, 1), no_bias=False)
    up_prob = mx.symbol.Activation(name=name + '_up_sigmoid', data=up, act_type='sigmoid')
    increase_ele = mx.symbol.broadcast_mul(increase_bn,
                                           mx.symbol.reshape(data=up_prob, shape=(-1, num_output, 1, 1)))

    if match:
        if name == 'seresx1':
            match_conv = mx.symbol.Convolution(name=name + '_match_conv', data=feat, num_filter=num_output, pad=(0, 0),
                                               kernel=(1, 1), stride=stride, no_bias=True)
        else:
            match_conv = mx.symbol.Convolution(name=name + '_match_conv', data=feat, num_filter=num_output, pad=(1, 1),
                                               kernel=(3, 3), stride=stride, no_bias=True)
        match_conv_bn = mx.symbol.BatchNorm(name=name + '_match_conv_bn', data=match_conv, use_global_stats=True,
                                            fix_gamma=False,
                                            eps=0.00001)
        eletwise = mx.symbol.broadcast_add(name=name, lhs=increase_ele, rhs=match_conv_bn)
    else:
        eletwise = mx.symbol.broadcast_add(name=name, lhs=feat, rhs=increase_ele)
    eletwise_relu = mx.symbol.Activation(name=name + '_relu', data=eletwise, act_type='relu')
    return eletwise_relu


def stage(start_num, num_block, feat, num_filter, num_output, stride, num_group, name):
    for i in xrange(num_block):
        if i == 0:
            feat = senet_block(feat=feat, name=name + str(start_num + i), num_filter=num_filter, num_output=num_output,
                               stride=(stride, stride),
                               num_group=num_group, match=True)
        else:
            feat = senet_block(feat=feat, name=name + str(start_num + i), num_filter=num_filter, num_output=num_output,
                               stride=(1, 1),
                               num_group=num_group, match=False)

    return feat


def get_symbol(num_classes, num_group):
    num_output = [256, 512, 1024, 2048]
    data = mx.symbol.Variable(name='data')

    conv1 = Conv(feat=data, name='conv1', num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    conv2 = Conv(feat=conv1, name='conv2', num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    conv3 = Conv(feat=conv2, name='conv3', num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    ########## pool1 #########
    pool1 = mx.symbol.Pooling(name='pool1', data=conv3, kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='full', cudnn_off=False, global_pool=False)
    ########## seresx1-seresx50 #########
    stage1 = stage(start_num=1, num_block=3, feat=pool1, num_filter=num_group*2, num_output=num_output[0],
                   stride=1, num_group=num_group, name='seresx')
    stage2 = stage(start_num=4, num_block=8, feat=stage1, num_filter=num_group*4, num_output=num_output[1],
                   stride=2, num_group=num_group, name='seresx')
    stage3 = stage(start_num=12, num_block=36, feat=stage2, num_filter=num_group*8,
                   num_output=num_output[2], stride=2, num_group=num_group, name='seresx')
    stage4 = stage(start_num=48, num_block=3, feat=stage3, num_filter=num_group*16,
                   num_output=num_output[3], stride=2, num_group=num_group, name='seresx')

    ########## pool5 #########
    avg_pool = mx.symbol.Pooling(name='avg_pool', data=stage4, kernel=(7, 7), stride=(1, 1), pool_type='avg',
                                 global_pool=True,
                                 pooling_convention='valid', cudnn_off=False)
    dropout = mx.sym.Dropout(data=avg_pool, p=0.2)
    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=dropout)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)
    return softmax

softmax = get_symbol(num_classes=1000, num_group=64)
softmax.save('se-resnext152-64x4d-symbol.json')


