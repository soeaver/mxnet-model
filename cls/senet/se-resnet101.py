import mxnet as mx


def Conv(last_layer, name, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0)):
    conv = mx.symbol.Convolution(name=name, data=last_layer, num_filter=num_filter, pad=pad,
                                 kernel=kernel,
                                 stride=stride, no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, use_global_stats=True,
                                  fix_gamma=False,
                                  eps=0.00001)
    conv_scale = conv_bn
    conv_relu = mx.symbol.Activation(name=name + '_relu', data=conv_scale, act_type='relu')
    return conv_relu


def senet_block(data, name, num_filter, stride, match):
    reduce_relu = Conv(data, name + '_1x1_reduce', num_filter * 2, (1, 1), stride, (0, 0))
    conv_3x3 = mx.symbol.Convolution(name=name + '_3x3', data=reduce_relu, num_filter=num_filter * 2, num_group = 32, pad=(1, 1),
                                 kernel=(3, 3),
                                 stride=(1, 1), no_bias=True)
    conv_3x3_bn = mx.symbol.BatchNorm(name=name + '_3x3_bn', data=conv_3x3, use_global_stats=True,
                                  fix_gamma=False,
                                  eps=0.00001)
    conv_3x3_relu = mx.symbol.Activation(name=name + '_3x3_relu', data=conv_3x3_bn, act_type='relu')
    increase = mx.symbol.Convolution(name=name + '_1x1_increase', data=conv_3x3_relu, num_filter=num_filter * 4,
                                     pad=(0, 0),
                                     kernel=(1, 1),
                                     stride=(1, 1), no_bias=True)
    increase_bn = mx.symbol.BatchNorm(name=name + '_1x1_increase_bn', data=increase, use_global_stats=True,
                                      fix_gamma=False, eps=0.00001)
    global_pool = mx.symbol.Pooling(name=name + '_global_pool', data=increase_bn, kernel=(7, 7), pool_type = 'avg',
                                    pooling_convention='full', global_pool=True)
    down = mx.symbol.Convolution(name=name + '_1x1_down', data=global_pool, num_filter= num_filter / 4,
                                 kernel=(1, 1),
                                 stride=(1, 1), no_bias=False)
    down_relu = mx.symbol.Activation(name=name + '_1x1_down_relu', data=down, act_type='relu')
    up = mx.symbol.Convolution(name=name + '_1x1_up', data=down_relu, num_filter=num_filter * 4, pad=(0, 0),
                               kernel=(1, 1),
                               stride=(1, 1), no_bias=False)
    up_prob = mx.symbol.Activation(name=name + '_1x1_up_prob', data=up, act_type='sigmoid')
    increase_ele = mx.symbol.broadcast_mul(increase_bn,
                                           mx.symbol.reshape(data=up_prob, shape=(-1, num_filter * 4, 1, 1)))

    if match:
        match_conv = mx.symbol.Convolution(name=name + '_1x1_proj', data=data, num_filter=num_filter * 4,
                                           pad=(0, 0),
                                           kernel=(1, 1),
                                           stride=stride, no_bias=True)
        match_conv_bn = mx.symbol.BatchNorm(name=name + '_1x1_proj_bn', data=match_conv, use_global_stats=True,
                                            fix_gamma=False,
                                            eps=0.00001)
        eletwise = mx.symbol.broadcast_add(name=name, lhs=increase_ele, rhs=match_conv_bn)
    else:
        eletwise = mx.symbol.broadcast_add(name=name, lhs=data, rhs=increase_ele)
    eletwise_relu = mx.symbol.Activation(name=name + '_relu', data=eletwise, act_type='relu')
    return eletwise_relu


def stage(num, num_block, data, num_filter, stride, name):
    for i in xrange(num_block):
        if i == 0:
            data = senet_block(data, name + str(num + i), num_filter, (stride, stride), True)
        else:
            data = senet_block(data, name + str(num + i), num_filter, (1, 1), False)

    return data


def get_symbol(num_classes=1000):
    data = mx.symbol.Variable(name='data')

    conv1 = Conv(data, 'conv1_7x7', 64, (7, 7), (2, 2), (3, 3))
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1, kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='full', cudnn_off=False, global_pool=False)
    stage1 = stage(1, 3, pool1, 64, 1, 'seres')
    stage2 = stage(4, 4, stage1, 128, 2, 'seres')
    stage3 = stage(8, 23, stage2, 256, 2, 'seres')
    stage4 = stage(31, 3, stage3, 512, 2, 'seres')

    avg_pool = mx.symbol.Pooling(name='avg_pool', data=stage4, kernel=(7, 7), stride = (1, 1), pool_type='avg',
                                 global_pool=True,
                                 pooling_convention='valid', cudnn_off=False)
    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)

    return softmax


softmax = get_symbol(1000)
softmax.save('se-resnext101-32x4d-symbol.json')


