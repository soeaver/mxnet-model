import mxnet as mx


def Conv(data, name, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0)):
    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, pad=pad,
                                 kernel=kernel,
                                 stride=stride, no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, use_global_stats=True,
                                  fix_gamma=False,
                                  eps=0.0001)
    conv_scale = conv_bn
    conv_relu = mx.symbol.Activation(name=name + '_relu', data=conv_scale, act_type='relu')
    return conv_relu


def air_block(data, num_filter, stride, match, name):
    conv1_1_relu = Conv(data, name + '_conv1_1', num_filter, (1, 1), (1, 1), (0, 0))
    conv1_2 = mx.symbol.Convolution(name=name + '_conv1_2', data=conv1_1_relu, num_filter=num_filter, pad=(1, 1),
                                    kernel=(3, 3),
                                    stride=stride, no_bias=True)
    conv2_1_relu = Conv(data, name + '_conv2_1', num_filter / 2, (1, 1), (1, 1), (0, 0))
    conv2_2_relu = Conv(conv2_1_relu, name + '_conv2_2', num_filter / 2, (3, 3), stride, (1, 1))
    conv2_3 = mx.symbol.Convolution(name=name + '_conv2_3', data=conv2_2_relu, num_filter=num_filter / 2, pad=(1, 1),
                                    kernel=(3, 3),
                                    stride=(1, 1), no_bias=True)

    concat = mx.sym.Concat(conv1_2, conv2_3, dim = 1, name=('%s' % name + '_concat'))
    concat_bn = mx.symbol.BatchNorm(name=name + '_concat_bn', data=concat, use_global_stats=True,
                                    fix_gamma=False,
                                    eps=0.0001)
    concat_scale = concat_bn
    concat_relu = mx.symbol.Activation(name=name + '_concat_relu', data=concat_scale, act_type='relu')

    conv3 = mx.symbol.Convolution(name=name + '_conv3', data=concat_relu, num_filter=num_filter * 4, pad=(0, 0),
                                  kernel=(1, 1),
                                  stride=(1, 1), no_bias=True)
    conv3_bn = mx.symbol.BatchNorm(name=name + '_conv3_bn', data=conv3, use_global_stats=True,
                                   fix_gamma=False,
                                   eps=0.0001)
    conv3_scale = conv3_bn

    if match:
        match_conv = mx.symbol.Convolution(name=name + '_match_conv', data=data, num_filter=num_filter * 4, pad=(0, 0),
                                           kernel=(1, 1),
                                           stride=stride, no_bias=True)
        match_conv_bn = mx.symbol.BatchNorm(name=name + '_match_conv_bn', data=match_conv, use_global_stats=True,
                                            fix_gamma=False,
                                            eps=0.0001)
        match_conv_scale = match_conv_bn

        eletwise = mx.symbol.broadcast_add(name=name + '_eletwise', lhs=match_conv_scale, rhs=conv3_scale)

    else:
        eletwise = mx.symbol.broadcast_add(name=name + '_eletwise', lhs=data, rhs=conv3_scale)

    eletwise_relu = mx.symbol.Activation(name=name + '_relu', data=eletwise, act_type='relu')

    return eletwise_relu


def stage(num, num_block, data, num_filter, stride, name):
    for i in xrange(num_block):
        if i == 0:
            data = air_block(data, num_filter, stride, True, name + str(num + i))
        else:
            data = air_block(data, num_filter, (1, 1), False, name + str(num + i))

    return data


def get_symbol(num_classes=1000):
    data = mx.symbol.Variable(name='data')

    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7),
                                  stride=(2, 2),
                                  no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1, use_global_stats=True, fix_gamma=False, eps=0.0001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1_scale, act_type='relu')

    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='valid', cudnn_off=False, global_pool=False)

    stage1 = stage(1, 3, pool1, 64, (1, 1), 'air')

    stage2 = stage(4, 4, stage1, 128, (2, 2), 'air')

    stage3 = stage(8, 23, stage2, 256, (2, 2), 'air')

    stage4 = stage(31, 3, stage3, 512, (2, 2), 'air')

    avg_pool = mx.symbol.Pooling(name='avg_pool', data=stage4, kernel=(1, 1), pool_type='avg',
                                 global_pool=True,
                                 pooling_convention='full', cudnn_off=False)

    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)

    return [stage1, stage2, stage3, stage4, softmax]

stage1, stage2, stage3, stage4, softmax = get_symbol(1000)
softmax.save('air101-symbol.json')

