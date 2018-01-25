import mxnet as mx


def Conv(feat, name, num_filter, kernel, stride, pad, num_group=1):
    conv = mx.symbol.Convolution(name=name, data=feat, num_filter=num_filter, pad=pad,
                                 kernel=kernel,
                                 stride=stride,  num_group=num_group, no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, use_global_stats=True, fix_gamma=False,
                                  eps=0.00001)
    conv_relu = mx.symbol.Activation(name=name + '_relu', data=conv_bn, act_type='relu')
    return conv_relu


def InceptionA(feat, name, num_1x1, num_3x3, num_3x3_out, num_3x3_2, num_3x3_2_out, num_proj):
    a_1x1 = Conv(feat=feat, name=name + '_1x1', num_filter=num_1x1, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    a_3x3_reduce = Conv(feat=feat, name=name + '_3x3_reduce', num_filter=num_3x3, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    a_3x3 = Conv(feat=a_3x3_reduce, name=name + '_3x3', num_filter=num_3x3_out, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    a_3x3_2_reduce = Conv(feat=data, name=name + '_3x3_2_reduce', num_filter=num_3x3_2, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    a_3x3_2 = Conv(feat=a_3x3_2_reduce, name=name + '_3x3_2', num_filter=num_3x3_2_out, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    a_3x3_3 = Conv(feat=a_3x3_2, name=name + '_3x3_3', num_filter=num_3x3_2_out, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    if name == 'inception_5b':
        a_pool = mx.symbol.Pooling(name=name + '_max_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   pool_type='max')
    else:
        a_pool = mx.symbol.Pooling(name=name + '_avg_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   pool_type='avg')
    a_pool_proj = Conv(a_pool, name + '_pool_proj', num_proj, (1, 1), (1, 1), (0, 0))

    a_concat = mx.sym.Concat(*[a_1x1, a_3x3, a_3x3_3, a_pool_proj], name=('%s_concat' % name))

    return a_concat


def ReductionA(feat, name, num_3x3_reduce, num_3x3, num_3x3_2_reduce, num_3x3_2):
    a_3x3_reduce = Conv(feat=feat, name=name + '_3x3_reduce', num_filter=num_3x3_reduce, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    a_3x3 = Conv(feat=a_3x3_reduce, name=name + '_3x3', num_filter=num_3x3, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    a_3x3_2_reduce = Conv(feat=feat, name=name + '_3x3_2_reduce', num_filter=num_3x3_2_reduce, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    a_3x3_2 = Conv(feat=a_3x3_2_reduce, name=name + '_3x3_2', num_filter=num_3x3_2, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    a_3x3_3 = Conv(feat=a_3x3_2, name=name + '_3x3_3', num_filter=num_3x3_2, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    a_pool = mx.symbol.Pooling(name=name + '_pool', data=feat, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                               pool_type='max',
                               pooling_convention='full', cudnn_off=False, global_pool=False)

    a_concat = mx.sym.Concat(*[a_3x3, a_3x3_3, a_pool], name=('%s_concat' % name))

    return a_concat


def SE_Block(feat, name, num_down, num_up):
    global_pool = mx.symbol.Pooling(name=name + '_global_pool', data=feat, kernel=(7, 7), pool_type = 'avg',
                                    pooling_convention='full', global_pool=True)
    down = mx.symbol.Convolution(name=name + '_1x1_down', data=global_pool, num_filter= num_down,
                                 kernel=(1, 1),
                                 stride=(1, 1), no_bias=False)
    down_relu = mx.symbol.Activation(name=name + '_1x1_down_relu', data=down, act_type='relu')
    up = mx.symbol.Convolution(name=name + '_1x1_up', data=down_relu, num_filter=num_up, pad=(0, 0),
                               kernel=(1, 1),
                               stride=(1, 1), no_bias=False)
    up_prob = mx.symbol.Activation(name=name + '_1x1_up_prob', data=up, act_type='sigmoid')
    prob_reshape = mx.symbol.reshape(data=up_prob, shape=(-1, num_up, 1, 1))
    scale = mx.symbol.broadcast_mul(feat, prob_reshape)

    return scale


def get_symbol(num_classes):
    data = mx.symbol.Variable(name='data')

    conv1 = Conv(feat=data, name='conv1', num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
    ########## pool1 #########
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1, kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='full', cudnn_off=False, global_pool=False)
    conv2_reduce = Conv(pool1, 'conv2_reduce', 64, (1, 1), (1, 1), (0, 0))
    conv2 = Conv(conv2_reduce, 'conv2', 192, (3, 3), (1, 1), (1, 1))
    ########## pool2 #########
    pool2 = mx.symbol.Pooling(name='pool2', data=conv2, kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='full', cudnn_off=False, global_pool=False)
    ########## inception #########
    inception_3a = InceptionA(feat=pool2, name='inception_3a', num_1x1=64, num_3x3=64, num_3x3_out=64, num_3x3_2=64,
                              num_3x3_2_out=96, num_proj=32)
    inception_3a_se = SE_Block(feat=inception_3a, name='inception_3a', num_down=16, num_up=256)
    inception_3b = InceptionA(inception_3a_se, 'inception_3b', num_1x1=64, num_3x3=64, num_3x3_out=96, num_3x3_2=64,
                              num_3x3_2_out=96, num_proj=64)
    inception_3b_se = SE_Block(feat=inception_3b, name='inception_3b', num_down=20, num_up=320)
    inception_3c = ReductionA(feat=inception_3b_se, name='inception_3c', num_3x3_reduce=128, num_3x3=160,
                              num_3x3_2_reduce=64, num_3x3_2=96)
    inception_3c_se = SE_Block(feat=inception_3c, name='inception_3c', num_down=36, num_up=576)

    inception_4a = InceptionA(feat=inception_3c_se, name='inception_4a', num_1x1=224, num_3x3=64, num_3x3_out=96, num_3x3_2=96,
                              num_3x3_2_out=128, num_proj=128)
    inception_4a_se = SE_Block(feat=inception_4a, name='inception_4a', num_down=36, num_up=576)
    inception_4b = InceptionA(feat=inception_4a_se, name='inception_4b', num_1x1=192, num_3x3=96, num_3x3_out=128, num_3x3_2=96,
                              num_3x3_2_out=128, num_proj=128)
    inception_4b_se = SE_Block(feat=inception_4b, name='inception_4b', num_down=36, num_up=576)
    inception_4c = InceptionA(feat=inception_4b_se, name='inception_4c', num_1x1=160, num_3x3=128, num_3x3_out=160, num_3x3_2=128,
                              num_3x3_2_out=160, num_proj=128)
    inception_4c_se = SE_Block(feat=inception_4c, name='inception_4c', num_down=38, num_up=608)
    inception_4d = InceptionA(feat=inception_4c_se, name='inception_4d', num_1x1=96, num_3x3=128, num_3x3_out=192, num_3x3_2=160,
                              num_3x3_2_out=192, num_proj=128)
    inception_4d_se = SE_Block(feat=inception_4d, name='inception_4d', num_down=38, num_up=608)
    inception_4e = ReductionA(feat=inception_4d_se, name='inception_4e', num_3x3_reduce=128, num_3x3=192,
                              num_3x3_2_reduce=192, num_3x3_2=256)
    inception_4e_se = SE_Block(feat=inception_4e, name='inception_4e', num_down=66, num_up=1056)

    inception_5a = InceptionA(feat=inception_4e_se, name='inception_5a', num_1x1=352, num_3x3=192, num_3x3_out=320, num_3x3_2=160,
                              num_3x3_2_out=224, num_proj=128)
    inception_5a_se = SE_Block(feat=inception_5a, name='inception_5a', num_down=64, num_up=1024)
    inception_5b = InceptionA(feat=inception_5a_se, name='inception_5b', num_1x1=352, num_3x3=192, num_3x3_out=320, num_3x3_2=192,
                              num_3x3_2_out=224, num_proj=128)
    inception_5b_se = SE_Block(feat=inception_5b, name='inception_5b', num_down=64, num_up=1024)

    ######## pool5 ######
    avg_pool = mx.symbol.Pooling(name='avg_pool', data=inception_5b_se, kernel=(7, 7), stride=(1, 1), pool_type='avg',
                                 global_pool=True,
                                 pooling_convention='valid', cudnn_off=False)
    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)

    return softmax

softmax = get_symbol(num_classes=1000)
softmax.save('se-inception-v2-symbol.json')
