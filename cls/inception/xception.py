import mxnet as mx


# def separable_conv(data, n_in_ch, n_out_ch, kernel, pad, name, depth_mult=1):
#     #  depthwise convolution
#     channels = mx.symbol.SliceChannel(data, axis=1, num_outputs=n_in_ch)
#     dw_outs = [mx.symbol.Convolution(data=channels[i], num_filter=depth_mult,
#                                   pad=pad, kernel=kernel, no_bias=True,
#                                   name=name+'_depthwise_kernel'+str(i))
#                for i in range(n_in_ch)]
#     dw_out = mx.symbol.Concat(*dw_outs)
#     #  pointwise convolution
#     pw_out = mx.symbol.Convolution(dw_out, num_filter=n_out_ch, kernel=(1, 1),
#                                 no_bias=True, name=name+'_pointwise_kernel')
#     return pw_out


def inception_stem(data,
                   num1_1,
                   num2_1,
                   use_global_stats
                   ):
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(3, 3),
                                         stride=(2, 2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1_scale, act_type='relu')
    conv2 = mx.symbol.Convolution(name='conv2', data=conv1_relu, num_filter=num2_1, pad=(0, 0), kernel=(3, 3),
                                         stride=(1, 1), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name='conv2_bn', data=conv2, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv2_scale = conv2_bn
    conv2_relu = mx.symbol.Activation(name='conv2_relu', data=conv2_scale, act_type='relu')

    return conv2_relu


def XceptionA(data, data_relu,
             num1_1,
             num2_1, num2_2, num2_3, num2_4, num2_5, num2_6,
             name, use_global_stats):
    # match_conv
    matchconv = mx.symbol.Convolution(name=name + '_match_conv', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(2, 2), no_bias=True)
    matchconv_bn = mx.symbol.BatchNorm(name=name + '_match_conv_bn', data=matchconv, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    matchconv_scale = matchconv_bn

    # ---------conv1_1,conv1_2,conv2_1,conv2_2-------------
    # conv1_1 = separable_conv(data_relu, num2_1, num2_2, (3, 3), (1, 1), name+'_conv1_1')
    conv1_1 = mx. symbol.Convolution(name=name + '_conv1_1', data=data_relu, num_filter=num2_1, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=num2_2, no_bias=True)
    conv1_2 = mx.symbol.Convolution(name=name + '_conv1_2', data=conv1_1, num_filter=num2_3, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv1_2_bn = mx.symbol.BatchNorm(name=name + '_conv1_bn', data=conv1_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv1_2_scale = conv1_2_bn
    conv1_2_relu = mx.symbol.Activation(name=name + '_conv1_relu', data=conv1_2_scale, act_type='relu')

    # conv2_1 = separable_conv(conv1_2_relu, num2_4, num2_5, (3, 3), (1, 1), name+'_conv2_1')
    conv2_1 = mx. symbol.Convolution(name=name + '_conv2_1', data=conv1_2_relu, num_filter=num2_4, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=num2_5, no_bias=True)
    conv2_2 = mx.symbol.Convolution(name=name + '_conv2_2', data=conv2_1, num_filter=num2_6, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv2_2_bn = mx.symbol.BatchNorm(name=name + '_conv2_bn', data=conv2_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv2_2_scale = conv2_2_bn

    # pool
    pool = mx.symbol.Pooling(name=name + '_pool', data=conv2_2_scale, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                           pool_type='max', pooling_convention='valid')
    # elewise
    elewise = mx.symbol.broadcast_add(matchconv_scale, pool, name=name + '_elewise')

    return elewise


def XceptionB(data,
             num1_1, num1_2, num1_3,
             num2_1, num2_2, num2_3,
             num3_1, num3_2, num3_3,
             name, use_global_stats):
    # relu
    relu = mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')

    # ----------conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2--------------
    # conv1_1 = separable_conv(data, num1_1, num1_2, (3, 3), (1, 1), name+'_conv1_1')
    conv1_1 = mx. symbol.Convolution(name=name + '_conv1_1', data=relu, num_filter=num1_1, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=num1_2, no_bias=True)

    conv1_2 = mx.symbol.Convolution(name=name + '_conv1_2', data=conv1_1, num_filter=num1_3, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv1_2_bn = mx.symbol.BatchNorm(name=name + '_conv1_bn', data=conv1_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv1_2_scale = conv1_2_bn
    conv1_2_relu = mx.symbol.Activation(name=name + '_conv1_relu', data=conv1_2_scale, act_type='relu')

    # conv2_1 = separable_conv(conv1_2_relu, num2_1, num2_2, (3, 3), (1, 1), name+'_conv2_1')
    conv2_1 = mx. symbol.Convolution(name=name + '_conv2_1', data=conv1_2_relu, num_filter=num2_1, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=num2_2, no_bias=True)
    conv2_2 = mx.symbol.Convolution(name=name + '_conv2_2', data=conv2_1, num_filter=num2_3, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv2_2_bn = mx.symbol.BatchNorm(name=name + '_conv2_bn', data=conv2_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv2_2_scale = conv2_2_bn
    conv2_2_relu = mx.symbol.Activation(name=name + '_conv2_relu', data=conv2_2_scale, act_type='relu')

    # conv3_1 = separable_conv(conv2_2_relu, num3_1, num3_2, (3, 3), (1, 1), name+'_conv3_1')
    conv3_1 = mx. symbol.Convolution(name=name + '_conv3_1', data=conv2_2_relu, num_filter=num3_1, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=num3_2, no_bias=True)
    conv3_2 = mx.symbol.Convolution(name=name + '_conv3_2', data=conv3_1, num_filter=num3_3, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv3_2_bn = mx.symbol.BatchNorm(name=name + '_conv3_bn', data=conv3_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv3_2_scale = conv3_2_bn

    # elewise
    elewise = mx.symbol.broadcast_add(data, conv3_2_scale, name=name + '_elewise')

    return elewise

def get_symbol(num_classes):
    use_global_stats = True
    # if 'use_global_stats' not in kwargs:
    #     use_global_stats = False
    # else:
    #     use_global_stats = kwargs['use_global_stats']

    # stem
    data = mx.symbol.Variable(name='data')
    num1_1 = 32
    num2_1 = 64
    stem = inception_stem(data, num1_1, num2_1, use_global_stats)

    # 3*XceptionA
    num1_1 = 128
    num2_1, num2_2, num2_3, num2_4, num2_5, num2_6 = (64, 64, 128, 128, 128, 128)
    name = 'xception1'
    xception1 = XceptionA(stem, stem, num1_1, num2_1, num2_2, num2_3, num2_4, num2_5, num2_6, name, use_global_stats)

    xception2_relu = mx.symbol.Activation(name='xception2_relu', data=xception1, act_type='relu')
    num1_1 = 256
    num2_1, num2_2, num2_3, num2_4, num2_5, num2_6 = (128, 128, 256, 256, 256, 256)
    name = 'xception2'
    xception2 = XceptionA(xception1, xception2_relu, num1_1, num2_1, num2_2, num2_3, num2_4, num2_5, num2_6, name, use_global_stats)

    xception3_relu = mx.symbol.Activation(name='xception3_relu', data=xception2, act_type='relu')
    num1_1 = 728
    num2_1, num2_2, num2_3, num2_4, num2_5, num2_6 = (256, 256, 728, 728, 728, 728)
    name = 'xception3'
    xception3 = XceptionA(xception2, xception3_relu, num1_1, num2_1, num2_2, num2_3, num2_4, num2_5, num2_6, name, use_global_stats)

    # 8*XceptionB
    num1_1, num1_2, num1_3 = (728, 728, 728)
    num2_1, num2_2, num2_3 = (728, 728, 728)
    num3_1, num3_2, num3_3 = (728, 728, 728)
    name = 'xception4'
    xception4 = XceptionB(xception3, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name, use_global_stats)

    name = 'xception5'
    xception5 = XceptionB(xception4, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name, use_global_stats)

    name = 'xception6'
    xception6 = XceptionB(xception5, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    name = 'xception7'
    xception7 = XceptionB(xception6, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    name = 'xception8'
    xception8 = XceptionB(xception7, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    name = 'xception9'
    xception9 = XceptionB(xception8, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    name = 'xception10'
    xception10 = XceptionB(xception9, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    name = 'xception11'
    xception11 = XceptionB(xception10, num1_1, num1_2, num1_3, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, name,
                          use_global_stats)

    # 1* XceptionA
    xception12_relu = mx.symbol.Activation(name='xception12_relu', data=xception11, act_type='relu')
    num1_1 = 1024
    num2_1, num2_2, num2_3, num2_4, num2_5, num2_6 = (728, 728, 728, 728, 728, 1024)
    name = 'xception12'
    xception12 = XceptionA(xception11, xception12_relu, num1_1, num2_1, num2_2, num2_3, num2_4, num2_5, num2_6, name, use_global_stats)

    # classifier
    # conv3_1 = separable_conv(xception12, 1024, 1024, (3, 3), (1, 1), 'conv3_1')
    conv3_1 = mx. symbol.Convolution(name='conv3_1', data=xception12, num_filter=1024, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=1024, no_bias=True)

    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=conv3_1, num_filter=1536, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv3_bn = mx.symbol.BatchNorm(name='conv3_bn', data=conv3_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv3_scale = conv3_bn
    conv3_relu = mx.symbol.Activation(name='conv3_relu', data=conv3_scale, act_type='relu')

    # conv4_1 = separable_conv(conv3_relu, 1536, 1536, (3, 3), (1, 1), 'conv4_1')
    conv4_1 = mx. symbol.Convolution(name='conv4_1', data=conv3_relu, num_filter=1536, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), num_group=1536, no_bias=True)
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=conv4_1, num_filter=2048, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    conv4_bn = mx.symbol.BatchNorm(name='conv4_bn', data=conv4_2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    conv4_scale = conv4_bn
    conv4_relu = mx.symbol.Activation(name='conv4_relu', data=conv4_scale, act_type='relu')
    pool_ave = mx.symbol.Pooling(name='pool_ave', data=conv4_relu, kernel=(1, 1), pool_type='avg',
                           pooling_convention='full', global_pool=True)
    flatten_0 = mx.symbol.Flatten(data=pool_ave, name="flatten_0")
    classifier = mx.symbol.FullyConnected(data=flatten_0, num_hidden=num_classes, name='classifier', no_bias=False)

    prob = mx.symbol.SoftmaxOutput(data=classifier, name='prob')

    return prob


if __name__ == '__main__':
    net = get_symbol(1000)
    # shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
    net.save('xception-symbol.json')
