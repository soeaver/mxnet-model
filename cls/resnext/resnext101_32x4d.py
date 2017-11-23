import mxnet as mx


def match_block(layer_name, last_output, input, output, num_group):
    match_conv = mx.symbol.Convolution(name=layer_name + '_match_conv', data=last_output, num_filter=output, pad=(0, 0),
                                       kernel=(1, 1), stride=(2, 2), no_bias=True)
    match_conv_bn = mx.symbol.BatchNorm(name=layer_name + '_match_conv_bn', data=match_conv, use_global_stats=use_global_stats,
                                        fix_gamma=False, eps=0.00001)
    match_conv_scale = match_conv_bn

    conv1 = mx.symbol.Convolution(name=layer_name + '_conv1', data=last_output, num_filter=input, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name=layer_name + '_conv1_bn', data=conv1, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name=layer_name + '_conv1_relu', data=conv1_scale, act_type='relu')

    conv2 = mx.symbol.Convolution(name=layer_name + '_conv2', data=conv1_relu, num_filter=input, num_group=num_group,
                                  pad=(1, 1), kernel=(3, 3), stride=(2, 2), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name=layer_name + '_conv2_bn', data=conv2, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv2_scale = conv2_bn
    conv2_relu = mx.symbol.Activation(name=layer_name + '_conv2_relu', data=conv2_scale, act_type='relu')

    conv3 = mx.symbol.Convolution(name=layer_name + '_conv3', data=conv2_relu, num_filter=output, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv3_bn = mx.symbol.BatchNorm(name=layer_name + '_conv3_bn', data=conv3, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv3_scale = conv3_bn

    eletwise = mx.symbol.broadcast_add(name=layer_name + '_eletwise', lhs=conv3_scale, rhs=match_conv_scale)
    eletwise_relu = mx.symbol.Activation(name=layer_name + '_eletwise_relu', data=eletwise, act_type='relu')
    return eletwise_relu


def block(layer_name, last_output, input, output, num_group):
    conv1 = mx.symbol.Convolution(name=layer_name + '_conv1', data=last_output, num_filter=input, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name=layer_name + '_conv1_bn', data=conv1, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name=layer_name + '_conv1_relu', data=conv1_scale, act_type='relu')

    conv2 = mx.symbol.Convolution(name=layer_name + '_conv2', data=conv1_relu, num_filter=input, num_group=num_group,
                                  pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name=layer_name + '_conv2_bn', data=conv2, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv2_scale = conv2_bn
    conv2_relu = mx.symbol.Activation(name=layer_name + '_conv2_relu', data=conv2_scale, act_type='relu')

    conv3 = mx.symbol.Convolution(name=layer_name + '_conv3', data=conv2_relu, num_filter=output, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv3_bn = mx.symbol.BatchNorm(name=layer_name + '_conv3_bn', data=conv3, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv3_scale = conv3_bn

    eletwise = mx.symbol.broadcast_add(name=layer_name + '_eletwise', lhs=last_output, rhs=conv3_scale)
    eletwise_relu = mx.symbol.Activation(name=layer_name + '_eletwise_relu', data=eletwise, act_type='relu')
    return eletwise_relu


def stage(start_num, num_block, model_name, output_layer, input, output, group):
    for i in xrange(num_block):
        if start_num == 1:
            if i == 0:
                resx1_match_conv = mx.symbol.Convolution(name='resx1_match_conv', data=output_layer, num_filter=output,
                                                         pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
                resx1_match_conv_bn = mx.symbol.BatchNorm(name='resx1_match_conv_bn', data=resx1_match_conv,
                                                          use_global_stats=use_global_stats, fix_gamma=False, eps=0.00001)
                resx1_match_conv_scale = resx1_match_conv_bn

                resx1_conv1 = mx.symbol.Convolution(name='resx1_conv1', data=output_layer, num_filter=input, pad=(0, 0),
                                                    kernel=(1, 1), stride=(1, 1), no_bias=True)
                resx1_conv1_bn = mx.symbol.BatchNorm(name='resx1_conv1_bn', data=resx1_conv1, use_global_stats=use_global_stats,
                                                     fix_gamma=False, eps=0.00001)
                resx1_conv1_scale = resx1_conv1_bn
                resx1_conv1_relu = mx.symbol.Activation(name='resx1_conv1_relu', data=resx1_conv1_scale,
                                                        act_type='relu')

                resx1_conv2 = mx.symbol.Convolution(name='resx1_conv2', data=resx1_conv1_relu, num_filter=input,
                                                    num_group=group, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                                    no_bias=True)
                resx1_conv2_bn = mx.symbol.BatchNorm(name='resx1_conv2_bn', data=resx1_conv2, use_global_stats=use_global_stats,
                                                     fix_gamma=False, eps=0.00001)
                resx1_conv2_scale = resx1_conv2_bn
                resx1_conv2_relu = mx.symbol.Activation(name='resx1_conv2_relu', data=resx1_conv2_scale,
                                                        act_type='relu')

                resx1_conv3 = mx.symbol.Convolution(name='resx1_conv3', data=resx1_conv2_relu, num_filter=output,
                                                    pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
                resx1_conv3_bn = mx.symbol.BatchNorm(name='resx1_conv3_bn', data=resx1_conv3, use_global_stats=use_global_stats,
                                                     fix_gamma=False, eps=0.00001)
                resx1_conv3_scale = resx1_conv3_bn

                resx1_eletwise = mx.symbol.broadcast_add(name='resx1_eletwise', lhs=resx1_conv3_scale,
                                                         rhs=resx1_match_conv_scale)
                output_layer = mx.symbol.Activation(name='resx1_eletwise_relu', data=resx1_eletwise, act_type='relu')
            else:
                output_layer = block(model_name + str(i + start_num), output_layer, input, output, group)
        else:
            if i == 0:
                output_layer = match_block(model_name + str(i + start_num), output_layer, input, output, group)
            else:
                output_layer = block(model_name + str(i + start_num), output_layer, input, output, group)

    return output_layer


def get_symbol(num_classes, num_filter, num_group, model_name, **kwargs):
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name='data')

    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=num_filter, pad=(3, 3), kernel=(7, 7),
                                  stride=(2, 2),
                                  no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1, use_global_stats=use_global_stats, fix_gamma=False, eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1_scale, act_type='relu')

    ########## pool1 #########
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2), pool_type='max',
                              pooling_convention='valid', cudnn_off=False, global_pool=False)

    output_layer = stage(1, 3, model_name, pool1, num_filter * 2, num_filter * 4, num_group)
    output_layer = stage(4, 4, model_name, output_layer, num_filter * 4, num_filter * 8, num_group)
    output_layer = stage(8, 23, model_name, output_layer, num_filter * 8, num_filter * 16, num_group)
    output_layer = stage(31, 3, model_name, output_layer, num_filter * 16, num_filter * 32, num_group)

    ########## avg_pool #########
    avg_pool = mx.symbol.Pooling(name='avg_pool', data=output_layer, kernel=(1, 1), pool_type='avg', global_pool=True,
                                 pooling_convention='full', cudnn_off=False)

    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)
    return softmax


softmax = get_symbol(1000, 64, 32, 'resx', 'use_global_stats')
softmax.save('resnext101-32x4d-symbol.json')
