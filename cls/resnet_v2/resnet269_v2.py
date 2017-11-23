import mxnet as mx


def match_block(layer_name, last_output, input, output):
    bn = mx.symbol.BatchNorm(name=layer_name + '_bn', data=last_output, use_global_stats=use_global_stats, fix_gamma=False,
                             eps=0.00001)
    scale = bn
    relu = mx.symbol.Activation(name=layer_name + '_relu', data=scale, act_type='relu')

    match_conv = mx.symbol.Convolution(name=layer_name + '_match_conv', data=relu, num_filter=output, pad=(0, 0),
                                       kernel=(1, 1), stride=(2, 2), no_bias=True)

    conv1 = mx.symbol.Convolution(name=layer_name + '_conv1', data=relu, num_filter=input, pad=(0, 0), kernel=(1, 1),
                                  stride=(1, 1), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name=layer_name + '_conv1_bn', data=conv1, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name=layer_name + '_conv1_relu', data=conv1_scale, act_type='relu')

    conv2 = mx.symbol.Convolution(name=layer_name + '_conv2', data=conv1_relu, num_filter=input, pad=(1, 1),
                                  kernel=(3, 3), stride=(2, 2), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name=layer_name + '_conv2_bn', data=conv2, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv2_scale = conv2_bn
    conv2_relu = mx.symbol.Activation(name=layer_name + '_conv2_relu', data=conv2_scale, act_type='relu')

    conv3 = mx.symbol.Convolution(name=layer_name + '_conv3', data=conv2_relu, num_filter=output, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)

    eletwise = mx.symbol.broadcast_add(name=layer_name + '_eletwise', lhs=conv3, rhs=match_conv)

    return eletwise


def block(layer_name, last_output, input, output):
    bn = mx.symbol.BatchNorm(name=layer_name + '_bn', data=last_output, use_global_stats=use_global_stats, fix_gamma=False,
                             eps=0.00001)
    scale = bn
    relu = mx.symbol.Activation(name=layer_name + '_relu', data=scale, act_type='relu')

    conv1 = mx.symbol.Convolution(name=layer_name + '_conv1', data=relu, num_filter=input, pad=(0, 0), kernel=(1, 1),
                                  stride=(1, 1), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name=layer_name + '_conv1_bn', data=conv1, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name=layer_name + '_conv1_relu', data=conv1_scale, act_type='relu')

    conv2 = mx.symbol.Convolution(name=layer_name + '_conv2', data=conv1_relu, num_filter=input, pad=(1, 1),
                                  kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name=layer_name + '_conv2_bn', data=conv2, use_global_stats=use_global_stats, fix_gamma=False,
                                   eps=0.00001)
    conv2_scale = conv2_bn
    conv2_relu = mx.symbol.Activation(name=layer_name + '_conv2_relu', data=conv2_scale, act_type='relu')

    conv3 = mx.symbol.Convolution(name=layer_name + '_conv3', data=conv2_relu, num_filter=output, pad=(0, 0),
                                  kernel=(1, 1), stride=(1, 1), no_bias=True)

    eletwise = mx.symbol.broadcast_add(name=layer_name + '_eletwise', lhs=last_output, rhs=conv3)
    return eletwise


def stage(start_num, num_block, model_name, output_layer, input, output):
    for i in xrange(num_block):
        if start_num == 1:
            if i == 0:
                res1_match_conv = mx.symbol.Convolution(name='res1_match_conv', data=output_layer, num_filter=output,
                                                        pad=(0, 0),
                                                        kernel=(1, 1),
                                                        stride=(1, 1), no_bias=True)

                res1_conv1 = mx.symbol.Convolution(name='res1_conv1', data=output_layer, num_filter=input, pad=(0, 0),
                                                   kernel=(1, 1),
                                                   stride=(1, 1), no_bias=True)
                res1_conv1_bn = mx.symbol.BatchNorm(name='res1_conv1_bn', data=res1_conv1, use_global_stats=use_global_stats,
                                                    fix_gamma=False,
                                                    eps=0.00001)
                res1_conv1_scale = res1_conv1_bn
                res1_conv1_relu = mx.symbol.Activation(name='res1_conv1_relu', data=res1_conv1_scale, act_type='relu')

                res1_conv2 = mx.symbol.Convolution(name='res1_conv2', data=res1_conv1_relu, num_filter=input,
                                                   pad=(1, 1),
                                                   kernel=(3, 3),
                                                   stride=(1, 1), no_bias=True)
                res1_conv2_bn = mx.symbol.BatchNorm(name='res1_conv2_bn', data=res1_conv2, use_global_stats=use_global_stats,
                                                    fix_gamma=False,
                                                    eps=0.00001)
                res1_conv2_scale = res1_conv2_bn
                res1_conv2_relu = mx.symbol.Activation(name='res1_conv2_relu', data=res1_conv2_scale, act_type='relu')

                res1_conv3 = mx.symbol.Convolution(name='res1_conv3', data=res1_conv2_relu, num_filter=output,
                                                   pad=(0, 0),
                                                   kernel=(1, 1),
                                                   stride=(1, 1), no_bias=True)

                output_layer = mx.symbol.broadcast_add(name='res1_eletwise', lhs=res1_conv3, rhs=res1_match_conv)
            else:
                output_layer = block(model_name + str(i + start_num), output_layer, input, output)
        else:
            if i == 0:
                output_layer = match_block(model_name + str(i + start_num), output_layer, input, output)
            else:
                output_layer = block(model_name + str(i + start_num), output_layer, input, output)

    return output_layer


def get_symbol(num_classes, num_filter, model_name, **kwargs):
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

    output_layer = stage(1, 3, model_name, pool1, num_filter, num_filter * 4)
    output_layer = stage(4, 30, model_name, output_layer, num_filter * 2, num_filter * 8)
    output_layer = stage(34, 48, model_name, output_layer, num_filter * 4, num_filter * 16)
    output_layer = stage(82, 8, model_name, output_layer, num_filter * 8, num_filter * 32)

    res89_eletwise_bn = mx.symbol.BatchNorm(name='res89_eletwise_bn', data=output_layer, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=0.00001)
    res89_eletwise_scale = res89_eletwise_bn
    res89_eletwise_relu = mx.symbol.Activation(name='res89_eletwise_relu', data=res89_eletwise_scale, act_type='relu')
    ########## avg_pool #########
    avg_pool = mx.symbol.Pooling(name='avg_pool', data=res89_eletwise_relu, kernel=(1, 1), pool_type='avg',
                              global_pool=True,
                              pooling_convention='full', cudnn_off=False)

    ######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)
    return softmax

softmax = get_symbol(1000, 64, 'res', 'use_global_stats')
softmax.save('resnet269-v2-symbol.json')
