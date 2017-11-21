import mxnet as mx

def get_symbol(num_classes, **kwargs):
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']


    data = mx.symbol.Variable(name='data')

    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    conv1_scale = conv1_bn
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1_scale , act_type='relu')

########## pool1 #########
    pool1 = mx.symbol.Pooling(name = 'pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2), pool_type='max',
                          pooling_convention='valid', cudnn_off=False, global_pool=False)

########## res1 ##########
    res1_conv1 = mx.symbol.Convolution(name='res1_conv1', data=pool1 , num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res1_conv1_bn = mx.symbol.BatchNorm(name='res1_conv1_bn', data=res1_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res1_conv1_scale = res1_conv1_bn
    res1_conv1_relu = mx.symbol.Activation(name='res1_conv1_relu', data=res1_conv1_scale , act_type='relu')

    res1_conv2 = mx.symbol.Convolution(name='res1_conv2', data=res1_conv1_relu , num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res1_conv2_bn = mx.symbol.BatchNorm(name='res1_conv2_bn', data=res1_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res1_conv2_scale = res1_conv2_bn

    res1 = mx.symbol.broadcast_add(name='res1', lhs=pool1, rhs=res1_conv2_scale)
    res1_relu = mx.symbol.Activation(name='res1_relu', data=res1 , act_type='relu')

########## res2 ##########
    res2_conv1 = mx.symbol.Convolution(name='res2_conv1', data=res1_relu, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res2_conv1_bn = mx.symbol.BatchNorm(name='res2_conv1_bn', data=res2_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res2_conv1_scale = res2_conv1_bn
    res2_conv1_relu = mx.symbol.Activation(name='res2_conv1_relu', data=res2_conv1_scale , act_type='relu')

    res2_conv2 = mx.symbol.Convolution(name='res2_conv2', data=res2_conv1_relu , num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res2_conv2_bn = mx.symbol.BatchNorm(name='res2_conv2_bn', data=res2_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res2_conv2_scale = res2_conv2_bn

    res2 = mx.symbol.broadcast_add(name='res2', lhs=res1_relu, rhs=res2_conv2_scale)
    res2_relu = mx.symbol.Activation(name='res2_relu', data=res2 , act_type='relu')

########## res3 ##########
    res3_match_conv = mx.symbol.Convolution(name='res3_match_conv', data=res2_relu , num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(2, 2), no_bias=True)
    res3_match_conv_bn = mx.symbol.BatchNorm(name='res3_match_conv_bn', data=res3_match_conv , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res3_match_conv_scale = res3_match_conv_bn

    res3_conv1 = mx.symbol.Convolution(name='res3_conv1', data=res2_relu, num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(2, 2), no_bias=True)
    res3_conv1_bn = mx.symbol.BatchNorm(name='res3_conv1_bn', data=res3_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res3_conv1_scale = res3_conv1_bn
    res3_conv1_relu = mx.symbol.Activation(name='res3_conv1_relu', data=res3_conv1_scale , act_type='relu')

    res3_conv2 = mx.symbol.Convolution(name='res3_conv2', data=res3_conv1_relu , num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res3_conv2_bn = mx.symbol.BatchNorm(name='res3_conv2_bn', data=res3_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res3_conv2_scale = res3_conv2_bn

    res3 = mx.symbol.broadcast_add(name='res3', lhs=res3_match_conv_scale, rhs=res3_conv2_scale)
    res3_relu = mx.symbol.Activation(name='res3_relu', data=res3 , act_type='relu')

########## res4 ##########
    res4_conv1 = mx.symbol.Convolution(name='res4_conv1', data=res3_relu, num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res4_conv1_bn = mx.symbol.BatchNorm(name='res4_conv1_bn', data=res4_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res4_conv1_scale = res4_conv1_bn
    res4_conv1_relu = mx.symbol.Activation(name='res4_conv1_relu', data=res4_conv1_scale , act_type='relu')

    res4_conv2 = mx.symbol.Convolution(name='res4_conv2', data=res4_conv1_relu , num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res4_conv2_bn = mx.symbol.BatchNorm(name='res4_conv2_bn', data=res4_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res4_conv2_scale = res4_conv2_bn

    res4 = mx.symbol.broadcast_add(name='res4', lhs=res3_relu, rhs=res4_conv2_scale)
    res4_relu = mx.symbol.Activation(name='res4_relu', data=res4 , act_type='relu')

########## res5 ##########
    res5_match_conv = mx.symbol.Convolution(name='res5_match_conv', data=res4_relu , num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(2, 2), no_bias=True)
    res5_match_conv_bn = mx.symbol.BatchNorm(name='res5_match_conv_bn', data=res5_match_conv , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res5_match_conv_scale = res5_match_conv_bn

    res5_conv1 = mx.symbol.Convolution(name='res5_conv1', data=res4_relu, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(2, 2), no_bias=True)
    res5_conv1_bn = mx.symbol.BatchNorm(name='res5_conv1_bn', data=res5_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res5_conv1_scale = res5_conv1_bn
    res5_conv1_relu = mx.symbol.Activation(name='res5_conv1_relu', data=res5_conv1_scale , act_type='relu')

    res5_conv2 = mx.symbol.Convolution(name='res5_conv2', data=res5_conv1_relu , num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res5_conv2_bn = mx.symbol.BatchNorm(name='res5_conv2_bn', data=res5_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res5_conv2_scale = res5_conv2_bn

    res5 = mx.symbol.broadcast_add(name='res5', lhs=res5_match_conv_scale, rhs=res5_conv2_scale)
    res5_relu = mx.symbol.Activation(name='res5_relu', data=res5 , act_type='relu')

########## res6 ##########
    res6_conv1 = mx.symbol.Convolution(name='res6_conv1', data=res5_relu, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res6_conv1_bn = mx.symbol.BatchNorm(name='res6_conv1_bn', data=res6_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res6_conv1_scale = res6_conv1_bn
    res6_conv1_relu = mx.symbol.Activation(name='res6_conv1_relu', data=res6_conv1_scale , act_type='relu')

    res6_conv2 = mx.symbol.Convolution(name='res6_conv2', data=res6_conv1_relu , num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res6_conv2_bn = mx.symbol.BatchNorm(name='res6_conv2_bn', data=res6_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res6_conv2_scale = res6_conv2_bn

    res6 = mx.symbol.broadcast_add(name='res6', lhs=res5_relu, rhs=res6_conv2_scale)
    res6_relu = mx.symbol.Activation(name='res6_relu', data=res6 , act_type='relu')

########## res7 ##########
    res7_match_conv = mx.symbol.Convolution(name='res7_match_conv', data=res6_relu , num_filter=512, pad=(0, 0), kernel=(1, 1), stride=(2, 2), no_bias=True)
    res7_match_conv_bn = mx.symbol.BatchNorm(name='res7_match_conv_bn', data=res7_match_conv , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res7_match_conv_scale = res7_match_conv_bn

    res7_conv1 = mx.symbol.Convolution(name='res7_conv1', data=res6_relu, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2), no_bias=True)
    res7_conv1_bn = mx.symbol.BatchNorm(name='res7_conv1_bn', data=res7_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res7_conv1_scale = res7_conv1_bn
    res7_conv1_relu = mx.symbol.Activation(name='res7_conv1_relu', data=res7_conv1_scale , act_type='relu')

    res7_conv2 = mx.symbol.Convolution(name='res7_conv2', data=res7_conv1_relu , num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res7_conv2_bn = mx.symbol.BatchNorm(name='res7_conv2_bn', data=res7_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res7_conv2_scale = res7_conv2_bn

    res7 = mx.symbol.broadcast_add(name='res7', lhs=res7_match_conv_scale, rhs=res7_conv2_scale)
    res7_relu = mx.symbol.Activation(name='res7_relu', data=res7 , act_type='relu')

########## res8 ##########
    res8_conv1 = mx.symbol.Convolution(name='res8_conv1', data=res7_relu, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res8_conv1_bn = mx.symbol.BatchNorm(name='res8_conv1_bn', data=res8_conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res8_conv1_scale = res8_conv1_bn
    res8_conv1_relu = mx.symbol.Activation(name='res8_conv1_relu', data=res8_conv1_scale , act_type='relu')

    res8_conv2 = mx.symbol.Convolution(name='res8_conv2', data=res8_conv1_relu , num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    res8_conv2_bn = mx.symbol.BatchNorm(name='res8_conv2_bn', data=res8_conv2 , use_global_stats=True, fix_gamma=False, eps=0.00001)
    res8_conv2_scale = res8_conv2_bn

    res8 = mx.symbol.broadcast_add(name='res8', lhs=res7_relu, rhs=res8_conv2_scale)
    res8_relu = mx.symbol.Activation(name='res8_relu', data=res8 , act_type='relu')

########## pool5 #########
    avg_pool = mx.symbol.Pooling(name='avg_pool', data=res8_relu, kernel=(1, 1), pool_type='avg', global_pool=True,
                          pooling_convention='full', cudnn_off=False)

######## classifier ######
    flatten = mx.symbol.Flatten(name='flatten', data=avg_pool)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten , num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)

    return softmax

softmax = get_symbol(1000)
softmax.save('resnet18-priv-symbol.json')







