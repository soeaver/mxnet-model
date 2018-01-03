import mxnet as mx


def inception_stem(data,
                   num1_1, num1_2, num1_3,
                   num2_1, num2_2,
                   num3_1, num3_2, num3_3, num3_4, num3_5, num3_6, num3_7,
                   use_global_stats
                   ):
    conv1_3x3_s2 = mx.symbol.Convolution(name='conv1_3x3_s2', data=data, num_filter=num1_1, pad=(0, 0), kernel=(3, 3),
                                         stride=(2, 2), no_bias=True)
    conv1_3x3_s2_bn = mx.symbol.BatchNorm(name='conv1_3x3_s2_bn', data=conv1_3x3_s2, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv1_3x3_s2_scale = conv1_3x3_s2_bn
    conv1_3x3_relu = mx.symbol.Activation(name='conv1_3x3_relu', data=conv1_3x3_s2_scale, act_type='relu')

    conv2_3x3_s1 = mx.symbol.Convolution(name='conv2_3x3_s1', data=conv1_3x3_relu, num_filter=num1_2, pad=(0, 0),
                                         kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv2_3x3_s1_bn = mx.symbol.BatchNorm(name='conv2_3x3_s1_bn', data=conv2_3x3_s1, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv2_3x3_s1_scale = conv2_3x3_s1_bn
    conv2_3x3_relu = mx.symbol.Activation(name='conv2_3x3_relu', data=conv2_3x3_s1_scale, act_type='relu')

    conv3_3x3_s1 = mx.symbol.Convolution(name='conv3_3x3_s1', data=conv2_3x3_relu, num_filter=num1_3, pad=(1, 1),
                                         kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv3_3x3_s1_bn = mx.symbol.BatchNorm(name='conv3_3x3_s1_bn', data=conv3_3x3_s1, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv3_3x3_s1_scale = conv3_3x3_s1_bn
    conv3_3x3_relu = mx.symbol.Activation(name='conv3_3x3_relu', data=conv3_3x3_s1_scale, act_type='relu')

    ########## pool1 #########
    pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv3_3x3_relu, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)

    conv4_3x3_reduce = mx.symbol.Convolution(name='conv4_3x3_reduce', data=pool1_3x3_s2, num_filter=num2_1, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv4_3x3_reduce_bn = mx.symbol.BatchNorm(name='conv4_3x3_reduce_bn', data=conv4_3x3_reduce,
                                              use_global_stats=use_global_stats, fix_gamma=False, eps=0.001)
    conv4_3x3_reduce_scale = conv4_3x3_reduce_bn
    conv4_relu_3x3_reduce = mx.symbol.Activation(name='conv4_relu_3x3_reduce', data=conv4_3x3_reduce_scale,
                                                 act_type='relu')

    conv4_3x3 = mx.symbol.Convolution(name='conv4_3x3', data=conv4_relu_3x3_reduce, num_filter=num2_2, pad=(0, 0),
                                      kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv4_3x3_bn = mx.symbol.BatchNorm(name='conv4_3x3_bn', data=conv4_3x3, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    conv4_3x3_scale = conv4_3x3_bn
    conv4_relu_3x3 = mx.symbol.Activation(name='conv4_relu_3x3', data=conv4_3x3_scale, act_type='relu')

    ########## pool2 #########
    pool2_3x3_s2 = mx.symbol.Pooling(name='pool2_3x3_s2', data=conv4_relu_3x3, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)
    # --------------------1x1------------------
    conv5_1x1 = mx.symbol.Convolution(name='conv5_1x1', data=pool2_3x3_s2, num_filter=num3_1, pad=(0, 0), kernel=(1, 1),
                                         stride=(1, 1), no_bias=True)
    conv5_1x1_bn = mx.symbol.BatchNorm(name='conv5_1x1_bn', data=conv5_1x1, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv5_1x1_scale = conv5_1x1_bn
    conv5_1x1_relu = mx.symbol.Activation(name='conv5_1x1_relu', data=conv5_1x1_scale, act_type='relu')

    # --------------------5x5-------------------
    conv5_5x5_reduce = mx.symbol.Convolution(name='conv5_5x5_reduce', data=pool2_3x3_s2, num_filter=num3_2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv5_5x5_reduce_bn = mx.symbol.BatchNorm(name='conv5_5x5_reduce_bn', data=conv5_5x5_reduce,
                                              use_global_stats=use_global_stats, fix_gamma=False, eps=0.001)
    conv5_5x5_reduce_scale = conv5_5x5_reduce_bn
    conv5_5x5_reduce_relu = mx.symbol.Activation(name='conv5_5x5_reduce_relu', data=conv5_5x5_reduce_scale,
                                                 act_type='relu')

    conv5_5x5 = mx.symbol.Convolution(name='conv5_5x5', data=conv5_5x5_reduce_relu, num_filter=num3_3, pad=(2, 2),
                                      kernel=(5, 5), stride=(1, 1), no_bias=True)
    conv5_5x5_bn = mx.symbol.BatchNorm(name='conv5_5x5_bn', data=conv5_5x5, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    conv5_5x5_scale = conv5_5x5_bn
    conv5_5x5_relu = mx.symbol.Activation(name='conv5_5x5_relu', data=conv5_5x5_scale, act_type='relu')

    # ---------------------3x3------------------
    conv5_3x3_reduce = mx.symbol.Convolution(name='conv5_3x3_reduce', data=pool2_3x3_s2, num_filter=num3_4, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv5_3x3_reduce_bn = mx.symbol.BatchNorm(name='conv5_3x3_reduce_bn', data=conv5_3x3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    conv5_3x3_reduce_scale = conv5_3x3_reduce_bn
    conv5_3x3_reduce_relu = mx.symbol.Activation(name='conv5_3x3_reduce_relu', data=conv5_3x3_reduce_scale, act_type='relu')

    conv5_3x3 = mx.symbol.Convolution(name='conv5_3x3', data=conv5_3x3_reduce_relu, num_filter=num3_5, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv5_3x3_bn = mx.symbol.BatchNorm(name='conv5_3x3_bn', data=conv5_3x3, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    conv5_3x3_scale = conv5_3x3_bn
    conv5_3x3_relu = mx.symbol.Activation(name='conv5_3x3_relu', data=conv5_3x3_scale, act_type='relu')

    conv5_3x3_2 = mx.symbol.Convolution(name='conv5_3x3_2', data=conv5_3x3_relu, num_filter=num3_6, pad=(1, 1), kernel=(3, 3),
                                 stride=(1, 1), no_bias=True)
    conv5_3x3_2_bn = mx.symbol.BatchNorm(name='conv5_3x3_2_bn', data=conv5_3x3_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    conv5_3x3_2_scale = conv5_3x3_2_bn
    conv5_3x3_2_relu = mx.symbol.Activation(name='conv5_3x3_2_relu', data=conv5_3x3_2_scale, act_type='relu')

    # --------------------pool------------------
    ave_pool = mx.symbol.Pooling(name='ave_pool', data=pool2_3x3_s2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), pool_type='avg',
                           pooling_convention='full')

    conv5_1x1_ave = mx.symbol.Convolution(name='conv5_1x1_ave', data=ave_pool, num_filter=num3_7, pad=(0, 0), kernel=(1, 1),
                                    stride=(1, 1),
                                    no_bias=True)
    conv5_1x1_ave_bn = mx.symbol.BatchNorm(name='conv5_1x1_ave_bn', data=conv5_1x1_ave, use_global_stats=use_global_stats,
                                     fix_gamma=False, eps=0.001)
    conv5_1x1_ave_scale = conv5_1x1_ave_bn
    conv5_1x1_ave_relu = mx.symbol.Activation(name='conv5_1x1_ave_relu', data=conv5_1x1_ave_scale, act_type='relu')

    stem_concat = mx.symbol.Concat(*[conv5_1x1_relu, conv5_5x5_relu, conv5_3x3_2_relu, conv5_1x1_ave_relu], name='stem_concat')
    stem_concat = 1.0*stem_concat

    return stem_concat


def Inception_resnet_blockA(data,
                           num1_1,
                           num2_1, num2_2,
                           num3_1, num3_2, num3_3,
                           num4_1,
                           name, use_global_stats,
                           ):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a1_scale, act_type='relu')

    # --------------3x3---------------
    a2_reduce = mx.symbol.Convolution(name=name + '_3x3_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a2_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_reduce_bn', data=a2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a2_reduce_scale = a2_reduce_bn
    a2_reduce_relu = mx.symbol.Activation(name=name + '_3x3_reduce_relu', data=a2_reduce_scale, act_type='relu')

    a2 = mx.symbol.Convolution(name=name + '_3x3', data=a2_reduce_relu, num_filter=num2_2, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    a2_bn = mx.symbol.BatchNorm(name=name + '_3x3_bn', data=a2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a2_scale = a2_bn
    a2_relu = mx.symbol.Activation(name=name + '_3x3_relu', data=a2_scale, act_type='relu')

    # --------------2*3x3---------------
    a3_reduce = mx.symbol.Convolution(name=name + '_3x3_2_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a3_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_reduce_bn', data=a3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a3_reduce_scale = a3_reduce_bn
    a3_reduce_relu = mx.symbol.Activation(name=name + '_3x3_2_reduce_relu', data=a3_reduce_scale, act_type='relu')

    a3_1 = mx.symbol.Convolution(name=name + '_3x3_2', data=a3_reduce_relu, num_filter=num3_2, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    a3_1_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_bn', data=a3_1, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_1_scale = a3_1_bn
    a3_1_relu = mx.symbol.Activation(name=name + '_3x3_2_relu', data=a3_1_scale, act_type='relu')

    a3_2 = mx.symbol.Convolution(name=name + '_3x3_3', data=a3_1_relu, num_filter=num3_3, pad=(1, 1), kernel=(3, 3),
                                 stride=(1, 1), no_bias=True)
    a3_2_bn = mx.symbol.BatchNorm(name=name + '_3x3_3_bn', data=a3_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_2_scale = a3_2_bn
    a3_2_relu = mx.symbol.Activation(name=name + '_3x3_3_relu', data=a3_2_scale, act_type='relu')

    # -------------concat,up,eltwise------------
    concat = mx.symbol.Concat(*[a1_relu, a2_relu, a3_2_relu], name=name + '_concat')
    up = mx.symbol.Convolution(name=name + '_up', data=concat, num_filter=num4_1, pad=(0, 0), kernel=(1, 1),
                            stride=(1, 1), no_bias=False)
    up = 0.170000001788*up
    eltwise = mx.symbol.elemwise_add(data, up, name=name + '_residual_eltwise')
    block_output = mx.symbol.Activation(name=name + '_residual_eltwise_relu', data=eltwise, act_type='relu')
    block_output = 1.0*block_output

    return block_output


def Inception_resnet_blockB(data,
               num1_1,
               num2_1, num2_2, num2_3,
               num3_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1),
                               no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a1_scale, act_type='relu')

    # --------------1x7,7x1---------------
    a2_reduce = mx.symbol.Convolution(name=name + '_1x7_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    a2_reduce_bn = mx.symbol.BatchNorm(name=name + '_1x7_reduce_bn', data=a2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a2_reduce_scale = a2_reduce_bn
    a2_reduce_relu = mx.symbol.Activation(name=name + '_1x7_reduce_relu', data=a2_reduce_scale, act_type='relu')

    a2 = mx.symbol.Convolution(name=name + '_1x7', data=a2_reduce_relu, num_filter=num2_2, pad=(0, 3), kernel=(1, 7),
                               stride=(1, 1),
                               no_bias=True)
    a2_bn = mx.symbol.BatchNorm(name=name + '_1x7_bn', data=a2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a2_scale = a2_bn
    a2_relu = mx.symbol.Activation(name=name + '_1x7_relu', data=a2_scale, act_type='relu')

    a2_2 = mx.symbol.Convolution(name=name + '_7x1', data=a2_relu, num_filter=num2_3, pad=(3, 0), kernel=(7, 1),
                                 stride=(1, 1),
                                 no_bias=True)
    a2_2_bn = mx.symbol.BatchNorm(name=name + '_7x1_bn', data=a2_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a2_2_scale = a2_2_bn
    a2_2_relu = mx.symbol.Activation(name=name + '_7x1_relu', data=a2_2_scale, act_type='relu')

    # -------------concat,up,eltwise------------
    concat = mx.symbol.Concat(*[a1_relu, a2_2_relu], name=name + '_concat')
    up = mx.symbol.Convolution(name=name + '_up', data=concat, num_filter=num3_1, pad=(0, 0), kernel=(1, 1),
                            stride=(1, 1), no_bias=False)
    up = 0.10000000149*up
    eltwise = mx.symbol.elemwise_add(data, up, name=name + '_residual_eltwise')
    block_output = mx.symbol.Activation(name=name + '_residual_eltwise_relu', data=eltwise, act_type='relu')
    block_output = 1.0*block_output

    return block_output


def Inception_resnet_blockC(data,
               num1_1,
               num2_1, num2_2, num2_3,
               num3_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a1_scale, act_type='relu')

    # --------------1x3,3x1,parallel---------------
    a2_reduce = mx.symbol.Convolution(name=name + '_1x3_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a2_reduce_bn = mx.symbol.BatchNorm(name=name + '_1x3_reduce_bn', data=a2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a2_reduce_scale = a2_reduce_bn
    a2_reduce_relu = mx.symbol.Activation(name=name + '_1x3_reduce_relu', data=a2_reduce_scale, act_type='relu')

    a2 = mx.symbol.Convolution(name=name + '_1x3', data=a2_reduce_relu, num_filter=num2_2, pad=(0, 1), kernel=(1, 3),
                               stride=(1, 1), no_bias=True)
    a2_bn = mx.symbol.BatchNorm(name=name + '_1x3_bn', data=a2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a2_scale = a2_bn
    a2_relu = mx.symbol.Activation(name=name + '_1x3_relu', data=a2_scale, act_type='relu')

    a2_2 = mx.symbol.Convolution(name=name + '_3x1', data=a2_relu, num_filter=num2_3, pad=(1, 0), kernel=(3, 1),
                                 stride=(1, 1), no_bias=True)
    a2_2_bn = mx.symbol.BatchNorm(name=name + '_3x1_bn', data=a2_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a2_2_scale = a2_2_bn
    a2_2_relu = mx.symbol.Activation(name=name + '_3x1_relu', data=a2_2_scale, act_type='relu')

    # -------------concat,up,eltwise------------
    concat = mx.symbol.Concat(*[a1_relu, a2_2_relu], name=name + '_concat')
    up = mx.symbol.Convolution(name=name + '_up', data=concat, num_filter=num3_1, pad=(0, 0), kernel=(1, 1),
                            stride=(1, 1), no_bias=False)
    up = 0.20000000298*up
    eltwise = mx.symbol.elemwise_add(data, up, name=name + '_residual_eltwise')
    block_output = mx.symbol.Activation(name=name + '_residual_eltwise_relu', data=eltwise, act_type='relu')
    block_output = 1.0*block_output

    return block_output


def ReductionA(data,
               num1_1,
               num2_1, num2_2, num2_3,
               name, use_global_stats):
    # ---------------3x3_1--------------
    r1 = mx.symbol.Convolution(name=name + '_3x3', data=data, num_filter=num1_1, pad=(0, 0), kernel=(3, 3),
                               stride=(2, 2),
                               no_bias=True)
    r1_bn = mx.symbol.BatchNorm(name=name + '_3x3_bn', data=r1, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r1_scale = r1_bn
    r1_relu = mx.symbol.Activation(name=name + '_3x3_relu', data=r1_scale, act_type='relu')

    # ---------------3x3_2--------------
    r2_reduce = mx.symbol.Convolution(name=name + '_3x3_2_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    r2_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_reduce_bn', data=r2_reduce,
                                       use_global_stats=use_global_stats, fix_gamma=False, eps=0.001)
    r2_reduce_scale = r2_reduce_bn
    r2_reduce_relu = mx.symbol.Activation(name=name + '_3x3_2_reduce_relu', data=r2_reduce_scale, act_type='relu')
    r2 = mx.symbol.Convolution(name=name + '_3x3_2', data=r2_reduce_relu, num_filter=num2_2, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1),
                               no_bias=True)
    r2_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_bn', data=r2, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r2_scale = r2_bn
    r2_relu = mx.symbol.Activation(name=name + '_3x3_2_relu', data=r2_scale, act_type='relu')
    r2_2 = mx.symbol.Convolution(name=name + '_3x3_3', data=r2_relu, num_filter=num2_3, pad=(0, 0), kernel=(3, 3),
                                 stride=(2, 2),
                                 no_bias=True)
    r2_2_bn = mx.symbol.BatchNorm(name=name + '_3x3_3_bn', data=r2_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    r2_2_scale = r2_2_bn
    r2_2_relu = mx.symbol.Activation(name=name + '_3x3_3_relu', data=r2_2_scale, act_type='relu')

    # ---------------pool--------------
    r3 = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    reduction_output = mx.symbol.Concat(*[r1_relu, r2_2_relu, r3], name=name + '_concat')
    reduction_output = 1.0*reduction_output

    return reduction_output


def ReductionB(data,
               num1_1, num1_2,
               num2_1, num2_2,
               num3_1, num3_2, num3_3,
               name, use_global_stats):
    # ---------------3x3--------------
    r1_reduce = mx.symbol.Convolution(name=name + '_3x3_reduce', data=data, num_filter=num1_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    r1_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_reduce_bn', data=r1_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    r1_reduce_scale = r1_reduce_bn
    r1_reduce_relu = mx.symbol.Activation(name=name + '_3x3_reduce_relu', data=r1_reduce_scale, act_type='relu')

    r1 = mx.symbol.Convolution(name=name + '_3x3', data=r1_reduce_relu, num_filter=num1_2, pad=(0, 0), kernel=(3, 3),
                               stride=(2, 2),
                               no_bias=True)
    r1_bn = mx.symbol.BatchNorm(name=name + '_3x3_bn', data=r1, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r1_scale = r1_bn
    r1_relu = mx.symbol.Activation(name=name + '_3x3_relu', data=r1_scale, act_type='relu')

    # ---------------3x3_2--------------
    r2_reduce = mx.symbol.Convolution(name=name + '_3x3_2_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    r2_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_reduce_bn', data=r2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    r2_reduce_scale = r2_reduce_bn
    r2_reduce_relu = mx.symbol.Activation(name=name + '_3x3_2_reduce_relu', data=r2_reduce_scale, act_type='relu')

    r2 = mx.symbol.Convolution(name=name + '_3x3_2', data=r2_reduce_relu, num_filter=num2_2, pad=(0, 0), kernel=(3, 3),
                               stride=(2, 2),
                               no_bias=True)
    r2_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_bn', data=r2, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r2_scale = r2_bn
    r2_relu = mx.symbol.Activation(name=name + '_3x3_2_relu', data=r2_scale, act_type='relu')

    # ---------------3x3_3,3x3_4--------------
    r3_reduce = mx.symbol.Convolution(name=name + '_3x3_3_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    r3_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_3_reduce_bn', data=r3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    r3_reduce_scale = r3_reduce_bn
    r3_reduce_relu = mx.symbol.Activation(name=name + '_3x3_3_reduce_relu', data=r3_reduce_scale, act_type='relu')

    r3_1 = mx.symbol.Convolution(name=name + '_3x3_3', data=r3_reduce_relu, num_filter=num3_2, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1),
                               no_bias=True)
    r3_1_bn = mx.symbol.BatchNorm(name=name + '_3x3_3_bn', data=r3_1, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r3_1_scale = r3_1_bn
    r3_1_relu = mx.symbol.Activation(name=name + '_3x3_3_relu', data=r3_1_scale, act_type='relu')

    r3_2 = mx.symbol.Convolution(name=name + '_3x3_4', data=r3_1_relu, num_filter=num3_3, pad=(0, 0), kernel=(3, 3),
                               stride=(2, 2),
                               no_bias=True)
    r3_2_bn = mx.symbol.BatchNorm(name=name + '_3x3_4_bn', data=r3_2, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r3_2_scale = r3_2_bn
    r3_2_relu = mx.symbol.Activation(name=name + '_3x3_4_relu', data=r3_2_scale, act_type='relu')

    # ---------------pool-----------------
    r3 = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    reduction_output = mx.symbol.Concat(*[r1_relu, r2_relu, r3_2_relu, r3], name=name + '_concat')
    reduction_output = 1.0*reduction_output

    return reduction_output


def get_symbol(num_classes):
    use_global_stats = True
    # if 'use_global_stats' not in kwargs:
    #     use_global_stats = False
    # else:
    #     use_global_stats = kwargs['use_global_stats']
    # stem
    data = mx.symbol.Variable(name='data')
    num1_1, num1_2, num1_3 = (32, 32, 64)
    num2_1, num2_2 = (80, 192)
    num3_1, num3_2, num3_3, num3_4, num3_5, num3_6, num3_7 = (96, 48, 64, 64, 96, 96, 64)
    stem = inception_stem(data, num1_1, num1_2, num1_3, num2_1, num2_2, num3_1, num3_2, num3_3, num3_4, num3_5, num3_6, num3_7, use_global_stats)

    # 10*Inception-resnet-blockA
    num1_1 = 32
    num2_1, num2_2 = (32, 32)
    num3_1, num3_2, num3_3 = (32, 48, 64)
    num4_1 = 320
    name = 'inception_resnet_v2_a1'
    inception_resnet_v2_a = Inception_resnet_blockA(stem, num1_1, num2_1, num2_2, num3_1, num3_2, num3_3, num4_1, name, use_global_stats)

    for i in range(9):
        name = 'inception_resnet_v2_a' + str(i+2)
        inception_resnet_v2_a = Inception_resnet_blockA(inception_resnet_v2_a, num1_1, num2_1, num2_2, num3_1, num3_2, num3_3, num4_1, name, use_global_stats)

    num1_1 = 384
    num2_1, num2_2, num2_3 = (256, 256, 384)
    name = 'reduction_a'
    reduction_a = ReductionA(inception_resnet_v2_a, num1_1, num2_1, num2_2, num2_3, name, use_global_stats)

    # 20*Inception-resnet-blockB
    num1_1 = 192
    num2_1, num2_2, num2_3 = (128, 160, 192)
    num3_1 = 1088
    name = 'inception_resnet_v2_b1'
    inception_resnet_v2_b = Inception_resnet_blockB(reduction_a, num1_1, num2_1, num2_2, num2_3, num3_1, name, use_global_stats)

    for i in range(19):
        name = 'inception_resnet_v2_b' + str(i + 2)
        inception_resnet_v2_b = Inception_resnet_blockB(inception_resnet_v2_b, num1_1, num2_1, num2_2, num2_3, num3_1, name, use_global_stats)

    # ReductionB
    num1_1, num1_2 = (256, 384)
    num2_1, num2_2 = (256, 288)
    num3_1, num3_2, num3_3 = (256, 288, 320)
    name = 'reduction_b'
    reduction_b = ReductionB(inception_resnet_v2_b, num1_1, num1_2, num2_1, num2_2, num3_1, num3_2, num3_3, name, use_global_stats)

    # 10*Inception-resnet-blockC
    num1_1 = 192
    num2_1, num2_2, num2_3 = (192, 224, 256)
    num3_1 = 2080
    name = 'inception_resnet_v2_c1'
    inception_resnet_v2_c = Inception_resnet_blockC(reduction_b, num1_1, num2_1, num2_2, num2_3, num3_1, name, use_global_stats)

    for i in range(9):
        name = 'inception_resnet_v2_c' + str(i+2)
        inception_resnet_v2_c = Inception_resnet_blockC(inception_resnet_v2_c, num1_1, num2_1, num2_2, num2_3, num3_1, name, use_global_stats)

    # classifier
    conv6_1x1 = mx.symbol.Convolution(name='conv6_1x1', data=inception_resnet_v2_c, num_filter=1536, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1),
                               no_bias=True)
    conv6_1x1_bn = mx.symbol.BatchNorm(name='conv6_1x1_bn', data=conv6_1x1, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    conv6_1x1_scale = conv6_1x1_bn
    conv6_1x1_relu = mx.symbol.Activation(name='conv6_1x1_relu', data=conv6_1x1_scale, act_type='relu')

    pool_8x8_s1 = mx.symbol.Pooling(name='pool_8x8_s1', data=conv6_1x1_relu, kernel=(1, 1), pool_type='avg',
                           pooling_convention='full', global_pool=True)
    pool_8x8_s1_drop = mx.symbol.Dropout(data=pool_8x8_s1, p=0.2, name='pool_8x8_s1_drop')
    flatten_0 = mx.symbol.Flatten(data=pool_8x8_s1_drop, name="flatten_0")
    classifier = mx.symbol.FullyConnected(data=flatten_0, num_hidden=num_classes, name='classifier', no_bias=False)
    prob = mx.symbol.SoftmaxOutput(data=classifier, name='prob')

    return prob

if __name__ == '__main__':
    net = get_symbol(1000)
    # shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
    net.save('inception-resnet-v2-symbol.json')






