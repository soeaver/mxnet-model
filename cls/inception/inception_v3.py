import mxnet as mx


def inception_stem(data,
                   num1_1, num1_2, num1_3,
                   num2_1, num2_2,
                   use_global_stats
                   ):
    conv1_3x3_s2 = mx.symbol.Convolution(name='conv1_3x3_s2', data=data, num_filter=num1_1, pad=(0, 0), kernel=(3, 3),
                                         stride=(2, 2), no_bias=True)
    conv1_3x3_s2_bn = mx.symbol.BatchNorm(name='conv1_3x3_s2_bn', data=conv1_3x3_s2, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv1_3x3_s2_scale = conv1_3x3_s2_bn
    conv1_3x3_s2_relu = mx.symbol.Activation(name='conv1_3x3_s2_relu', data=conv1_3x3_s2_scale, act_type='relu')

    conv2_3x3_s1 = mx.symbol.Convolution(name='conv2_3x3_s1', data=conv1_3x3_s2_relu, num_filter=num1_2, pad=(0, 0),
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
    pool2_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv4_relu_3x3, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)

    return pool2_3x3_s2


def InceptionA(data,
               num1_1,
               num2_1, num2_2,
               num3_1, num3_2, num3_3,
               num4_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a1_scale, act_type='relu')

    # --------------5x5---------------
    a2_reduce = mx.symbol.Convolution(name=name + '_5x5_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a2_reduce_bn = mx.symbol.BatchNorm(name=name + '_5x5_reduce_bn', data=a2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a2_reduce_scale = a2_reduce_bn
    a2_reduce_relu = mx.symbol.Activation(name=name + '_5x5_reduce_relu', data=a2_reduce_scale, act_type='relu')

    a2 = mx.symbol.Convolution(name=name + '_5x5', data=a2_reduce_relu, num_filter=num2_2, pad=(2, 2), kernel=(5, 5),
                               stride=(1, 1), no_bias=True)
    a2_bn = mx.symbol.BatchNorm(name=name + '_5x5_bn', data=a2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a2_scale = a2_bn
    a2_relu = mx.symbol.Activation(name=name + '_5x5_relu', data=a2_scale, act_type='relu')

    # --------------3x3---------------
    a3_reduce = mx.symbol.Convolution(name=name + '_3x3_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a3_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_reduce_bn', data=a3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a3_reduce_scale = a3_reduce_bn
    a3_reduce_relu = mx.symbol.Activation(name=name + '_3x3_reduce_relu', data=a3_reduce_scale, act_type='relu')

    a3_1 = mx.symbol.Convolution(name=name + '_3x3_1', data=a3_reduce_relu, num_filter=num3_2, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    a3_1_bn = mx.symbol.BatchNorm(name=name + '_3x3_1_bn', data=a3_1, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_1_scale = a3_1_bn
    a3_1_relu = mx.symbol.Activation(name=name + '_3x3_1_relu', data=a3_1_scale, act_type='relu')

    a3_2 = mx.symbol.Convolution(name=name + '_3x3_2', data=a3_1_relu, num_filter=num3_3, pad=(1, 1), kernel=(3, 3),
                                 stride=(1, 1), no_bias=True)
    a3_2_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_bn', data=a3_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_2_scale = a3_2_bn
    a3_2_relu = mx.symbol.Activation(name=name + '_3x3_2_relu', data=a3_2_scale, act_type='relu')

    # --------------pool---------------
    a4 = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), pool_type='avg',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    a4_proj = mx.symbol.Convolution(name=name + '_pool_proj', data=a4, num_filter=num4_1, pad=(0, 0), kernel=(1, 1),
                                    stride=(1, 1),
                                    no_bias=True)
    a4_proj_bn = mx.symbol.BatchNorm(name=name + '_pool_proj_bn', data=a4_proj, use_global_stats=use_global_stats,
                                     fix_gamma=False, eps=0.001)
    a4_proj_scale = a4_proj_bn
    a4_proj_relu = mx.symbol.Activation(name=name + '_pool_proj_relu', data=a4_proj_scale, act_type='relu')

    inception_output = mx.symbol.Concat(*[a1_relu, a2_relu, a3_2_relu, a4_proj_relu], name=name + '_output')

    return inception_output


def InceptionB(data,
               num1_1,
               num2_1, num2_2, num2_3,
               num3_1, num3_2, num3_3, num3_4, num3_5,
               num4_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1_2', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1),
                               no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_2_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_2_relu', data=a1_scale, act_type='relu')

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

    # --------------7x1,1x7,7x1,1x7---------------
    a3_reduce = mx.symbol.Convolution(name=name + '_7x1_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    a3_reduce_bn = mx.symbol.BatchNorm(name=name + '_7x1_reduce_bn', data=a3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a3_reduce_scale = a3_reduce_bn
    a3_reduce_relu = mx.symbol.Activation(name=name + '_7x1_reduce_relu', data=a3_reduce_scale, act_type='relu')

    a3_1 = mx.symbol.Convolution(name=name + '_7x1_2', data=a3_reduce_relu, num_filter=num3_2, pad=(3, 0),
                                 kernel=(7, 1), stride=(1, 1),
                                 no_bias=True)
    a3_1_bn = mx.symbol.BatchNorm(name=name + '_7x1_2_bn', data=a3_1, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_1_scale = a3_1_bn
    a3_1_relu = mx.symbol.Activation(name=name + '_7x1_2_relu', data=a3_1_scale, act_type='relu')

    a3_2 = mx.symbol.Convolution(name=name + '_1x7_2', data=a3_1_relu, num_filter=num3_3, pad=(0, 3), kernel=(1, 7),
                                 stride=(1, 1),
                                 no_bias=True)
    a3_2_bn = mx.symbol.BatchNorm(name=name + '_1x7_2_bn', data=a3_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_2_scale = a3_2_bn
    a3_2_relu = mx.symbol.Activation(name=name + '_1x7_2_relu', data=a3_2_scale, act_type='relu')

    a3_3 = mx.symbol.Convolution(name=name + '_7x1_3', data=a3_2_relu, num_filter=num3_4, pad=(3, 0), kernel=(7, 1),
                                 stride=(1, 1),
                                 no_bias=True)
    a3_3_bn = mx.symbol.BatchNorm(name=name + '_7x1_3_bn', data=a3_3, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_3_scale = a3_3_bn
    a3_3_relu = mx.symbol.Activation(name=name + '_7x1_3_relu', data=a3_3_scale, act_type='relu')

    a3_4 = mx.symbol.Convolution(name=name + '_1x7_3', data=a3_3_relu, num_filter=num3_5, pad=(0, 3), kernel=(1, 7),
                                 stride=(1, 1),
                                 no_bias=True)
    a3_4_bn = mx.symbol.BatchNorm(name=name + '_1x7_3_bn', data=a3_4, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_4_scale = a3_4_bn
    a3_4_relu = mx.symbol.Activation(name=name + '_1x7_3_relu', data=a3_4_scale, act_type='relu')

    # --------------pool---------------
    a4 = mx.symbol.Pooling(name=name + '_pool_ave', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                           pool_type='avg',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    a4_proj = mx.symbol.Convolution(name=name + '_1x1', data=a4, num_filter=num4_1, pad=(0, 0), kernel=(1, 1),
                                    stride=(1, 1),
                                    no_bias=True)
    a4_proj_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a4_proj, use_global_stats=use_global_stats,
                                     fix_gamma=False, eps=0.001)
    a4_proj_scale = a4_proj_bn
    a4_proj_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a4_proj_scale, act_type='relu')

    inception_output = mx.symbol.Concat(*[a1_relu, a2_2_relu, a3_4_relu, a4_proj_relu], name=name + '_concat')

    return inception_output


def InceptionC(data,
               num1_1,
               num2_1, num2_2, num2_3,
               num3_1, num3_2, num3_3, num3_4,
               num4_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1_2', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_2_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_2_relu', data=a1_scale, act_type='relu')

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

    a2_2 = mx.symbol.Convolution(name=name + '_3x1', data=a2_reduce_relu, num_filter=num2_3, pad=(1, 0), kernel=(3, 1),
                                 stride=(1, 1), no_bias=True)
    a2_2_bn = mx.symbol.BatchNorm(name=name + '_3x1_bn', data=a2_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a2_2_scale = a2_2_bn
    a2_2_relu = mx.symbol.Activation(name=name + '_3x1_relu', data=a2_2_scale, act_type='relu')

    # --------------3x3,(1x3,3x1,parallel)---------------
    a3_reduce = mx.symbol.Convolution(name=name + '_3x3_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a3_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_reduce_bn', data=a3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a3_reduce_scale = a3_reduce_bn
    a3_reduce_relu = mx.symbol.Activation(name=name + '_3x3_reduce_relu', data=a3_reduce_scale, act_type='relu')

    a3_1 = mx.symbol.Convolution(name=name + '_3x3', data=a3_reduce_relu, num_filter=num3_2, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    a3_1_bn = mx.symbol.BatchNorm(name=name + '_3x3_bn', data=a3_1, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_1_scale = a3_1_bn
    a3_1_relu = mx.symbol.Activation(name=name + '_3x3_relu', data=a3_1_scale, act_type='relu')

    a3_2 = mx.symbol.Convolution(name=name + '_1x3_2', data=a3_1_relu, num_filter=num3_3, pad=(0, 1), kernel=(1, 3),
                                 stride=(1, 1), no_bias=True)
    a3_2_bn = mx.symbol.BatchNorm(name=name + '_1x3_2_bn', data=a3_2, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_2_scale = a3_2_bn
    a3_2_relu = mx.symbol.Activation(name=name + '_1x3_2_relu', data=a3_2_scale, act_type='relu')

    a3_3 = mx.symbol.Convolution(name=name + '_3x1_2', data=a3_1_relu, num_filter=num3_4, pad=(1, 0), kernel=(3, 1),
                                 stride=(1, 1), no_bias=True)
    a3_3_bn = mx.symbol.BatchNorm(name=name + '_3x1_2_bn', data=a3_3, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_3_scale = a3_3_bn
    a3_3_relu = mx.symbol.Activation(name=name + '_3x1_2_relu', data=a3_3_scale, act_type='relu')

    # --------------pool---------------
    a4 = mx.symbol.Pooling(name=name + '_pool_ave', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                           pool_type='avg', pooling_convention='full', cudnn_off=False, global_pool=False)

    a4_proj = mx.symbol.Convolution(name=name + '_1x1', data=a4, num_filter=num4_1, pad=(0, 0), kernel=(1, 1),
                                    stride=(1, 1),
                                    no_bias=True)
    a4_proj_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a4_proj, use_global_stats=use_global_stats,
                                     fix_gamma=False, eps=0.001)
    a4_proj_scale = a4_proj_bn
    a4_proj_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a4_proj_scale, act_type='relu')

    inception_output = mx.symbol.Concat(*[a1_relu, a2_relu, a2_2_relu, a3_2_relu, a3_3_relu, a4_proj_relu], name=name + '_concat')

    return inception_output


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

    return reduction_output


def ReductionB(data,
               num1_1, num1_2,
               num2_1, num2_2, num2_3, num2_4,
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

    # ---------------1x7,7x1--------------
    r2_reduce = mx.symbol.Convolution(name=name + '_1x7_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1),
                                      no_bias=True)
    r2_reduce_bn = mx.symbol.BatchNorm(name=name + '_1x7_reduce_bn', data=r2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    r2_reduce_scale = r2_reduce_bn
    r2_reduce_relu = mx.symbol.Activation(name=name + '_1x7_reduce_relu', data=r2_reduce_scale, act_type='relu')

    r2 = mx.symbol.Convolution(name=name + '_1x7', data=r2_reduce_relu, num_filter=num2_2, pad=(0, 3), kernel=(1, 7),
                               stride=(1, 1),
                               no_bias=True)
    r2_bn = mx.symbol.BatchNorm(name=name + '_1x7_bn', data=r2, use_global_stats=use_global_stats, fix_gamma=False,
                                eps=0.001)
    r2_scale = r2_bn
    r2_relu = mx.symbol.Activation(name=name + '_1x7_relu', data=r2_scale, act_type='relu')

    r2_2 = mx.symbol.Convolution(name=name + '_7x1', data=r2_relu, num_filter=num2_3, pad=(3, 0), kernel=(7, 1),
                                 stride=(1, 1),
                                 no_bias=True)
    r2_2_bn = mx.symbol.BatchNorm(name=name + '_7x1_bn', data=r2_2, use_global_stats=use_global_stats, fix_gamma=False,
                                  eps=0.001)
    r2_2_scale = r2_2_bn
    r2_2_relu = mx.symbol.Activation(name=name + '_7x1_relu', data=r2_2_scale, act_type='relu')

    r2_3 = mx.symbol.Convolution(name=name + '_3x3_2', data=r2_2_relu, num_filter=num2_4, pad=(0, 0), kernel=(3, 3),
                                 stride=(2, 2),
                                 no_bias=True)
    r2_3_bn = mx.symbol.BatchNorm(name=name + '_3x3_2_bn', data=r2_3, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    r2_3_scale = r2_3_bn
    r2_3_relu = mx.symbol.Activation(name=name + '_3x3_2_relu', data=r2_3_scale, act_type='relu')

    # pool
    r3 = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    reduction_output = mx.symbol.Concat(*[r1_relu, r2_3_relu, r3], name=name + '_concat')

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
    stem = inception_stem(data, num1_1, num1_2, num1_3, num2_1, num2_2, use_global_stats)

    # 3*InceptionA
    num1_1 = 64
    num2_1, num2_2 = (48, 64)
    num3_1, num3_2, num3_3 = (64, 96, 96)
    num4_1 = 32
    name = 'inception_a1'
    inception_a1 = InceptionA(stem, num1_1, num2_1, num2_2, num3_1, num3_2, num3_3, num4_1, name, use_global_stats)

    num4_1 = 64
    name = 'inception_a2'
    inception_a2 = InceptionA(inception_a1, num1_1, num2_1, num2_2, num3_1, num3_2, num3_3, num4_1, name,
                              use_global_stats)

    name = 'inception_a3'
    inception_a3 = InceptionA(inception_a2, num1_1, num2_1, num2_2, num3_1, num3_2, num3_3, num4_1, name,
                              use_global_stats)

    # ReductionA
    num1_1 = 384
    num2_1, num2_2, num2_3 = (64, 96, 96)
    name = 'reduction_a'
    reduction_a = ReductionA(inception_a3, num1_1, num2_1, num2_2, num2_3, name, use_global_stats)

    # 4*InceptionB
    num1_1 = 192
    num2_1, num2_2, num2_3 = (128, 128, 192)
    num3_1, num3_2, num3_3, num3_4, num3_5 = (128, 128, 128, 128, 192)
    num4_1 = 192
    name = 'inception_b1'
    inception_b1 = InceptionB(reduction_a, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num3_5,
                              num4_1, name, use_global_stats)

    num2_1, num2_2, num2_3 = (160, 160, 192)
    num3_1, num3_2, num3_3, num3_4, num3_5 = (160, 160, 160, 160, 192)
    name = 'inception_b2'
    inception_b2 = InceptionB(inception_b1, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num3_5,
                              num4_1, name, use_global_stats)

    name = 'inception_b3'
    inception_b3 = InceptionB(inception_b2, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num3_5,
                              num4_1, name, use_global_stats)

    num2_1, num2_2, num2_3 = (192, 192, 192)
    num3_1, num3_2, num3_3, num3_4, num3_5 = (192, 192, 192, 192, 192)
    name = 'inception_b4'
    inception_b4 = InceptionB(inception_b3, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num3_5,
                              num4_1, name, use_global_stats)

    # ReductionB
    num1_1, num1_2 = (192, 320)
    num2_1, num2_2, num2_3, num2_4 = (192, 192, 192, 192)
    name = 'reduction_b'
    reduction_b = ReductionB(inception_b4, num1_1, num1_2, num2_1, num2_2, num2_3, num2_4, name, use_global_stats)

    # 2*InceptionC
    num1_1 = 320
    num2_1, num2_2, num2_3 = (384, 384, 384)
    num3_1, num3_2, num3_3, num3_4 = (448, 384, 384, 384)
    num4_1 = 192
    name = 'inception_c1'
    inception_c1 = InceptionC(reduction_b, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num4_1, name, use_global_stats)

    name = 'inception_c2'
    inception_c2 = InceptionC(inception_c1, num1_1, num2_1, num2_2, num2_3, num3_1, num3_2, num3_3, num3_4, num4_1, name, use_global_stats)

    # classifier
    pool_8x8_s1 = mx.symbol.Pooling(name='pool_8x8_s1', data=inception_c2, kernel=(1, 1), pool_type='avg',
                           pooling_convention='full', cudnn_off=False, global_pool=True)
    pool_8x8_s1_drop = mx.symbol.Dropout(data=pool_8x8_s1, p=0.2)
    flatten_0 = mx.symbol.Flatten(data=pool_8x8_s1_drop, name="flatten_0")
    classifier = mx.symbol.FullyConnected(data=flatten_0, num_hidden=num_classes, name='classifier')
    prob = mx.symbol.SoftmaxOutput(data=classifier, name='prob')

    return prob

if __name__ == '__main__':
    net = get_symbol(1000)
    # shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
    net.save('inception-v3-symbol.json')


