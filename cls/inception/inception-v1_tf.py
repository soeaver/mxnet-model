import mxnet as mx


def inception_stem(data,
                   num1_1,
                   num2_1, num2_2,
                   use_global_stats
                   ):
    conv1_7x7_s2 = mx.symbol.Convolution(name='conv1_7x7_s2', data=data, num_filter=num1_1, pad=(3, 3), kernel=(7, 7),
                                         stride=(2, 2), no_bias=True)
    conv1_7x7_s2_bn = mx.symbol.BatchNorm(name='conv1_7x7_s2_bn', data=conv1_7x7_s2, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv1_7x7_s2_scale = conv1_7x7_s2_bn
    conv1_7x7_s2_relu = mx.symbol.Activation(name='conv1_7x7_s2_relu', data=conv1_7x7_s2_scale, act_type='relu')

    ########## pool1 #########
    pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv1_7x7_s2_relu, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)

    conv2_3x3_reduce = mx.symbol.Convolution(name='conv2_3x3_reduce', data=pool1_3x3_s2, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv2_3x3_reduce_bn = mx.symbol.BatchNorm(name='conv2_3x3_reduce_bn', data=conv2_3x3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    conv2_3x3_reduce_scale = conv2_3x3_reduce_bn
    conv2_3x3_reduce_relu = mx.symbol.Activation(name='conv2_3x3_reduce_relu', data=conv2_3x3_reduce_scale, act_type='relu')

    conv2_3x3 = mx.symbol.Convolution(name='conv2_3x3', data=conv2_3x3_reduce_relu, num_filter=num2_2, pad=(1, 1),
                                         kernel=(3, 3), stride=(1, 1), no_bias=True)
    conv2_3x3_bn = mx.symbol.BatchNorm(name='conv2_3x3_bn', data=conv2_3x3, use_global_stats=use_global_stats,
                                          fix_gamma=False, eps=0.001)
    conv2_3x3_scale = conv2_3x3_bn
    conv2_3x3_relu = mx.symbol.Activation(name='conv2_3x3_relu', data=conv2_3x3_scale, act_type='relu')

    ########## pool2 #########
    pool2_3x3_s2 = mx.symbol.Pooling(name='pool2_3x3_s2', data=conv2_3x3_relu, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)

    return pool2_3x3_s2


def Inception(data,
               num1_1,
               num2_1, num2_2,
               num3_1, num3_2,
               num4_1,
               name, use_global_stats):
    # --------------1x1---------------
    a1 = mx.symbol.Convolution(name=name + '_1x1', data=data, num_filter=num1_1, pad=(0, 0), kernel=(1, 1),
                               stride=(1, 1), no_bias=True)
    a1_bn = mx.symbol.BatchNorm(name=name + '_1x1_bn', data=a1, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a1_scale = a1_bn
    a1_relu = mx.symbol.Activation(name=name + '_1x1_relu', data=a1_scale, act_type='relu')

    # --------------3x3-a---------------
    a2_reduce = mx.symbol.Convolution(name=name + '_3x3_a_reduce', data=data, num_filter=num2_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a2_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_a_reduce_bn', data=a2_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a2_reduce_scale = a2_reduce_bn
    a2_reduce_relu = mx.symbol.Activation(name=name + '_3x3_a_reduce_relu', data=a2_reduce_scale, act_type='relu')

    a2 = mx.symbol.Convolution(name=name + '_3x3_a', data=a2_reduce_relu, num_filter=num2_2, pad=(1, 1), kernel=(3, 3),
                               stride=(1, 1), no_bias=True)
    a2_bn = mx.symbol.BatchNorm(name=name + '_3x3_a_bn', data=a2, use_global_stats=use_global_stats,
                                fix_gamma=False, eps=0.001)
    a2_scale = a2_bn
    a2_relu = mx.symbol.Activation(name=name + '_3x3_a_relu', data=a2_scale, act_type='relu')

    # --------------3x3-b---------------
    a3_reduce = mx.symbol.Convolution(name=name + '_3x3_b_reduce', data=data, num_filter=num3_1, pad=(0, 0),
                                      kernel=(1, 1), stride=(1, 1), no_bias=True)
    a3_reduce_bn = mx.symbol.BatchNorm(name=name + '_3x3_b_reduce_bn', data=a3_reduce, use_global_stats=use_global_stats,
                                       fix_gamma=False, eps=0.001)
    a3_reduce_scale = a3_reduce_bn
    a3_reduce_relu = mx.symbol.Activation(name=name + '_3x3_b_reduce_relu', data=a3_reduce_scale, act_type='relu')

    a3= mx.symbol.Convolution(name=name + '_3x3_b', data=a3_reduce_relu, num_filter=num3_2, pad=(1, 1),
                                 kernel=(3, 3), stride=(1, 1), no_bias=True)
    a3_bn = mx.symbol.BatchNorm(name=name + '_3x3_b_bn', data=a3, use_global_stats=use_global_stats,
                                  fix_gamma=False, eps=0.001)
    a3_scale = a3_bn
    a3_relu = mx.symbol.Activation(name=name + '_3x3_b_relu', data=a3_scale, act_type='relu')

    # --------------pool---------------
    a4 = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), pool_type='max',
                           pooling_convention='full', cudnn_off=False, global_pool=False)

    a4_proj = mx.symbol.Convolution(name=name + '_pool_proj', data=a4, num_filter=num4_1, pad=(0, 0), kernel=(1, 1),
                                    stride=(1, 1),
                                    no_bias=True)
    a4_proj_bn = mx.symbol.BatchNorm(name=name + '_pool_proj_bn', data=a4_proj, use_global_stats=use_global_stats,
                                     fix_gamma=False, eps=0.001)
    a4_proj_scale = a4_proj_bn
    a4_proj_relu = mx.symbol.Activation(name=name + '_pool_proj_relu', data=a4_proj_scale, act_type='relu')

    inception_output = mx.symbol.Concat(*[a1_relu, a2_relu, a3_relu, a4_proj_relu], name=name + '_output')

    return inception_output


def get_symbol(num_classes):
    use_global_stats = True
    data = mx.symbol.Variable(name='data')
    num1_1 = 64
    num2_1, num2_2 = (64, 192)
    stem = inception_stem(data, num1_1, num2_1, num2_2, use_global_stats)

    # 2*InceptionA
    num1_1 = 64
    num2_1, num2_2 = (96, 128)
    num3_1, num3_2 = (16, 32)
    num4_1 = 32
    name = 'inception_a1'
    inception_a1 = Inception(stem, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 128
    num2_1, num2_2 = (128, 192)
    num3_1, num3_2 = (32, 96)
    num4_1 = 64
    name = 'inception_a2'
    inception_a2 = Inception(inception_a1, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    # pool
    pool3_3x3_s2 = mx.symbol.Pooling(name='pool3_3x3_s2', data=inception_a2, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)
    # 5*InceptionB
    num1_1 = 192
    num2_1, num2_2 = (96, 208)
    num3_1, num3_2 = (16, 48)
    num4_1 = 64
    name = 'inception_b1'
    inception_b1 = Inception(pool3_3x3_s2, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 160
    num2_1, num2_2 = (112, 224)
    num3_1, num3_2 = (24, 64)
    name = 'inception_b2'
    inception_b2 = Inception(inception_b1, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 128
    num2_1, num2_2 = (128, 256)
    num3_1, num3_2 = (24, 64)
    name = 'inception_b3'
    inception_b3 = Inception(inception_b2, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 112
    num2_1, num2_2 = (144, 288)
    num3_1, num3_2 = (32, 64)
    name = 'inception_b4'
    inception_b4 = Inception(inception_b3, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 256
    num2_1, num2_2 = (160, 320)
    num3_1, num3_2 = (32, 128)
    num4_1 = 128
    name = 'inception_b5'
    inception_b5 = Inception(inception_b4, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    # pool
    pool4_3x3_s2 = mx.symbol.Pooling(name='pool4_3x3_s2', data=inception_b5, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                     pool_type='max', pooling_convention='full', cudnn_off=False, global_pool=False)

    # 2InceptionC
    num1_1 = 256
    num2_1, num2_2 = (160, 320)
    num3_1, num3_2 = (32, 128)
    num4_1 = 128
    name = 'inception_c1'
    inception_c1 = Inception(pool4_3x3_s2, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    num1_1 = 384
    num2_1, num2_2 = (192, 384)
    num3_1, num3_2 = (48, 128)
    num4_1 = 128
    name = 'inception_c2'
    inception_c2 = Inception(inception_c1, num1_1, num2_1, num2_2, num3_1, num3_2, num4_1, name, use_global_stats)

    # classifier
    pool_7x7_s1 = mx.symbol.Pooling(name='pool_7x7_s1', data=inception_c2, kernel=(1, 1), pool_type='avg',
                           pooling_convention='full', cudnn_off=False, global_pool=True)
    pool_7x7_s1_drop = mx.symbol.Dropout(data=pool_7x7_s1, p=0.2)
    classifier = mx.symbol.Convolution(name='classifier', data=pool_7x7_s1_drop, num_filter=num_classes, pad=(0, 0), kernel=(1, 1),
                                         stride=(1, 1), no_bias=False)
    reshape = mx.symbol.Reshape(data=classifier, shape=(0, 0))
    prob = mx.symbol.SoftmaxOutput(data=reshape, name='prob')

    return prob


if __name__ == '__main__':
    net = get_symbol(1000)
    # shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
    net.save('inception-v1_tf-symbol.json')
