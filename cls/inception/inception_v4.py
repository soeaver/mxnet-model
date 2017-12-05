import sys
sys.path.append('/home/priv-lab1/workspace/mxnet')
import mxnet as mx



def Conv(last_layer, name, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0)):
    conv = mx.symbol.Convolution(name=name, data=last_layer, num_filter=num_filter, pad=pad,
                                 kernel=kernel,
                                 stride=stride, no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, use_global_stats=True,
                                  fix_gamma=False,
                                  eps=0.001)
    conv_scale = conv_bn
    conv_relu = mx.symbol.Activation(name=name + '_relu', data=conv_scale, act_type='relu')
    return conv_relu


def inception_stem(data, name, num_stem1, num_stem2_input, num_stem2_output, num_stem3):
    stem1_3x3 = Conv(data, name + '1_3x3_s2', num_stem1, (3, 3), (2, 2), (0, 0))
    stem1_pool = mx.symbol.Pooling(name=name + '1_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                   pool_type='max',
                                   pooling_convention='valid', cudnn_off=False, global_pool=False)
    stem1_concat = mx.sym.Concat(*[stem1_pool, stem1_3x3], name=('%s1' % name))

    stem2_3x3_reduce = Conv(stem1_concat, name + '2_3x3_reduce', num_stem2_input, (1, 1), (1, 1), (0, 0))
    stem2_3x3 = Conv(stem2_3x3_reduce, name + '2_3x3', num_stem2_output, (3, 3), (1, 1), (0, 0))
    stem2_1x7_reduce = Conv(stem1_concat, name + '2_1x7_reduce', num_stem2_input, (1, 1), (1, 1), (0, 0))
    stem2_1x7 = Conv(stem2_1x7_reduce, name + '2_1x7', num_stem2_input, (1, 7), (1, 1), (0, 3))
    stem2_7x1 = Conv(stem2_1x7, name + '2_7x1', num_stem2_input, (7, 1), (1, 1), (3, 0))
    stem2_3x3_2 = Conv(stem2_7x1, name + '2_3x3_2', num_stem2_output, (3, 3), (1, 1), (0, 0))
    stem2_concat = mx.sym.Concat(*[stem2_3x3_2, stem2_3x3], name=('%s2' % name))

    stem3_3x3 = Conv(stem2_concat, name + '3_3x3_s2', num_stem3, (3, 3), (2, 2), (0, 0))
    stem3_pool = mx.symbol.Pooling(name=name + '3_pool', data=stem2_concat, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                                   pool_type='max',
                                   pooling_convention='valid', cudnn_off=False, global_pool=False)
    stem3_concat = mx.sym.Concat(*[stem3_3x3, stem3_pool], name=('%s3' % name))

    return stem3_concat


def InceptionA(data, name, num_input, num_1x1_2, num_1x1):
    a_1x1_2 = Conv(data, name + '_1x1_2', num_1x1_2, (1, 1), (1, 1), (0, 0))

    a_3x3_reduce = Conv(data, name + '_3x3_reduce', num_input, (1, 1), (1, 1), (0, 0))
    a_3x3 = Conv(a_3x3_reduce, name + '_3x3', num_input + 32, (3, 3), (1, 1), (1, 1))

    a_3x3_2_reduce = Conv(data, name + '_3x3_2_reduce', num_input, (1, 1), (1, 1), (0, 0))
    a_3x3_2 = Conv(a_3x3_2_reduce, name + '_3x3_2', num_input + 32, (3, 3), (1, 1), (1, 1))
    a_3x3_3 = Conv(a_3x3_2, name + '_3x3_3', num_input + 32, (3, 3), (1, 1), (1, 1))

    a_avg_pool = mx.symbol.Pooling(name=name + '_avg_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   pool_type='avg')
    a_1x1 = Conv(a_avg_pool, name + '_1x1', num_1x1, (1, 1), (1, 1), (0, 0))

    a_concat = mx.sym.Concat(*[a_1x1_2, a_3x3, a_3x3_3, a_1x1], name=('%s_concat' % name))

    return a_concat


def InceptionB(data, name, num_input, num_1x1_2, num_1x1):
    b_1x1_2 = Conv(data, name + '_1x1_2', num_1x1_2, (1, 1), (1, 1), (0, 0))

    b_1x7_reduce = Conv(data, name + '_1x7_reduce', num_input, (1, 1), (1, 1), (0, 0))
    b_1x7 = Conv(b_1x7_reduce, name + '_1x7', num_input + 32, (1, 7), (1, 1), (0, 3))
    b_7x1 = Conv(b_1x7, name + '_7x1', num_input + 64, (7, 1), (1, 1), (3, 0))

    b_7x1_2_reduce = Conv(data, name + '_7x1_2_reduce', num_input, (1, 1), (1, 1), (0, 0))
    b_7x1_2 = Conv(b_7x1_2_reduce, name + '_7x1_2', num_input, (7, 1), (1, 1), (3, 0))
    b_1x7_2 = Conv(b_7x1_2, name + '_1x7_2', num_input + 32, (1, 7), (1, 1), (0, 3))
    b_7x1_3 = Conv(b_1x7_2, name + '_7x1_3', num_input + 32, (7, 1), (1, 1), (3, 0))
    b_1x7_3 = Conv(b_7x1_3, name + '_1x7_3', num_input + 64, (1, 7), (1, 1), (0, 3))

    b_avg_pool = mx.symbol.Pooling(name=name + '_avg_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   pool_type='avg')
    b_1x1 = Conv(b_avg_pool, name + '_1x1', num_1x1, (1, 1), (1, 1), (0, 0))

    b_concat = mx.sym.Concat(*[b_1x1_2, b_7x1, b_1x7_3, b_1x1], name=('%s_concat' % name))
    return b_concat


def InceptionC(data, name, num_input, num_output, num_1x1_2, num_1x1):
    c_1x1_2 = Conv(data, name + '_1x1_2', num_1x1_2, (1, 1), (1, 1), (0, 0))

    c_1x1_3 = Conv(data, name + '_1x1_3', num_input, (1, 1), (1, 1), (0, 0))
    c_1x3 = Conv(c_1x1_3, name + '_1x3', num_output, (1, 3), (1, 1), (0, 1))
    c_3x1 = Conv(c_1x1_3, name + '_3x1', num_output, (3, 1), (1, 1), (1, 0))

    c_1x1_4 = Conv(data, name + '_1x1_4', num_input, (1, 1), (1, 1), (0, 0))
    c_3x1_2 = Conv(c_1x1_4, name + '_3x1_2', num_input + 64, (3, 1), (1, 1), (1, 0))
    c_1x3_2 = Conv(c_3x1_2, name + '_1x3_2', num_input + 128, (1, 3), (1, 1), (0, 1))
    c_1x3_3 = Conv(c_1x3_2, name + '_1x3_3', num_output, (1, 3), (1, 1), (0, 1))
    c_3x1_3 = Conv(c_1x3_2, name + '_3x1_3', num_output, (3, 1), (1, 1), (1, 0))

    c_avg_pool = mx.symbol.Pooling(name=name + '_avg_pool', data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   pool_type='avg')
    c_1x1 = Conv(c_avg_pool, name + '_1x1', num_1x1, (1, 1), (1, 1), (0, 0))

    c_concat = mx.sym.Concat(*[c_1x1_2, c_1x3, c_3x1, c_1x3_3, c_3x1_3, c_1x1], name=(name + '_concat'))

    return c_concat


def ReductionA(data, name, num_input, num_3x3):
    a_3x3 = Conv(data, name + '_3x3', num_3x3, (3, 3), (2, 2), (0, 0))

    a_3x3_2_reduce = Conv(data, name + '_3x3_2_reduce', num_input, (1, 1), (1, 1), (0, 0))
    a_3x3_2 = Conv(a_3x3_2_reduce, name + '_3x3_2', num_input + 32, (3, 3), (1, 1), (1, 1))
    a_3x3_3 = Conv(a_3x3_2, name + '_3x3_3', num_input + 64, (3, 3), (2, 2), (0, 0))

    a_pool = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                               pool_type='max',
                               pooling_convention='valid', cudnn_off=False, global_pool=False)

    a_concat = mx.sym.Concat(*[a_3x3, a_3x3_3, a_pool], name=('%s_concat' % name))

    return a_concat


def ReductionB(data, name, num_1x7, num_3x3, num_7x1):
    b_3x3_reduce = Conv(data, name + '_3x3_reduce', num_3x3, (1, 1), (1, 1), (0, 0))
    b_3x3 = Conv(b_3x3_reduce, name + '_3x3', num_3x3, (3, 3), (2, 2), (0, 0))

    b_1x7_reduce = Conv(data, name + '_1x7_reduce', num_1x7, (1, 1), (1, 1), (0, 0))
    b_1x7 = Conv(b_1x7_reduce, name + '_1x7', num_1x7, (1, 7), (1, 1), (0, 3))
    b_7x1 = Conv(b_1x7, name + '_7x1', num_7x1, (7, 1), (1, 1), (3, 0))
    b_3x3_2 = Conv(b_7x1, name + '_3x3_2', num_7x1, (3, 3), (2, 2), (0, 0))

    b_pool = mx.symbol.Pooling(name=name + '_pool', data=data, pad=(0, 0), kernel=(3, 3), stride=(2, 2),
                               pool_type='max',
                               pooling_convention='valid', cudnn_off=False, global_pool=False)

    b_concat = mx.sym.Concat(*[b_3x3, b_3x3_2, b_pool], name=('%s_concat' % name))
    return b_concat


def stageA(num_stage, data, name, num_input, num_1x1_2, num_1x1):
    incep_a = data
    for i in xrange(num_stage):
        incep_a = InceptionA(incep_a, name + str(i + 1), num_input, num_1x1_2, num_1x1)

    return incep_a


def stageB(num_stage, data, name, num_input, num_1x1_2, num_1x1):
    incep_b = data
    for i in xrange(num_stage):
        incep_b = InceptionB(incep_b, name + str(i + 1), num_input, num_1x1_2, num_1x1)

    return incep_b


def stageC(num_stage, data, name, num_input, num_output, num_1x1_2, num_1x1):
    incep_c = data
    for i in xrange(num_stage):
        incep_c = InceptionC(incep_c, name + str(i + 1), num_input, num_output, num_1x1_2, num_1x1)

    return incep_c


def get_symbol(num_classes=1000):
    data = mx.symbol.Variable(name='data')

    conv1 = Conv(data, 'conv1_3x3_s2', 32, (3, 3), (2, 2), (0, 0))
    conv2 = Conv(conv1, 'conv2_3x3_s1', 32, (3, 3), (1, 1), (0, 0))
    conv3 = Conv(conv2, 'conv3_3x3_s1', 64, (3, 3), (1, 1), (1, 1))

    stem = inception_stem(conv3, 'inception_stem', 96, 64, 96, 192)

    stage_a = stageA(4, stem, 'inception_a', 64, 96, 96)

    reduc_a = ReductionA(stage_a, 'reduction_a', 192, 384)

    stage_b = stageB(7, reduc_a, 'inception_b', 192, 384, 128)

    reduc_b = ReductionB(stage_b, 'reduction_b', 256, 192, 320)

    stage_c = stageC(3, reduc_b, 'inception_c', 384, 256, 256, 256)

    avg_pool = mx.symbol.Pooling(name='avg_pool', data=stage_c, kernel=(1, 1), pool_type='avg',
                                 global_pool=True,
                                 pooling_convention='full', cudnn_off=False)
    dropout = mx.sym.Dropout(data=avg_pool, p=0.2)
    flatten = mx.symbol.Flatten(name='flatten', data=dropout)
    classifier = mx.symbol.FullyConnected(name='classifier', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=classifier)

    return softmax

softmax = get_symbol(1000)
softmax.save('inception_v4-symbol.json')

