import numpy as np
import mxnet as mx
import cv2
import datetime
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])
prefix = 'resnet18-priv'

gpu_mode = True
gpu_id = 0
data_root = '~/Database/ILSVRC2012'
val_file = 'ILSVRC2012_val.txt'
save_log = 'log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
class_num = 1000
base_size = 256
crop_size = 224
color_scale = 1.0
mean_value = np.array([123.675, 116.28, 103.52])  # RGB
std = np.array([58.395, 57.12, 57.375])  # RGB
crop_num = 1  # 1 and others for center(single)-crop, 12 for mirror(12)-crop, 144 for multi(144)-crop
batch_size = 1
top_k = (1, 5)
save_score_map = False

sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
if gou_mode:
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(gpu_id), label_names=None)
else:
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

#### if model is trained with BGR input channels, swap conv1 layer parameter channels ####
# conv1 = net.params['conv1'][0].data.copy()
# conv1[:, [0, 2], :, :] = conv1[:, [2, 0], :, :]
# net.params['conv1'][0].data[...] = conv1
# net.save('model_path/modelname.caffemodel')

def eval_batch():
    eval_images = []
    ground_truth = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip().split(' ')[0])
        ground_truth.append(int(i.strip().split(' ')[1]))
    f.close()

    skip_num = 0
    eval_len = len(eval_images)
    # eval_len = 1000
    accuracy = np.zeros(len(top_k))
    if save_score_map:
        all_score_map = np.zeros((eval_len - skip_num, class_num), dtype=np.float32)
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(data_root + eval_images[i + skip_num], 1)

        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _img = cv2.resize(_img, (int(_img.shape[1] * base_size / min(_img.shape[:2])),
                                 int(_img.shape[0] * base_size / min(_img.shape[:2])))
                          )
        _img = image_preprocess(_img)
        _img = _img * color_scale

        score_vec = np.zeros(class_num, dtype=np.float32)
        crops = []
        if crop_num == 1:
            crops.append(center_crop(_img))
        elif crop_num == 12:
            crops.extend(mirror_crop(_img))
        elif crop_num == 144:
            crops.extend(multi_crop(_img))
        else:
            crops.append(center_crop(_img))

        iter_num = int(len(crops) / batch_size)
        timer_pt1 = datetime.datetime.now()
        for j in xrange(iter_num):
            score_vec += mxnet_process(np.asarray(crops, dtype=np.float32)[j*batch_size:(j+1)*batch_size])
        score_index = np.argsort(np.squeeze(score_vec))[::-1]
        timer_pt2 = datetime.datetime.now()

        if save_score_map:
            all_score_map[i] = score_vec / len(crops)

        print 'Testing image: {}/{} {} {}/{} {}s' \
            .format(str(i + 1), str(eval_len - skip_num), str(eval_images[i + skip_num]),
                    str(score_index[0]), str(ground_truth[i + skip_num]),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds)),

        for j in xrange(len(top_k)):
            if ground_truth[i + skip_num] in score_index[:top_k[j]]:
                accuracy[j] += 1
            tmp_acc = float(accuracy[j]) / float(i + 1)
            if top_k[j] == 1:
                print '\ttop_' + str(top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(top_k[j]) + ':' + str(tmp_acc)
    end_time = datetime.datetime.now()

    if save_score_map:
        np.savetxt(save_log.replace('log', 'scoremap'), all_score_map, fmt='%6f')
    w = open(save_log, 'w')
    s1 = 'Evaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    s2 = '\nThe model is: {}. \nThe val file is: {}. \n{} images has been tested, crop_num is: {}, base_size is: {}, ' \
         'crop_size is: {}.'.format(prefix, val_file, str(eval_len), str(crop_num), str(base_size), str(crop_size))
    s3 = '\nThe mean value is: ({}, {}, {}), std is : ({}, {}, {}).'\
        .format(str(mean_value[0]), str(mean_value[1]), str(mean_value[2]), str(std[0]), str(std[1]), str(std[2]))
    s4 = ''
    for i in xrange(len(top_k)):
        _acc = float(accuracy[i]) / float(eval_len)
        s4 += '\nAccuracy of top_{} is: {}; correct num is {}.'.format(str(top_k[i]), str(_acc), str(int(accuracy[i])))
    print s1, s2, s3, s4
    w.write(s1 + s2 + s3 + s4)
    w.close()


def image_preprocess(img):
    c1, c2, c3 = cv2.split(img)
    return cv2.merge([(c1-mean_value[0])/std[0], (c2-mean_value[1])/std[1], (c3-mean_value[2])/std[2]])


def center_crop(img):  # single crop
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]


def over_sample(img):  # 12 crops of image
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    sample_list = [img[:crop_size, :crop_size], img[-crop_size:, -crop_size:], img[:crop_size, -crop_size:],
                   img[-crop_size:, :crop_size], img[yy: yy + crop_size, xx: xx + crop_size],
                   cv2.resize(img, (crop_size, crop_size))]
    return sample_list


def mirror_crop(img):  # 12*len(size_list) crops
    crop_list = []
    img_resize = cv2.resize(img, (base_size, base_size))
    mirror = img[:, ::-1]
    crop_list.extend(over_sample(img))
    crop_list.extend(over_sample(mirror))
    return crop_list


def multi_crop(img):  # 144(12*12) crops
    crop_list = []
    # size_list = [256, 288, 320, 352]  # crop_size: 224
    # size_list = [270, 300, 330, 360]  # crop_size: 235
    # size_list = [320, 352, 384, 416]  # crop_size: 299
    # size_list = [352, 384, 416, 448]  # crop_size: 320
    size_list = [395, 427, 459, 491]
    short_edge = min(img.shape[:2])
    for i in size_list:
        img_resize = cv2.resize(img, (img.shape[1] * i / short_edge, img.shape[0] * i / short_edge))
        yy = int((img_resize.shape[0] - i) / 2)
        xx = int((img_resize.shape[1] - i) / 2)
        for j in xrange(3):
            left_center_right = img_resize[yy * j: yy * j + i, xx * j: xx * j + i]
            mirror = left_center_right[:, ::-1]
            crop_list.extend(over_sample(left_center_right))
            crop_list.extend(over_sample(mirror))
    return crop_list

def mxnet_process(_input):
    _input = _input.transpose(0, 3, 1, 2)
    mod.forward(Batch([mx.nd.array(_input)]))
    prob = mod.get_outputs()[0].asnumpy()

    return np.sum(prob, axis=0)

if __name__ == '__main__':
    eval_batch()
