import numpy as np
import os
import scipy.misc


def crop_image(image_in, image_ref, rotation, bounding_box, padding=0.15, jitter=0.15, flip=True, image_size=224):
    # based on https://github.com/shubhtuls/drc/blob/master/utils/cropUtils.lua

    left, top, width_box, height_box = bounding_box
    height_image, width_image = image_in.shape[:2]

    # random cropping
    y_min = int(top + (np.random.uniform(-jitter, jitter) - padding) * height_box)
    y_max = int(top + height_box + (np.random.uniform(-jitter, jitter) + padding) * height_box)
    x_min = int(left + (np.random.uniform(-jitter, jitter) - padding) * width_box)
    x_max = int(left + width_box + (np.random.uniform(-jitter, jitter) + padding) * width_box)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(height_image, y_max)
    x_max = min(width_image, x_max)
    image_in = image_in[y_min:y_max, x_min:x_max]

    # random flipping
    if flip:
        if np.random.uniform(0, 1) < 0.5:
            image_in = image_in[:, ::-1]

    image_in = scipy.misc.imresize(image_in, (image_size, image_size)).astype('float32') / 255.
    image_in = image_in.transpose((2, 0, 1))
    return image_in, image_ref, rotation


class Pascal(object):
    def __init__(self, directory, class_ids, set_name):
        self.name = 'pascal'
        self.directory = directory
        self.class_ids = class_ids
        self.set_name = set_name
        self.image_size = 224

        self.images_original = {}
        self.images_ref = {}
        self.bounding_boxes = {}
        self.rotation_matrices = {}
        self.voxels = {}
        self.num_data = {}
        for class_id in class_ids:
            data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, set_name)), allow_pickle=True, encoding='latin1')
            self.images_original[class_id] = data['images']
            self.images_ref[class_id] = data['images_ref']
            self.bounding_boxes[class_id] = data['bounding_boxes']
            self.rotation_matrices[class_id] = data['rotation_matrices']
            self.voxels[class_id] = data['voxels']
            if set_name == 'train':
                # add ImageNet data
                data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, 'imagenet')), allow_pickle=True,  encoding='latin1')
                self.images_original[class_id] = np.concatenate(
                    (self.images_original[class_id], data['images']), axis=0)
                self.images_ref[class_id] = np.concatenate(
                    (self.images_ref[class_id], data['images_ref']), axis=0)
                self.bounding_boxes[class_id] = np.concatenate(
                    (self.bounding_boxes[class_id], data['bounding_boxes']), axis=0)
                self.rotation_matrices[class_id] = np.concatenate(
                    (self.rotation_matrices[class_id], data['rotation_matrices']), axis=0)
                self.voxels[class_id] = np.concatenate((self.voxels[class_id], data['voxels']), axis=0)
            self.images_ref[class_id] = self.images_ref[class_id].transpose((0, 3, 1, 2))
            self.num_data[class_id] = self.images_ref[class_id].shape[0]

    def get_random_batch(self, batch_size):
        labels = np.zeros(batch_size, 'int32')
        images_in = np.zeros((batch_size, 3, self.image_size, self.image_size), 'float32')
        images_ref = np.zeros((batch_size, 4, self.image_size, self.image_size), 'float32')
        rotation_matrices = np.zeros((batch_size, 3, 3), 'float32')
        rotation_matrices_random = np.zeros((batch_size, 3, 3), 'float32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_num = np.random.choice(range(self.num_data[class_id]))

            image_in = self.images_original[class_id][object_num]
            image_ref = self.images_ref[class_id][object_num].astype('float32') / 255.
            rotation = self.rotation_matrices[class_id][object_num]
            bonding_box = self.bounding_boxes[class_id][object_num]
            images_in[i], images_ref[i], rotation_matrices[i] = crop_image(
                image_in,
                image_ref,
                rotation,
                bonding_box,
                padding=0.15,
                jitter=0.15,
                flip=True,
            )
            labels[i] = self.class_ids.index(class_id)

        rms = [rotation_matrices[np.nonzero(labels == i)[0]] for i in range(3)]
        counts = np.zeros(3, 'int32')
        for i in range(batch_size):
            label = labels[i]
            rotation_matrices_random[i] = rms[label][(counts[label] + 1) % len(rms[label])].copy()
            counts[label] += 1

        # delete masked pixels
        images_ref[:, :3, :, :] *= images_ref[:, 3][:, None, :, :]

        return images_in, images_ref, rotation_matrices, rotation_matrices_random, labels

    def get_all_batches_for_evaluation(self, batch_size, class_id, num_views=None):
        num_objects = self.num_data[class_id]
        for batch_num in range((num_objects - 1) / batch_size + 1):
            batch_size2 = min(num_objects - batch_num * batch_size, batch_size)
            images_in = np.zeros((batch_size2, 3, self.image_size, self.image_size), 'float32')
            voxels = np.zeros((batch_size2, 32, 32, 32), 'float32')
            for i in range(batch_size2):
                object_num = batch_num * batch_size + i
                image_in = self.images_original[class_id][object_num]
                image_ref = self.images_ref[class_id][object_num].astype('float32') / 255.
                rotation = self.rotation_matrices[class_id][object_num]
                bounding_box = self.bounding_boxes[class_id][object_num]
                images_in[i], _, _ = crop_image(
                    image_in,
                    image_ref,
                    rotation,
                    bounding_box,
                    padding=0.15,
                    jitter=0,
                    flip=False,
                )
                voxels[i] = self.voxels[class_id][object_num].astype('float32')

            yield images_in, voxels

