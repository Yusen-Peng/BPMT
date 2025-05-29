import numpy as np
import random

from torch.utils.data import Dataset

import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window, thres):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
        c_b = begin + bias
        c_e = end - bias
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        
        #cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)  # constraint cropped_length lower bound as thres
        
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p).item()), thres),
            valid_size
        )



        # FIXME: edge case handling
        if valid_size - cropped_length + 1 <= 0:
            bias = 0
        else:
            bias = np.random.randint(0, valid_size - cropped_length + 1)


        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
        c_b = begin + bias
        c_e = begin + bias + cropped_length

    # resize
    data = torch.tensor(data, dtype=torch.float)  # C, crop_t, V, M
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, cropped_length)  # V*M, C, crop_t
    data = F.interpolate(data, size=window, mode='linear', align_corners=False)  # V*M, C, T
    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float)
    index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()


def valid_crop_uniform(data_numpy, valid_frame_num, p_interval, window, thres):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        #cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)

        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p).item()), thres),
            valid_size
        )


        bias = int((1 - p) * valid_size / 2)

        if cropped_length < window:
            inds = np.arange(cropped_length)
        else:
            bids = np.array(
                [i * cropped_length // window for i in range(window + 1)])
            bst = bids[:window]
            inds = bst

        inds = inds + bias
        data = data_numpy[:, inds, :, :]

    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        # cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres),
        #                             valid_size)  # constraint cropped_length lower bound as 64
        cropped_length = np.mininum(
            np.maximum(int(np.floor(valid_size * p).item()), thres),
            valid_size
        )
        
        # FIXME: edge case handling
        if valid_size - cropped_length + 1 <= 0:
            bias = 0
        else:
            bias = np.random.randint(0, valid_size - cropped_length + 1)

        if cropped_length < window:
            inds = np.arange(cropped_length)
        elif window <= cropped_length < 2 * window:
            basic = np.arange(window)
            inds = np.random.choice(window + 1, cropped_length - window, replace=False)
            offset = np.zeros(window + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bsize = np.diff(bids)
            bst = bids[:window]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + bias
        data = data_numpy[:, inds, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    index_t = torch.tensor(inds, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(inds))  # V*M, C, crop_t

    if len(inds) != window:
        data = F.interpolate(data, size=window, mode='linear', align_corners=False)  # V*M, C, T
        index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()

    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()


def scale(data_numpy, scale=0.2, p=0.5):
    if random.random() < p:
        scale = 1 + np.random.uniform(-1, 1, size=(3, 1, 1, 1)) * np.array(scale)
        return data_numpy * scale
    else:
        return data_numpy.copy()


''' AimCLR '''
transform_order = {'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]}


def subtract(data_numpy, p=0.5):
    joint = random.randint(0, 24)

    C, T, V, M = data_numpy.shape
    if random.random() < p:
        data_numpy_new = np.zeros((C, T, V, M))
        for i in range(V):
            data_numpy_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, joint, :]
        return data_numpy_new
    else:
        return data_numpy.copy()


def temporal_flip(data_numpy, index_t, p=0.5):
    C, T, V, M = data_numpy.shape
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return data_numpy[:, time_range_reverse, :, :], -index_t
    else:
        return data_numpy.copy(), index_t.copy()


def spatial_flip(data_numpy, p=0.5):
    if random.random() < p:
        index = transform_order['ntu']
        return data_numpy[:, :, index, :]
    else:
        return data_numpy.copy()


def rotate(data_numpy, axis=None, angle=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)

    if angle != None:
        angle_next = random.uniform(-angle, angle)
    else:
        angle_next = random.uniform(-30, 30)

    if random.random() < p:
        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, math.cos(angle), math.sin(angle)],
                          [0, -math.sin(angle), math.cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[math.cos(angle), 0, -math.sin(angle)],
                          [0, 1, 0],
                          [math.sin(angle), 0, math.cos(angle)]])
        # z
        if axis_next == 2:
            R = np.array([[math.cos(angle), math.sin(angle), 0],
                          [-math.sin(angle), math.cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def shear(data_numpy, s1=None, s2=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        if s1 != None:
            s1_list = s1
        else:
            s1_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        if s2 != None:
            s2_list = s2
        else:
            s2_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

        R = np.array([[1, s1_list[0], s2_list[0]],
                      [s1_list[1], 1, s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def drop_axis(data_numpy, axis=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)

    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp
    else:
        return data_numpy.copy()


def drop_joint(data_numpy, joint_list=None, time_range=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        if joint_list != None:
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, joint_list)
            joint_list_ = sorted(joint_list_)
        else:
            random_int = random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)

        if time_range != None:
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, time_range)
            time_range_ = sorted(time_range_)
        else:
            random_int = random.randint(16, 32)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp
    else:
        return data_numpy.copy()


def gaussian_noise(data_numpy, mean=0, std=0.05, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy.copy()


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 2], p=0.5):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)
        self.p = p

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x).double()
        if prob < self.p:
            x = x.permute(3, 0, 2, 1)
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2)), groups=self.channels)
            x = x.permute(1, -1, -2, 0)

        return x.numpy()


def gaussian_filter(data_numpy, kernel=15, sig_list=[0.1, 2], p=0.5):
    g = GaussianBlurConv(3, kernel, sig_list, p)
    return g(data_numpy)


''' Skeleton AdaIN '''
def skeleton_adain_bone_length(input, ref): # C T V M
    eps = 1e-5
    center = 1
    ref_c = ref[:, :, center, :]

    # joint to bone (joint2bone)
    j2b = joint2bone()
    bone_i = j2b(input) # C T V M
    bone_r = j2b(ref)

    bone_length_i = np.linalg.norm(bone_i, axis=0) # T V M
    bone_length_r = np.linalg.norm(bone_r, axis=0)

    bone_length_scale = (bone_length_r + eps) / (bone_length_i + eps) # T V M
    bone_length_scale = np.expand_dims(bone_length_scale, axis=0) # 1 T V M

    bone_i = bone_i * bone_length_scale

    # bone to joint (bone2joint)
    b2j = bone2joint()
    joint = b2j(bone_i, ref_c)
    return joint


class joint2bone(nn.Module):
    def __init__(self):
        super(joint2bone, self).__init__()
        self.pairs = [(0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                      (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                      (19, 18), (20, 1), (21, 7), (22, 7), (23, 11), (24, 11)]

    def __call__(self, joint):
        bone = np.zeros_like(joint)
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
        return bone


class bone2joint(nn.Module):
    def __init__(self):
        super(bone2joint, self).__init__()
        self.center = 1
        self.pairs_1 = [(0, 1), (20, 1)]
        self.pairs_2 = [(2, 20), (4, 20), (8, 20), (12, 0), (16, 0)]
        self.pairs_3 = [(3, 2), (5, 4), (9, 8), (13, 12), (17, 16)]
        self.pairs_4 = [(6, 5), (10, 9), (14, 13), (18, 17)]
        self.pairs_5 = [(7, 6), (11, 10), (15, 14), (19, 18)]
        self.pairs_6 = [(21, 7), (22, 7), (23, 11), (24, 11)]

    def __call__(self, bone, center):
        joint = np.zeros_like(bone)
        joint[:, :, self.center, :] = center
        for v1, v2 in self.pairs_1:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_2:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_3:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_4:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_5:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_6:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        return joint

def to_motion(input): # C T V M
    C, T, V, M = input.shape
    motion = np.zeros_like(input)
    motion[:, :T - 1] = np.diff(input, axis=1)
    return motion

class SF_NTU_Dataset(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()
        if partition:
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
       
        # FIXME: shape issues
        #N, T, _ = self.data.shape
        #self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.data = self.data.transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num < self.thres:
            # skip or dummy return
            data_numpy = np.zeros((3, self.window_size, 25, 2), dtype=np.float32)
            index_t = np.linspace(-1, 1, self.window_size)
            label = self.label[index]
            return data_numpy, index_t, label, index



        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        if self.uniform:
            data_numpy, index_t = valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                           self.window_size, self.thres)
        else:

            data_numpy, index_t = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)
            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                data_adain = data_adain[:, t_idx]
                data_numpy = skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()

        # modality
        if self.data_type == 'b':
            j2b = joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]
        
        # reshape (3, 64, 24, 2) → (64, 144)
        data_reshaped = data_numpy.transpose(1, 0, 2, 3).reshape(64, -1)

        return data_reshaped, index_t, label, index

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":

    # ==== subject-split =====
    train_dataset = SF_NTU_Dataset(
        data_path='NTU_SF_xsub.npz',
        p_interval=[0.5, 1.0],
        split='train',
        data_type='j',
        aug_method='123456789ab',
        intra_p=0.8,
        inter_p=0.2,
        window_size=64,
        debug=False,
        thres=8,
        uniform=False,
        partition=True
    )
    
    for i in range(len(train_dataset)):
        data, index_t, label, idx = train_dataset[i]
        if i == 100:
            print(f"[TRAIN] Sample {i}: Data shape: {data.shape}, Index_t shape: {index_t.shape}, Label: {label}, Index: {idx}")
    print(f"Total training samples: {len(train_dataset)}")


    test_dataset = SF_NTU_Dataset(
        data_path='NTU_SF_xsub.npz',
        p_interval=[0.5, 1.0],
        split='test',
        data_type='j',
        aug_method='123456789ab',
        intra_p=0.8,
        inter_p=0.2,
        window_size=64,
        debug=False,
        thres=8,
        uniform=False,
        partition=True
    )
    for i in range(len(test_dataset)):
        data, index_t, label, idx = test_dataset[i]
        if i == 100:
            print(f"[TEST] Sample {i}: Data shape: {data.shape}, Index_t shape: {index_t.shape}, Label: {label}, Index: {idx}")
    print(f"Total test samples: {len(test_dataset)}")

    print("=" * 80)
    print(f"✅ Total samples (train + test): {len(train_dataset) + len(test_dataset)}")
