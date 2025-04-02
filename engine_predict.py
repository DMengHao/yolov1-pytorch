import os
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
from new_resnet import resnet50
import warnings
import tensorrt as trt
from pycuda import driver
from collections import OrderedDict, namedtuple
import pycuda.autoinit
warnings.simplefilter(action='ignore', category=FutureWarning)


model = resnet50()
model.load_state_dict(torch.load("./weights/YOLOv1_VOC.pth")) # 权重参数
model.eval()
confident = 0.2
iou_con = 0.4

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')  # 将自己的名称输入 （使用自己的数据集时需要更改）
CLASS_NUM = len(VOC_CLASSES)


class Pred():
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ctx = driver.Device(int(0)).make_context()
        self.stream = driver.Stream()
        logger = trt.Logger(trt.Logger.INFO)
        with open('./weights/YOLOv1_VOC.engine', 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        for index in range(self.model.num_bindings):
            if trt.__version__ >= '8.6.1':
                name = self.model.get_tensor_name(index)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(index)
                dtype = trt.nptype(self.model.get_binding_dtype(index))
                shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def letterbox_image(self, image, size):
        # 对图片进行resize，使图片不失真。在空缺的地方进行padding
        h,w,_ = image.shape
        ratio = min(size / h, size / w)
        nw = int(w * ratio)
        nh = int(h * ratio)
        dw = size - nw
        dh = size - nh
        dw /= 2
        dh /= 2
        if image.shape[:2] != (size, size):
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
        return image

    def preprocess(self, img_path):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        h,w,_ = img.shape
        image = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
        # image = self.letterbox_image(img, 448.0)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = np.array([123,117,104], dtype=np.float32)
        img = img - mean
        transform = ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)
        return image, img


    def postprocess(self, Result):
        result = Result.squeeze()  # 7*7*30
        grid_ceil1 = result[:, :, 4].unsqueeze(2)  # 7*7*1
        grid_ceil2 = result[:, :, 9].unsqueeze(2)
        grid_ceil_con = torch.cat((grid_ceil1, grid_ceil2), 2)  # 7*7*2
        grid_ceil_con, grid_ceil_index = grid_ceil_con.max(2)  # 按照第二个维度求最大值  7*7   一个grid ceil两个bbox，两个confidence
        class_p, class_index = result[:, :, 10:].max(2)  # size -> 7*7   找出单个grid ceil预测的物体类别最大者
        class_confidence = class_p * grid_ceil_con  # 7*7   真实的类别概率
        bbox_info = torch.zeros(7, 7, 6)
        for i in range(0, 7):
            for j in range(0, 7):
                bbox_index = grid_ceil_index[i, j]
                bbox_info[i, j, :5] = result[i, j, (bbox_index * 5):(bbox_index + 1) * 5]  # 删选bbox 0-5 或者5-10
        bbox_info[:, :, 4] = class_confidence
        bbox_info[:, :, 5] = class_index
        bbox = bbox_info
        C = 448.0/7
        for i in range(0, 7):
            for j in range(0, 7):
                xc = bbox[i, j, 0] # 相对于所在grid cell 左上角相对坐标的偏移量
                yc = bbox[i, j, 1]
                Xc = (j+xc)*C # 相对于远点的中心点坐标
                Yc = (i+yc)*C
                w = bbox[i, j, 2]
                h = bbox[i, j, 3]
                w_abs = w*448.0 # 相对于448.0的宽高
                h_abs = h*448.0
                xmin = Xc - w_abs/2
                ymin = Yc - h_abs/2
                xmax = Xc + w_abs/2
                ymax = Yc + h_abs/2
                bbox[i, j, 0] = max(0, min(xmin, 448.0))
                bbox[i, j, 1] = max(0, min(xmax, 448.0))
                bbox[i, j, 2] = max(0, min(ymin, 448.0))
                bbox[i, j, 3] = max(0, min(ymax, 448.0))

        # NMS
        '''
            YOLOv1的NMS逻辑梳理：
                1.根据类别顺序和置信度从大到小进行重塑bbox(49,6)
                2.循环标签个数次用于抑制每一种类别
                3.判断类别中置信度是否超过给定阈值，超过则进行判断与后续框的IOU，如果IOU小于阈值或后续框置信度低于confident，后续框置信度置为0
                4.处理完一个类别后，将置信度大于0的框添加到bboxes列表当中
            Deepseek梳理流程为：
                1.按类别分组
                2.置信度筛选
                3.抑制冗余框
        '''
        bbox = bbox.view(-1, 6)   # 49*6
        bboxes = []
        ori_class_index = bbox[:, 5]
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        class_index = class_index.tolist()   # 从0开始排序到7
        bbox = bbox[class_order, :]  # 更改bbox排列顺序
        a = 0
        for i in range(0, CLASS_NUM):
            num = class_index.count(i)
            if num == 0:
                continue
            x = bbox[a:a+num, :]   # 提取同一类别的所有信息
            score = x[:, 4]
            score_index, score_order = score.sort(dim=0, descending=True) # 从大到小进行排序 score
            y = x[score_order, :]   # 同一种类别按照置信度排序
            if y[0, 4] >= confident:    # 物体类别的最大置信度大于给定值才能继续删选bbox，否则丢弃全部bbox
                for k in range(0, num):
                    y_score = y[:, 4]   # 每一次将置信度置零后都重新进行排序，保证排列顺序依照置信度递减
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    if y[k, 4] > 0:
                        area0 = (y[k, 1] - y[k, 0]) * (y[k, 3] - y[k, 2])
                        for j in range(k+1, num):
                            area1 = (y[j, 1] - y[j, 0]) * (y[j, 3] - y[j, 2])
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 1], y[j, 1])
                            y1 = max(y[k, 2], y[j, 2])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = 0
                                h = 0
                            inter = w * h
                            iou = inter / (area0 + area1 - inter + 1e-10)
                            # iou大于一定值则认为两个bbox识别了同一物体删除置信度较小的bbox
                            # 同时物体类别概率小于一定值则认为不包含物体
                            if iou >= iou_con or y[j, 4] < confident:
                                y[j, 4] = 0
                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])
            a = num + a
        return bboxes

    def result(self, img_path):
        try:
            self.ctx.push()
            im0, im = self.preprocess(img_path)
            im = im.to('cuda')

            self.binding_addrs['input'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()
            Result = self.bindings['output'].data

            bboxes = self.postprocess(Result)
            return im0, bboxes
        finally:
            self.ctx.pop()

    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime

    def draw(self, im0, bboxes, name):
        if len(bboxes) == 0:
            print(f"{name}未识别到任何物体")
        for i in range(0, len(bboxes)):    # bbox坐标将其转换为原图像的分辨率
            bboxes[i][0] = bboxes[i][0]
            bboxes[i][1] = bboxes[i][1]
            bboxes[i][2] = bboxes[i][2]
            bboxes[i][3] = bboxes[i][3]

            x1 = bboxes[i][0].item()    # 后面加item()是因为画框时输入的数据不可一味tensor类型
            x2 = bboxes[i][1].item()
            y1 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            class_name = bboxes[i][5].item()

            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))   # 画框
            cv2.putText(im0, VOC_CLASSES[int(class_name)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)
        cv2.imencode('.jpg', im0)[1].tofile(name)


if __name__ == "__main__":
    save_path = './results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    xml_files = os.listdir('VOCdevkit//VOC2007//JPEGImages//')
    Pred = Pred(model)
    for i, img_path in enumerate(xml_files):
        im0,bboxes = Pred.result('VOCdevkit//VOC2007//JPEGImages//'+img_path)
        Pred.draw(im0, bboxes, name=f'{save_path}result{i}.jpg')

