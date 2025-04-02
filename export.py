import argparse
import logging
import os
import sys
import torch
from new_resnet import resnet50
import onnx
from onnx import shape_inference
from pathlib import Path
FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # YOLOv1 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 创建日志记录器并设置级别
logger = logging.getLogger('YOLOv1_export')
logger.setLevel(logging.INFO)
# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个控制台输出的处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 添加处理器
logger.addHandler(console_handler)

def export_onnx(model, im, file,input_names, output_names, dynamic_axes, opset_version)->None:
    logger.info(f'开始导出ONNX模型到{os.path.splitext(file)}')
    # Step 1:导出原始ONNX模型（保留动态折叠优化）
    torch.onnx.export(model,
                      im,
                      file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=None,
                      opset_version=opset_version,
                      do_constant_folding=True)
    # Step 2:加载到处的ONNX模型并进行形状推断
    onnx_model = onnx.load(file)
    # 执行形状推断（自动传播维度信息）
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred_model, file)
    logger.info(f'ONNX模型成功保存至{os.path.splitext(file)}')

def export_engine(im, file, workspace=4):
    im = im.to('cuda')
    assert im.device.type != 'cpu', '导出engine文件必须在GPU上面运行'
    import tensorrt as trt
    # 创建Builder与Config
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    config = builder.create_builder_config()
    # 工作空间设置
    config.max_workspace_size = workspace*1 << 30
    # 时间缓存 读取或创建时间缓存加速构建
    exists = False
    cache = ''
    timing_cache = config.create_timing_cache(Path(cache).read_bytes() if exists else b'')
    config.set_timing_cache(timing_cache, ignore_mismatch=True)
    # 网络解析
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, log)
    onnx = file
    f = os.path.splitext(file)[0] + '.engine'
    parser.parse_from_file(str(onnx))
    # 输入输出配置
    # 引擎构建与保存
    engine = builder.build_engine(network, config)
    with open(f, 'wb') as t:
        t.write(engine.serialize())

@torch.no_grad()
def run(weights=ROOT / 'yolov1.pt',  # weights path
        imgsz=(448, 448),  # image (height, width)
        batch_size=1,
        include=('onnx', 'engine')
        ):

    model = resnet50().eval()
    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    im = torch.zeros((batch_size, 3, *imgsz), dtype=torch.float32)

    for i,type in enumerate(include):
        if type == 'onnx':
            # onnx转换输入
            file = os.path.splitext(weights)[0]+'.onnx'
            input_names = ['input']
            output_names = ['output']
            dynamic_axes={'input':{0:'-1'},'output':{0:'-1'}}
            opset_version=11
            export_onnx(model, im, file,input_names, output_names, dynamic_axes, opset_version)
        elif type == 'engine':
            file = os.path.splitext(weights)[0] + '.onnx'
            if not Path(file).exists():
                input_names = ['input']
                output_names = ['output']
                dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
                opset_version = 11
                export_onnx(model, im, file, input_names, output_names, dynamic_axes, opset_version)
            export_engine(im, file, workspace=4)
        else:
            continue
    logger.info('所有转换全部转换成功！')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/YOLOv1_VOC.pth', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[448,448], help='image (h, w)')
    parser.add_argument('--include', nargs='+',default=['onnx', 'engine'], help='onnx, engine')
    opt = parser.parse_args()
    return opt

def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)