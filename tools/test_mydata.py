import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

config = './tutorial_exps/my_customconfig_2.py'
checkpoint = './tutorial_exps/latest.pth'

device='cuda:0'
config = mmcv.Config.fromfile(config)
config.model.pretrained = None
model = build_detector(config.model)
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

model.CLASSES = checkpoint['meta']['CLASSES']
model.cfg = config
model.to(device)
model.eval()

img = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/1.png'
result = inference_detector(model, img)
show_result_pyplot(model, img, result)