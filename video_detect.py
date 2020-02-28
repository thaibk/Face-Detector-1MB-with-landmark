import torch
import imutils
import numpy as np
import time
from data import cfg_mnet, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from imutils.video import VideoStream, FileVideoStream
from fps import FPS


def judge_side_face(facial_landmarks):
    wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = facial_landmarks[0] - facial_landmarks[2]
    vec_B = facial_landmarks[1] - facial_landmarks[2]
    vec_C = facial_landmarks[3] - facial_landmarks[2]
    vec_D = facial_landmarks[4] - facial_landmarks[2]
    dist_A = np.linalg.norm(vec_A)
    dist_B = np.linalg.norm(vec_B)
    dist_C = np.linalg.norm(vec_C)
    dist_D = np.linalg.norm(vec_D)

    # cal rate
    high_rate = dist_A / dist_C
    width_rate = dist_C / dist_D
    high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
    width_ratio_variance = np.fabs(width_rate - 1)

    if dist_rate < 1.3 and width_ratio_variance < 0.5:
        return True
    return False


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

use_cpu = False
cfg = cfg_mnet
weight_paths = "weights/mobilenet0.25_Final.pth"
net = RetinaFace(cfg=cfg, phase='test')
# cfg = cfg_rfb
# weight_paths = "./weights/RBF_Final.pth"
# net = RFB(cfg=cfg, phase='test')
torch.set_grad_enabled(False)
net = load_model(net, weight_paths, use_cpu)
net.eval()

import torch2trt
# convert tensorrt
x = torch.randn((1, 3, 640, 640)).cuda()
net = torch2trt(net, [x])
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
torch.backends.cudnn.benchmark = True

# vs = FileVideoStream("/home/mdt/ownCloud/datasets/Face_Recognition/output.avi").start()
vs = FileVideoStream("/home/mdt/Downloads/Captures/ 2020-02-25 14-58-37.mp4").start()
# vs = VideoStream("rtsp://admin:meditech123@192.168.100.64:554").start()
# vs = VideoStream("rtsp://admin:meditech123@322nguyentrai.ddns.net:554").start()
# vs = VideoStream("rtsp://admin:admin@118.70.82.46:554").start()
# vs = VideoStream("rtsp://admin:meditech123@192.168.101.65:555").start()
# vs = VideoStream("rtsp://admin:BNGVTX@113.161.36.165:554").start()
fps = FPS().start()
net_inshape = (640, 640)  # h, w
rgb_mean = (104, 117, 123) # bgr order
priorbox = PriorBox(cfg, image_size=net_inshape)
priors = priorbox.forward()
priors = priors.numpy()
while True:
    frame = vs.read()
    # frame = imutils.rotate_bound(frame, 90)
    if frame is None:
        break
    frame_raw = frame.copy()
    # image_path = "fail.jpg"
    # frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # frame_raw = frame.copy()

    h, w = frame.shape[:2]
    d = max(h, w)
    dy = (d - h)
    dx = (d - w)
    img = cv2.copyMakeBorder(frame, 0, dy, 0, dx, borderType=cv2.BORDER_CONSTANT, value=rgb_mean)
    # img = frame.copy()

    img = np.float32(img)
    resize = float(net_inshape[0]) / float(img.shape[0])
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width = net_inshape

    scale = torch.Tensor([im_width, im_height] * 2)
    scale = scale.to(device)
    img -= rgb_mean
    img = img.transpose(2, 0, 1)
    # img = np.stack([img] * 15)
    # img = torch.from_numpy(img)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    tic = time.time()
    loc, conf, landms = net(img)
    # loc = loc[0]
    conf = conf[0]
    # landms = landms[0]
    # print('net forward time: {:.4f}'.format(time.time() - tic))

    prior_data = torch.from_numpy(priors)
    prior_data = prior_data.to(device)
    boxes = decode(loc[0], prior_data, cfg['variance'])  # loc[0] cho batchsize = 1
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]
    landms = decode_landm(landms[0], prior_data, cfg['variance'])
    scale1 = torch.Tensor([im_width, im_height] * 5)
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:750, :]
    landms = landms[:750, :]

    dets = np.concatenate((dets, landms), axis=1)
    dets = dets[dets[:, 4] > 0.8]
    for b in dets:
        text = "{:.4f}".format(b[4])
        landm = b[5:15]
        landm = landm.reshape((5, 2))
        b = list(map(int, b))
        color = (0, 0, 255)
        if judge_side_face(landm):
            color = (0, 255, 0)
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms
        landm = landm.astype(np.int32)
        cv2.circle(frame, tuple(landm[0]), 1, (0, 0, 255), 4)
        cv2.circle(frame, tuple(landm[1]), 1, (0, 255, 255), 4)
        cv2.circle(frame, tuple(landm[2]), 1, (255, 0, 255), 4)
        cv2.circle(frame, tuple(landm[3]), 1, (0, 255, 0), 4)
        cv2.circle(frame, tuple(landm[4]), 1, (255, 0, 0), 4)

    fps.update()
    text_fps = "FPS: {:.3f}".format(fps.get_fps_n())
    cv2.putText(frame, text_fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("frame", imutils.resize(frame, height=1000))
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
    elif key == ord("c"):
        cv2.imwrite("cap.jpg", frame)

cv2.destroyAllWindows()
fps.stop()
print("Total FPS: {}".format(fps.fps()))
vs.stop()