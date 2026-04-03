import cv2
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(pose2d=animal)
img = cv2.imread(data/coco/val2017/000000169076.jpg)
x, y, w, h = 57, 126, 316, 219 # dog bounding box
generator = inferencer(img, bboxes=[[x, y, x+w, y+h]])
result = next(generator)
data_sample = result[data_samples]
pred = data_sample.pred_instances
print(Keypoints
