import torch
import torch.nn.functional as F

def roi_pooling(feature_map, rois, size=(7, 7)):
	"""
	:param feature_map: (1, C, H, W)
	:param rois: (1, N, 4) N refers to bbox num, 4 represent (ltx, lty, w, h) 
	:param size: output size
	:return: (1, C, size[0], size[1])
	"""
	output = []
	indices = []
	rois_num = rois.size(1)

	for j in range(feature_map.shape[0]):
		for i in range(rois_num):
			roi = rois[j][i]
			H,W = feature_map.shape[2:]
			x1,y1,x2,y2 = roi
			x1,y1,x2,y2 = map(int,[x1*W,y1*H,x2*W,y2*H])
			if x2<=x1 or y2<=y1:
				continue
			output.append(F.adaptive_max_pool2d(feature_map[[j], :, y1:y2, x1:x2], size))
			indices.append(j*rois_num + i)

	return torch.cat(output), indices

# inp = torch.zeros(3,512,7,7)
# rois = torch.cat([torch.cat([torch.tensor([0,0,0.7,0.5]).unsqueeze(0) for j in range(2)]).unsqueeze(0) for i in range(3)])
# print(rois.shape)
# print(roi_pooling(inp, rois).shape)