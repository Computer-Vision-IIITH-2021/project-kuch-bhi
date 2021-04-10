import cv2

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

def find_region_proposals(image,limit=2000):
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()
	ssresults = ss.process()
	ssresults[:,2] += ssresults[:,0]
	ssresults[:,3] += ssresults[:,1]
	return ssresults[:limit]

# img = cv2.imread(r"C:\Users\trizo\Downloads\Documents\Sem6\CV\Project\VOC2012\JPEGImages\2007_000027.jpg", cv2.IMREAD_COLOR)
# print(find_region_proposals(img)[:10])