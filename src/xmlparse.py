import xml.etree.ElementTree as ET

def parseXML(xml_file: str):

	tree = ET.parse(xml_file)
	root = tree.getroot()

	list_with_all_boxes = []
	list_with_classnames = []

	for boxes in root.iter('object'):
		# filename = root.find('filename').text
		classname = boxes.find('name').text
		ymin, xmin, ymax, xmax = None, None, None, None

		ymin = int(boxes.find("bndbox/ymin").text)
		xmin = int(boxes.find("bndbox/xmin").text)
		ymax = int(boxes.find("bndbox/ymax").text)
		xmax = int(boxes.find("bndbox/xmax").text)

		list_with_single_boxes = [xmin, ymin, xmax, ymax]
		list_with_all_boxes.append(list_with_single_boxes)
		list_with_classnames.append(classname)

	return list_with_classnames, list_with_all_boxes

# classes, boxes = parseXML(r"C:\Users\trizo\Downloads\Documents\Sem6\CV\Project\VOC2012\Annotations\2007_000032.xml")
# print(classes)
# print()
# print(boxes)