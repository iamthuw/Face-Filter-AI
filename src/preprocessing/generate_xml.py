import xml.etree.ElementTree as ET
import os

def create_xml(data, output_path):
    dataset = ET.Element("dataset")
    images = ET.SubElement(dataset, "images")

    for item in data:
        img_elem = ET.SubElement(images, "image", file=item["file"])
        box_elem = ET.SubElement(img_elem, "box", top="0", left="0", width="256", height="256")
        for i, (x, y) in enumerate(item["landmarks"]):
            ET.SubElement(box_elem, "part", name=str(i), x=str(x), y=str(y))

    tree = ET.ElementTree(dataset)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
