import os
import xmltodict

# Load the class mapping
def load_class_mapping(mapping_file):
    class_mapping = {}
    with open(mapping_file) as f:
        for line in f:
            class_id, class_name = line.strip().split()
            class_mapping[class_id] = class_name
    return class_mapping

def convert_voc_to_yolo(xml_file, class_mapping, output_dir):
    with open(xml_file) as f:
        data = xmltodict.parse(f.read())

    img_width = int(data['annotation']['size']['width'])
    img_height = int(data['annotation']['size']['height'])

    yolo_annotations = []

    if 'object' in data['annotation']:
        objects = data['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]  # Ensure it's a list

        for obj in objects:
            class_id = obj['name']
            class_name = class_mapping.get(class_id)
            print(f"Class ID: {class_id}, Class Name: {class_name}")  # Debug print
            if class_name is None:
                continue  # Skip if the class ID is not found in the mapping

            yolo_class_id = list(class_mapping.values()).index(class_name)
            bbox = obj['bndbox']
            xmin = int(round(float(bbox['xmin'])))
            ymin = int(round(float(bbox['ymin'])))
            xmax = int(round(float(bbox['xmax'])))
            ymax = int(round(float(bbox['ymax'])))

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{yolo_class_id} {x_center} {y_center} {width} {height}")

    print(f"Annotations for {xml_file}: {yolo_annotations}")  # Debug print

    output_file = os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.txt'))
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))

def convert_dataset(xml_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            convert_voc_to_yolo(os.path.join(xml_dir, xml_file), class_mapping, output_dir)




if __name__ == '__main__':
    # Load class mappings
    class_mapping = load_class_mapping("class_mapping.txt")

    xml_directory = "E:/Aditya_Thesis_Project_Data_Backup/combined dataset/xmls"
    yolo_output_directory = "E:/Aditya_Thesis_Project_Data_Backup/combined dataset//labels"
    
    convert_dataset(xml_directory, yolo_output_directory, class_mapping)
