# Road Surface Deterioration Detection using ITS, AI, and IoT

## Overview
This project presents a real-time road surface deterioration detection system utilizing Intelligent Transportation Systems (ITS), Artificial Intelligence (AI), and Internet of Things (IoT). The primary goal is to create a lightweight model capable of identifying road defects such as potholes and cracks using edge devices like dashcams. The developed model is based on YOLOv11n-OBB and has been optimized for deployment on Raspberry Pi 5.

This project is part of a **B.Sc. thesis** for the **Robotics and Automation Engineering** program at **German International University (Berlin Campus)**.

## Special Thanks
A special thanks to **Dr. Amr Talaat** for his invaluable guidance and supervision throughout this research.

## Dataset Sources
The model was trained using a combination of datasets:
- **RDD2022 Dataset**: A comprehensive dataset from the Crowd Sensing-based Road Damage Detection Challenge (CRDDC’2022), containing images from multiple countries annotated with various types of road damage.
- **Kaggle Pothole Dataset**: A dataset compiled by Kaggle user DenisG04, consisting of pothole images that were re-annotated in YOLO format.

All datasets were preprocessed and converted to grayscale for improved efficiency and compatibility with the model.

## Code Inspiration and References
This project took inspiration from various open-source repositories and research papers. Notable references include:
- The **RDD2022 GitHub repository** by Ahmed Nahmad
- The **YOLO implementation by Ultralytics**
- **Darknet YOLOv7 implementation**
- Various pruning techniques referenced from the **CSDN Blog by the user 数学人学python**

All sources and references have been properly cited in the thesis document.

## Results
The model was trained for **450 epochs**, achieving the following results:
- **mAP50 Score**: 76.819%
- **Performance on Raspberry Pi 5**:
  - **Original Model**: 9.56 FPS, mAP50 = 76.819%
  - **10% Pruned Model**: 10 FPS, mAP50 = 75.81%
  - **20% Pruned Model**: 12.5 FPS, mAP50 = 74.5%

This demonstrates that real-time, edge-based road defect detection is feasible for proactive infrastructure maintenance.

## Additional Resources
- **Presentation and Thesis Document**: Both the presentation and the complete thesis are available in the **documents folder**.
- **Citations**: All referenced works and datasets have been cited appropriately within the thesis.

## License
This project is developed for academic research purposes. Please refer to the license file for terms of use.

For any questions or collaboration, feel free to contact the author!

