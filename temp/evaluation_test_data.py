import Levenshtein
import json
from collections import Counter
import os
import cv2
import numpy as np

def oder2index(order):
    order_map = {num: i for i, num in enumerate(order)}
    order_index = [order_map[num] for num in range(len(order_map))]
    return order_index

def levenshtein_lib(list1, list2):
    # 将列表转换为字符串（假设元素可哈希且可区分）
    str1 = ''.join(map(str, list1))
    str2 = ''.join(map(str, list2))
    return Levenshtein.distance(str1, str2)

def draw_bboxes_with_order(
    image_path: str, 
    layout_bboxes,
    sorted_layout_indices = None,
    output_path: str = "output_images"
):
    """
    绘制边界框并可视化结果，显示layout的顺序
    
    Args:
        image_path: 原始图像路径
        layout_bboxes: layout边界框列表
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # 绘制layout bboxes (红色)并标记顺序
    for layout_idx, order in enumerate(sorted_layout_indices):
        if order == -1 or layout_idx >= len(layout_bboxes):
            continue
            
        bbox = layout_bboxes[layout_idx]
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制layout bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # 标记layout顺序（从1开始）
        display_order = order
        cv2.putText(img, f"#{display_order} id:{layout_idx}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 保存结果图像
    cv2.imwrite(output_path, img)
    
    print(f"Result image saved to: {output_path}")   

with open('/root/surya/train_temp/dataset_test_doclayout.txt','r') as f:
    lines = f.readlines()
    
surya_orig_lev_dist_list = []
surya_train_lev_dist_list = []

for line in lines:
    img_path, json_path = line.strip().split()
    
    with open(json_path,'r') as f:
        json_data = json.load(f)
    manual_order = json_data['manual_order']

    surya_train_json_path = json_path.replace('test_data_manual','test_data_epoch4').replace('_annotated','')
    with open(surya_train_json_path,'r') as f:
        json_data = json.load(f)
    surya_train_order = json_data['surya_order']

    surya_json_path = json_path.replace('test_data_manual','test_data_surya').replace('_annotated','')
    with open(surya_json_path,'r') as f:
        json_data = json.load(f)
    surya_order = json_data['surya_order']

    surya_orig_lev_dist = levenshtein_lib(manual_order,surya_order)
    surya_orig_lev_dist_list.append(surya_orig_lev_dist)  

    surya_train_lev_dist = levenshtein_lib(manual_order,surya_train_order)
    surya_train_lev_dist_list.append(surya_train_lev_dist)    
    # print(manual_order,surya_train_order,surya_order)
    # break
print(f"before:{np.mean(surya_orig_lev_dist_list)},after:{np.mean(surya_train_lev_dist_list)}")



