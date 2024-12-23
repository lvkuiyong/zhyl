import os
import shutil

dataset_path = "COVID-19_Radiography_Dataset"  # 数据集源路径
output_path = "new_COVID_19_Radiography_Dataset"  # 数据集的产出路径

# 创建test、train和val文件夹
subfolders = ['test', 'train', 'val']
categories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# 创建各类文件夹
for subfolder in subfolders:
    for category in categories:
        os.makedirs(os.path.join(output_path, subfolder, category, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, subfolder, category, "masks"), exist_ok=True)

# 遍历每个类别进行数据集划分和文件复制
for category in categories:
    image_folder = os.path.join(dataset_path, category, "images")
    mask_folder = os.path.join(dataset_path, category, "masks")

    # 获取文件列表
    images = sorted(os.listdir(image_folder))
    masks = sorted(os.listdir(mask_folder))

    # 计算划分的索引
    total_images = len(images)
    test_size = int(0.2 * total_images)
    val_size = int(0.1 * (total_images - test_size))

    # 划分数据集
    test_images = images[:test_size]
    val_images = images[test_size:test_size + val_size]
    train_images = images[test_size + val_size:]

    # 将图片和mask复制到对应的文件夹
    def copy_images_to_folder(image_list, subset):
        for img in image_list:
            src_img = os.path.join(image_folder, img)
            src_mask = os.path.join(mask_folder, img)
            dest_img = os.path.join(output_path, subset, category, "images", img)
            dest_mask = os.path.join(output_path, subset, category, "masks", img)
            shutil.copy(src_img, dest_img)
            shutil.copy(src_mask, dest_mask)

    # 复制到各个数据集子文件夹
    copy_images_to_folder(test_images, "test")
    copy_images_to_folder(val_images, "val")
    copy_images_to_folder(train_images, "train")