from PIL import Image
import os

def convert_transparency_to_black(img_path, save_path):
    img = Image.open(img_path).convert("RGBA")  # 確保圖片是 RGBA 模式
    black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))  # 創建全黑色背景
    black_bg.paste(img, (0, 0), img)  # 將原始圖片貼在黑色背景上
    black_bg.convert("RGB").save(save_path)  # 最終轉為 RGB 模式並保存


path = 'data/emoji/Facebook'
opath = 'data/Newemoji/Facebook'

if path:
    for filename in os.listdir(path):
        if filename.endswith(('.bmp','.png', '.jpg', '.jpeg')):
            convert_transparency_to_black(os.path.join(path, filename), os.path.join(opath, filename))

