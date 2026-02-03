#!/usr/bin/env python3
"""
Icon Extractor - 图标提取工具
从合成图片中自动检测并提取独立图标，转换为透明背景 PNG 文件。
"""

import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("错误: 请先运行安装脚本 install.sh")
    sys.exit(1)


def remove_black_background(img, threshold=30):
    """将黑色背景转换为透明"""
    img = img.convert("RGBA")
    pixels = img.load()
    width, height = img.size
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r < threshold and g < threshold and b < threshold:
                pixels[x, y] = (0, 0, 0, 0)
    
    return img


def find_icon_regions(img, threshold=30, min_pixels=500):
    """使用连通组件检测找到图片中所有图标区域"""
    img_rgb = img.convert("RGB")
    pixels = img_rgb.load()
    width, height = img_rgb.size
    
    # 创建二值化掩码
    mask = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r > threshold or g > threshold or b > threshold:
                mask[y][x] = True
    
    # 使用 flood fill 找到连通区域
    visited = [[False] * width for _ in range(height)]
    regions = []
    
    def flood_fill(start_x, start_y):
        queue = [(start_x, start_y)]
        min_x, min_y = start_x, start_y
        max_x, max_y = start_x, start_y
        pixel_count = 0
        
        while queue:
            x, y = queue.pop(0)
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if visited[y][x] or not mask[y][x]:
                continue
            
            visited[y][x] = True
            pixel_count += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    queue.append((x + dx, y + dy))
        
        return (min_x, min_y, max_x, max_y, pixel_count)
    
    for y in range(height):
        for x in range(width):
            if mask[y][x] and not visited[y][x]:
                region = flood_fill(x, y)
                if region[4] >= min_pixels:
                    regions.append(region)
    
    return regions


def merge_close_regions(regions, distance_threshold=20):
    """合并距离较近的区域"""
    if not regions:
        return []
    
    merged = []
    used = [False] * len(regions)
    
    for i, r1 in enumerate(regions):
        if used[i]:
            continue
        
        min_x, min_y, max_x, max_y = r1[0], r1[1], r1[2], r1[3]
        
        changed = True
        while changed:
            changed = False
            for j, r2 in enumerate(regions):
                if used[j] or i == j:
                    continue
                
                dx = max(0, max(r2[0] - max_x, min_x - r2[2]))
                dy = max(0, max(r2[1] - max_y, min_y - r2[3]))
                
                if dx < distance_threshold and dy < distance_threshold:
                    min_x = min(min_x, r2[0])
                    min_y = min(min_y, r2[1])
                    max_x = max(max_x, r2[2])
                    max_y = max(max_y, r2[3])
                    used[j] = True
                    changed = True
        
        used[i] = True
        merged.append((min_x, min_y, max_x, max_y))
    
    return merged


def sort_regions_by_position(regions, height=None):
    """按位置排序区域（从左到右，从上到下）"""
    if not regions:
        return []
    
    y_coords = [(r[1] + r[3]) / 2 for r in regions]
    if y_coords:
        y_min, y_max = min(y_coords), max(y_coords)
        row_height = (y_max - y_min) / max(1, len(set(int(y / 100) for y in y_coords)))
        row_height = max(row_height, 100)
    else:
        row_height = 100
    
    def get_sort_key(region):
        center_y = (region[1] + region[3]) / 2
        center_x = (region[0] + region[2]) / 2
        row = int(center_y / row_height)
        return (row, center_x)
    
    return sorted(regions, key=get_sort_key)


def extract_icons(source_path, output_dir, names=None, threshold=30, 
                  min_pixels=500, merge_distance=20, padding=5):
    """从图片中提取所有图标"""
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(source_path)
    width, height = img.size
    print(f"图片尺寸: {width} x {height}")
    
    print("正在检测图标区域...")
    regions = find_icon_regions(img, threshold, min_pixels)
    print(f"检测到 {len(regions)} 个初始区域")
    
    merged_regions = merge_close_regions(regions, merge_distance)
    print(f"合并后得到 {len(merged_regions)} 个图标")
    
    sorted_regions = sort_regions_by_position(merged_regions, height=height)
    
    for idx, region in enumerate(sorted_regions):
        min_x, min_y, max_x, max_y = region
        
        crop_left = max(0, min_x - padding)
        crop_top = max(0, min_y - padding)
        crop_right = min(width, max_x + padding)
        crop_bottom = min(height, max_y + padding)
        
        icon = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        icon_transparent = remove_black_background(icon, threshold)
        
        if names and idx < len(names):
            name = names[idx].strip()
        else:
            name = f"icon_{idx + 1:02d}"
        
        output_path = os.path.join(output_dir, f"{name}.png")
        icon_transparent.save(output_path, "PNG")
        print(f"  已保存: {name}.png ({crop_right - crop_left}x{crop_bottom - crop_top})")
    
    print(f"\n完成！共提取 {len(sorted_regions)} 个图标到 {output_dir}")
    return len(sorted_regions)


def main():
    parser = argparse.ArgumentParser(description="从图片中提取图标并转换为透明背景")
    parser.add_argument("source", help="源图片路径")
    parser.add_argument("output", help="输出目录")
    parser.add_argument("-t", "--threshold", type=int, default=30, help="黑色检测阈值 (默认: 30)")
    parser.add_argument("-m", "--min-pixels", type=int, default=500, help="最小图标像素数 (默认: 500)")
    parser.add_argument("-d", "--merge-distance", type=int, default=20, help="区域合并距离 (默认: 20)")
    parser.add_argument("-p", "--padding", type=int, default=5, help="裁剪边距 (默认: 5)")
    parser.add_argument("-n", "--names", help="图标名称文件（每行一个）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"错误: 找不到文件 '{args.source}'")
        sys.exit(1)
    
    names = None
    if args.names and os.path.exists(args.names):
        with open(args.names, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    
    extract_icons(args.source, args.output, names, args.threshold,
                  args.min_pixels, args.merge_distance, args.padding)


if __name__ == "__main__":
    main()
