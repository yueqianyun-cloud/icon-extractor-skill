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


def remove_black_background(img, threshold=30, clean_edge=0):
    """
    将黑色背景转换为透明
    """
    img = img.convert("RGBA")
    pixels = img.load()
    width, height = img.size
    
    edge_multiplier = 1 + clean_edge * 2
    effective_threshold = threshold * edge_multiplier
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            brightness = max(r, g, b)
            
            if brightness < threshold:
                pixels[x, y] = (0, 0, 0, 0)
            elif brightness < effective_threshold:
                max_rgb = max(r, g, b)
                min_rgb = min(r, g, b)
                saturation = (max_rgb - min_rgb) / max(max_rgb, 1) if max_rgb > 0 else 0
                
                if saturation < 0.35:
                    ratio = (brightness - threshold) / (effective_threshold - threshold)
                    alpha = int(ratio * 255)
                    pixels[x, y] = (r, g, b, alpha)
    
    if clean_edge > 0:
        for _ in range(clean_edge):
            img2 = img.copy()
            pixels2 = img2.load()
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    r, g, b, a = pixels[x, y]
                    if a == 0:
                        continue
                    
                    transparent_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            _, _, _, na = pixels[x + dx, y + dy]
                            if na < 128:
                                transparent_count += 1
                    
                    if transparent_count > 0:
                        brightness = max(r, g, b)
                        if brightness < effective_threshold:
                            factor = 1 - (transparent_count / 8) * 0.7
                            new_alpha = int(a * factor * (brightness / effective_threshold))
                            pixels2[x, y] = (r, g, b, max(0, new_alpha))
            
            img = img2
            pixels = img.load()
    
    return img


def remove_light_background(img, threshold=230, clean_edge=2):
    """
    将浅色/白色背景转换为透明
    """
    img = img.convert("RGBA")
    pixels = img.load()
    width, height = img.size
    
    # 阈值范围
    hard_threshold = threshold  # 纯白
    soft_threshold = threshold - 30 - clean_edge * 10  # 浅灰过渡区
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            brightness = (r + g + b) / 3
            
            # 计算饱和度
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            saturation = (max_rgb - min_rgb) / max(max_rgb, 1) if max_rgb > 0 else 0
            
            if brightness > hard_threshold and saturation < 0.1:
                # 纯白/浅灰 -> 完全透明
                pixels[x, y] = (0, 0, 0, 0)
            elif brightness > soft_threshold and saturation < 0.15:
                # 过渡区 -> 渐变透明
                ratio = (brightness - soft_threshold) / (hard_threshold - soft_threshold)
                alpha = int((1 - ratio) * 255)
                pixels[x, y] = (r, g, b, max(0, alpha))
    
    # 边缘清理
    if clean_edge > 0:
        for _ in range(clean_edge):
            img2 = img.copy()
            pixels2 = img2.load()
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    r, g, b, a = pixels[x, y]
                    if a == 0:
                        continue
                    
                    # 检查相邻透明像素
                    transparent_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            _, _, _, na = pixels[x + dx, y + dy]
                            if na < 128:
                                transparent_count += 1
                    
                    # 边缘且亮度高 -> 增加透明度
                    if transparent_count > 0:
                        brightness = (r + g + b) / 3
                        max_rgb = max(r, g, b)
                        min_rgb = min(r, g, b)
                        saturation = (max_rgb - min_rgb) / max(max_rgb, 1) if max_rgb > 0 else 0
                        
                        if brightness > soft_threshold - 20 and saturation < 0.2:
                            factor = 1 - (transparent_count / 8) * 0.8
                            new_alpha = int(a * factor)
                            pixels2[x, y] = (r, g, b, max(0, new_alpha))
            
            img = img2
            pixels = img.load()
    
    return img


def detect_background_type(img, sample_size=20):
    """
    自动检测背景类型（黑色或浅色）
    """
    img_rgb = img.convert("RGB")
    pixels = img_rgb.load()
    width, height = img_rgb.size
    
    # 采样四个角落的像素
    corners = [
        (0, 0), (width-1, 0),
        (0, height-1), (width-1, height-1)
    ]
    
    total_brightness = 0
    sample_count = 0
    
    for cx, cy in corners:
        for dy in range(sample_size):
            for dx in range(sample_size):
                x = min(max(cx + dx if cx == 0 else cx - dx, 0), width - 1)
                y = min(max(cy + dy if cy == 0 else cy - dy, 0), height - 1)
                r, g, b = pixels[x, y]
                total_brightness += (r + g + b) / 3
                sample_count += 1
    
    avg_brightness = total_brightness / sample_count
    return "light" if avg_brightness > 128 else "dark"


def find_icon_regions(img, threshold=30, min_pixels=500, bg_type="dark"):
    """使用连通组件检测找到图片中所有图标区域"""
    img_rgb = img.convert("RGB")
    pixels = img_rgb.load()
    width, height = img_rgb.size
    
    # 创建二值化掩码
    mask = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            brightness = (r + g + b) / 3
            
            if bg_type == "light":
                # 浅色背景：检测深色像素（图标）
                # 排除纯白和接近白色的背景
                max_rgb = max(r, g, b)
                min_rgb = min(r, g, b)
                saturation = (max_rgb - min_rgb) / max(max_rgb, 1) if max_rgb > 0 else 0
                
                # 有颜色的像素或者不是太亮的像素
                if brightness < 220 or saturation > 0.15:
                    mask[y][x] = True
            else:
                # 深色背景：检测亮色像素（图标）
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


def extract_icons_grid(img, rows, cols, output_dir, names=None, 
                       clean_edge=2, bg_type="light", skip_text=True):
    """使用网格模式提取图标（适用于规则排列的图标）"""
    width, height = img.size
    cell_width = width // cols
    cell_height = height // rows
    
    # 如果跳过文字，只取单元格上部的图标部分
    if skip_text:
        icon_height_ratio = 0.6  # 图标占单元格高度的比例（排除底部文字）
    else:
        icon_height_ratio = 1.0
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            
            # 计算裁剪区域
            left = col * cell_width
            top = row * cell_height
            right = (col + 1) * cell_width
            bottom = row * cell_height + int(cell_height * icon_height_ratio)
            
            # 裁剪
            icon = img.crop((left, top, right, bottom))
            
            # 去背景
            if bg_type == "light":
                icon_transparent = remove_light_background(icon, 230, clean_edge)
            else:
                icon_transparent = remove_black_background(icon, 30, clean_edge)
            
            # 自动裁剪透明边缘
            bbox = icon_transparent.getbbox()
            if bbox:
                icon_transparent = icon_transparent.crop(bbox)
            
            # 文件名
            if names and idx < len(names):
                name = names[idx].strip()
            else:
                name = f"icon_{idx + 1:02d}"
            
            output_path = os.path.join(output_dir, f"{name}.png")
            icon_transparent.save(output_path, "PNG")
            w, h = icon_transparent.size
            print(f"  已保存: {name}.png ({w}x{h})")
            count += 1
    
    return count


def extract_icons(source_path, output_dir, names=None, threshold=30, 
                  min_pixels=500, merge_distance=20, padding=5, clean_edge=2,
                  bg_type="auto", grid=None):
    """从图片中提取所有图标"""
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(source_path)
    width, height = img.size
    print(f"图片尺寸: {width} x {height}")
    
    # 自动检测背景类型
    if bg_type == "auto":
        bg_type = detect_background_type(img)
        print(f"检测到背景类型: {'浅色' if bg_type == 'light' else '深色'}")
    
    # 网格模式
    if grid:
        rows, cols = grid
        print(f"使用网格模式: {rows}行 x {cols}列")
        count = extract_icons_grid(img, rows, cols, output_dir, names, clean_edge, bg_type)
        print(f"\n完成！共提取 {count} 个图标到 {output_dir}")
        return count
    
    print("正在检测图标区域...")
    regions = find_icon_regions(img, threshold, min_pixels, bg_type)
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
        
        # 根据背景类型选择处理方法
        if bg_type == "light":
            icon_transparent = remove_light_background(icon, 230, clean_edge)
        else:
            icon_transparent = remove_black_background(icon, threshold, clean_edge)
        
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
    parser.add_argument("-c", "--clean-edge", type=int, default=2, choices=[0,1,2,3],
                        help="边缘清理强度: 0=关闭, 1=轻度, 2=中度(默认), 3=强力")
    parser.add_argument("-b", "--bg-type", choices=["auto", "dark", "light"], default="auto",
                        help="背景类型: auto=自动检测(默认), dark=深色/黑色, light=浅色/白色")
    parser.add_argument("-g", "--grid", help="网格模式: 行x列 (如 2x5 表示2行5列)")
    parser.add_argument("-n", "--names", help="图标名称文件（每行一个）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"错误: 找不到文件 '{args.source}'")
        sys.exit(1)
    
    names = None
    if args.names and os.path.exists(args.names):
        with open(args.names, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    
    # 解析网格参数
    grid = None
    if args.grid:
        try:
            rows, cols = map(int, args.grid.lower().split('x'))
            grid = (rows, cols)
        except:
            print(f"错误: 网格格式无效 '{args.grid}'，应为 行x列 (如 2x5)")
            sys.exit(1)
    
    extract_icons(args.source, args.output, names, args.threshold,
                  args.min_pixels, args.merge_distance, args.padding, 
                  args.clean_edge, args.bg_type, grid)


if __name__ == "__main__":
    main()
