# python ./avg_color.py '#7490BB' '#6992C2' '#6693C6' '#6992C2' '#68AAE4' '#5EA0D6' '#6EACE4' '#6AA2DB'  '#5AA3E4' '#7188A5' 




import sys
import numpy as np
import cv2
from statistics import median, stdev

def hex_to_rgb(hex_color):
    """将十六进制颜色代码转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """将RGB元组转换为十六进制颜色代码"""
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def analyze_channels(rgb_colors):
    """分析每个颜色通道的统计数据"""
    # 将颜色按通道分开
    r_vals = [c[0] for c in rgb_colors]
    g_vals = [c[1] for c in rgb_colors]
    b_vals = [c[2] for c in rgb_colors]
    
    # 计算各通道统计量
    stats = {
        'Red': {
            'min': min(r_vals),
            'max': max(r_vals),
            'mean': round(np.mean(r_vals), 2),
            'median': median(r_vals),
            'std': round(stdev(r_vals), 2)
        },
        'Green': {
            'min': min(g_vals),
            'max': max(g_vals),
            'mean': round(np.mean(g_vals), 2),
            'median': median(g_vals),
            'std': round(stdev(g_vals), 2)
        },
        'Blue': {
            'min': min(b_vals),
            'max': max(b_vals),
            'mean': round(np.mean(b_vals), 2),
            'median': median(b_vals),
            'std': round(stdev(b_vals), 2)
        }
    }
    
    return stats

def display_stats(stats):
    """打印通道统计信息"""
    print("\n颜色通道统计分析:")
    print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        '通道', '最小值', '最大值', '平均值', '中位数', '标准差'))
    
    for channel, data in stats.items():
        print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
            channel,
            data['min'],
            data['max'],
            data['mean'],
            data['median'],
            data['std']
        ))

def display_color_blocks(colors, avg_color, stats):
    """使用OpenCV显示颜色对比图和统计信息"""
    # 设置图像参数
    block_height = 200
    block_width = 200
    margin = 10
    text_height = 30
    stats_height = 100
    
    # 计算图像总高度和宽度
    num_colors = len(colors)
    total_width = (block_width + margin) * (num_colors + 1) + margin
    total_height = block_height + text_height * 2 + margin * 3 + stats_height
    
    # 创建白色背景图像
    image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 绘制原始颜色块
    for i, color in enumerate(colors):
        x_start = margin + i * (block_width + margin)
        x_end = x_start + block_width
        
        # 绘制颜色块
        rgb = hex_to_rgb(color)
        image[text_height:text_height+block_height, x_start:x_end] = rgb[::-1]  # OpenCV使用BGR格式
        
        # 添加颜色文本
        cv2.putText(image, color, (x_start, text_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制平均颜色块
    avg_x_start = margin + num_colors * (block_width + margin)
    avg_x_end = avg_x_start + block_width
    
    # 绘制平均颜色块
    image[text_height:text_height+block_height, avg_x_start:avg_x_end] = avg_color[::-1]
    
    # 添加"平均颜色"文本
    avg_hex = rgb_to_hex(avg_color)
    cv2.putText(image, f"Average: {avg_hex}", (avg_x_start, text_height - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 添加标题
    cv2.putText(image, "Original Colors", (margin, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "Average Color", (avg_x_start, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 添加统计信息
    y_start = block_height + text_height + margin * 2
    
    # 绘制统计表标题
    cv2.putText(image, "Color Channel Statistics:", (margin, y_start + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制统计表
    headers = ["Channel", "Min", "Max", "Mean", "Median", "Std"]
    for i, header in enumerate(headers):
        cv2.putText(image, header, (margin + i*100, y_start + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    row_height = 20
    for i, (channel, data) in enumerate(stats.items()):
        y = y_start + 60 + i * row_height
        cv2.putText(image, channel, (margin, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, str(data['min']), (margin + 100, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, str(data['max']), (margin + 200, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, str(data['mean']), (margin + 300, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, str(data['median']), (margin + 400, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, str(data['std']), (margin + 500, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 显示图像
    cv2.imshow("Color Analysis", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python color_average.py color1 color2 color3 ...")
        sys.exit(1)
    
    colors = sys.argv[1:]
    
    # 验证颜色格式
    for color in colors:
        if not (color.startswith('#') and len(color) == 7):
            print(f"错误: '{color}' 不是有效的十六进制颜色代码 (例如 #RRGGBB)")
            sys.exit(1)
    
    # 将所有颜色转换为RGB
    rgb_colors = [hex_to_rgb(color) for color in colors]
    
    # 计算平均颜色
    avg_rgb = (
        round(np.mean([c[0] for c in rgb_colors])),
        round(np.mean([c[1] for c in rgb_colors])),
        round(np.mean([c[2] for c in rgb_colors]))
    )
    avg_hex = rgb_to_hex(avg_rgb)
    
    print(f"平均颜色 (RGB): {avg_rgb}")
    print(f"平均颜色 (十六进制): {avg_hex}")
    
    if len(rgb_colors) > 1:
        # 分析通道统计信息
        stats = analyze_channels(rgb_colors)
        display_stats(stats)
        
        # 显示颜色对比图和统计信息
        display_color_blocks(colors, avg_rgb, stats)