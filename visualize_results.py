#!/usr/bin/env python3
"""
可视化结果脚本
用于快速从CSV文件生成可视化图像
"""

import argparse
import os
from pathlib import Path
from visualization_data import (
    visualize_from_csv, 
    create_comparison_plot, 
    analyze_training_trends,
    example_usage
)


def main():
    parser = argparse.ArgumentParser(description="CSI实验结果可视化工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 单个实验可视化
    single_parser = subparsers.add_parser('single', help='可视化单个实验结果')
    single_parser.add_argument('csv_file', help='CSV文件路径')
    single_parser.add_argument('--output_dir', help='输出目录')
    single_parser.add_argument('--class_names', nargs='+', 
                              default=['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown'],
                              help='类别名称列表')
    single_parser.add_argument('--no_save', action='store_true', help='不保存图像')
    single_parser.add_argument('--no_show', action='store_true', help='不显示图像')
    
    # 实验对比
    compare_parser = subparsers.add_parser('compare', help='对比多个实验结果')
    compare_parser.add_argument('csv_files', nargs='+', help='CSV文件路径列表')
    compare_parser.add_argument('--labels', nargs='+', help='实验标签列表')
    compare_parser.add_argument('--output_dir', help='输出目录')
    compare_parser.add_argument('--no_save', action='store_true', help='不保存图像')
    compare_parser.add_argument('--no_show', action='store_true', help='不显示图像')
    
    # 趋势分析
    trend_parser = subparsers.add_parser('trend', help='分析训练趋势')
    trend_parser.add_argument('csv_file', help='CSV文件路径')
    trend_parser.add_argument('--output_dir', help='输出目录')
    
    # 示例
    subparsers.add_parser('example', help='显示使用示例')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        # 单个实验可视化
        print(f"正在处理文件: {args.csv_file}")
        visualize_from_csv(
            csv_file_path=args.csv_file,
            output_dir=args.output_dir,
            class_names=args.class_names,
            save_plots=not args.no_save,
            show_plots=not args.no_show
        )
        
    elif args.command == 'compare':
        # 实验对比
        print(f"正在对比 {len(args.csv_files)} 个实验")
        create_comparison_plot(
            csv_files=args.csv_files,
            labels=args.labels,
            output_dir=args.output_dir,
            save_plot=not args.no_save,
            show_plot=not args.no_show
        )
        
    elif args.command == 'trend':
        # 趋势分析
        print(f"正在分析训练趋势: {args.csv_file}")
        analyze_training_trends(
            csv_file_path=args.csv_file,
            output_dir=args.output_dir
        )
        
    elif args.command == 'example':
        # 显示示例
        example_usage()
        
    else:
        # 默认显示帮助
        parser.print_help()


def interactive_mode():
    """交互模式"""
    print("CSI实验结果可视化工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 可视化单个实验结果")
        print("2. 对比多个实验结果")
        print("3. 分析训练趋势")
        print("4. 显示使用示例")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            csv_file = input("请输入CSV文件路径: ").strip()
            if os.path.exists(csv_file):
                output_dir = input("请输入输出目录 (回车使用默认): ").strip()
                output_dir = output_dir if output_dir else None
                
                visualize_from_csv(
                    csv_file_path=csv_file,
                    output_dir=output_dir
                )
            else:
                print(f"错误: 文件 {csv_file} 不存在")
                
        elif choice == '2':
            print("请输入CSV文件路径 (用空格分隔):")
            csv_files = input().strip().split()
            if all(os.path.exists(f) for f in csv_files):
                print("请输入实验标签 (用空格分隔，回车使用默认):")
                labels = input().strip().split()
                labels = labels if labels else None
                
                output_dir = input("请输入输出目录 (回车使用默认): ").strip()
                output_dir = output_dir if output_dir else None
                
                create_comparison_plot(
                    csv_files=csv_files,
                    labels=labels,
                    output_dir=output_dir
                )
            else:
                print("错误: 某些文件不存在")
                
        elif choice == '3':
            csv_file = input("请输入CSV文件路径: ").strip()
            if os.path.exists(csv_file):
                output_dir = input("请输入输出目录 (回车使用默认): ").strip()
                output_dir = output_dir if output_dir else None
                
                analyze_training_trends(
                    csv_file_path=csv_file,
                    output_dir=output_dir
                )
            else:
                print(f"错误: 文件 {csv_file} 不存在")
                
        elif choice == '4':
            example_usage()
            
        elif choice == '5':
            print("再见!")
            break
            
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 没有参数时启动交互模式
        interactive_mode()
    else:
        # 有参数时使用命令行模式
        main() 