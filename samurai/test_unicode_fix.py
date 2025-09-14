#!/usr/bin/env python3
"""
测试Unicode修复
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试导入功能"""
    print("测试导入功能...")
    
    try:
        from scripts.onnx_inference import SAMURAIONNXPredictor
        print("SAMURAIONNXPredictor 导入成功")
        
        from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker
        print("SAMURAITracker 导入成功")
        
        return True
    except ImportError as e:
        print(f"导入失败: {e}")
        return False

def test_simple_demo():
    """测试简单演示"""
    print("\n测试简单演示...")
    
    try:
        # 测试简单示例的演示模式
        import subprocess
        result = subprocess.run([
            sys.executable, "simple_tracking_example.py", "--demo"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("简单演示测试通过")
            return True
        else:
            print(f"简单演示测试失败:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("简单演示测试超时")
        return False
    except Exception as e:
        print(f"简单演示测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("Unicode修复测试")
    print("=" * 30)
    
    tests = [
        ("导入功能测试", test_imports),
        ("简单演示测试", test_simple_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 20)
        
        try:
            if test_func():
                passed += 1
                print(f"{test_name} 通过")
            else:
                print(f"{test_name} 失败")
        except Exception as e:
            print(f"{test_name} 异常: {e}")
    
    # 总结
    print(f"\n测试结果总结")
    print("=" * 20)
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("所有测试通过！Unicode问题已修复。")
        return 0
    else:
        print("部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())
