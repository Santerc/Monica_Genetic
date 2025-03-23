import math
import numpy as np

def mapping(dst_up, dst_down, src, num):
    """
        将二进制映射到指定范围内，便于遗传算法使用

        Args:
            dst_up (float): 目标上界
            dst_down (float): 目标下界
            src (int): 原范围
            num (int): 待转换二进制数
        Returns:
            (float): 映射后数
    """
    return num * (dst_up - dst_down) / src + dst_down

def bit_concat(a, b):
    """
    将两个二进制数拼接在一起。

    Args:
        a (int): 第一个二进制数
        b (int): 第二个二进制数

    Returns:
        int: 拼接后的二进制数
    """
    a = int(a)
    b = int(b)
    return (a << 32) | b

def bit_break(src):
    """
    将一个64位二进制数裁开为两个32位int

    Args:
        src (int64_t): 被拆开二进制数

    Returns:
        tuple: (前32位, 后32位)
    """
    src = int(src)

    high_32 = (src >> 32) & 0xFFFFFFFF
    low_32 = src & 0xFFFFFFFF

    return int(high_32), int(low_32)

def bit_exchange(src_1, src_2, break_pos):
    """
    将一个64位二进制数裁开为两个32位int

    Args:
        src_1 (int64_t): 第一个二进制数
        src_2 (int64_t): 第二个二进制数
        break_pos (int): 交换位置

    Returns:
        tuple: (交换后第一个数, 交换后第二个数)
    """
    if break_pos < 0 or break_pos > 63:
        raise ValueError("break_pos must be between 0 and 63")

    src_1 = int(src_1)
    src_2 = int(src_2)

    mask = (1 << break_pos) - 1

    high1 = src_1 >> break_pos
    low1 = src_1 & mask
    high2 = src_2 >> break_pos
    low2 = src_2 & mask

    new_src_1 = (high1 << break_pos) | low2
    new_src_2 = (high2 << break_pos) | low1

    return new_src_1, new_src_2

def bit_toggle(src, toggle_pos):
    """
    将64位的toggle_pos位翻转

    Args:
        src (int64_t): 待处理二进制数
        toggle_pos (int64_t): 翻转位

    Returns:
        (int64_t): 反转后结果
    """
    src = int(src)
    if toggle_pos < 0 or toggle_pos > 63:
        raise ValueError("toggle_pos 必须在 0 到 63 之间")
    mask = 1 << toggle_pos
    return src ^ mask


