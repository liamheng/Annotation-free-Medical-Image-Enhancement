# -*- coding: UTF-8 -*-
"""
@Function:
@File: find_gpu.py
@Date: 2022/1/3 21:49 
@Author: Hever
"""
import os
import time

def find_gpu(allocate_id, need_memory=4000):
    print('need memory {}, finding gpu...'.format(need_memory))
    first = True
    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        # 若已经分配了gpu_id且gpu有足够的内存，则可以直接开始
        if first and idx_freeMemory_pair[allocate_id][1] > need_memory:
            print('Using gpu id: {}. {}'.format(allocate_id, time.ctime()))
            return allocate_id
        idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        if idx_freeMemory_pair[0][1] > need_memory:
            gpu_id = idx_freeMemory_pair[0][0]
            # 第一次检查到可以直接开始
            if first:
                print('Using gpu id: {}. {}'.format(gpu_id, time.ctime()))
                return gpu_id
            if double_check(gpu_id, need_memory):
                print('Using gpu id: {}. {}'.format(gpu_id, time.ctime()))
                return gpu_id
        first = False
        time.sleep(10)


def double_check(gpu_id, need_memory=4000):
    # 两次确定没有人使用
    for i in range(2):
        time.sleep(20)
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        for p in idx_freeMemory_pair:
            if p[0] == gpu_id and p[1] > need_memory:
                print('gpu {} double check is ok.'.format(gpu_id))
                return True
    print('gpu {} double check is fail.'.format(gpu_id))
    return False

