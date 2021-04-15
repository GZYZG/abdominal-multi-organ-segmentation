#!usr/bin/python
# 关于可视化的配置
import os
import argparse

"""
共13种器官＋背景
(0) 背景
(1) spleen 脾
(2) right kidney 右肾
(3) left kidney 左肾
(4) gallbladder 胆囊
(5) esophagus 食管
(6) liver 肝脏
(7) stomach 胃
(8) aorta 大动脉
(9) inferior vena cava 下腔静脉
(10) portal vein and splenic vein 门静脉和脾静脉
(11) pancreas 胰腺
(12) right adrenal gland 右肾上腺
(13) left adrenal gland 左肾上腺
"""

parser = argparse.ArgumentParser(description="arguments for visualization")


# arguments about path info
class VizConfig:
    pass


viz_config = parser.parse_args()
