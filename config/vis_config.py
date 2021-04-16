#!usr/bin/python
# 关于可视化的配置
import os
import argparse
from collections import OrderedDict
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
    organs = OrderedDict({0: 'background', 1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder', 5: 'esophagus',
              6: 'liver', 7: 'stomach', 8: 'aorta', 9: 'inferior_vena_cava', 10: 'portal_vein_and_splenic_vein',
              11: 'pancreas', 12: 'right_adrenal_gland', 13: 'left_adrenal_gland'})
    colors = [(1, 1, 1, 1), (1, 170/255, 1, 1), (1, 85/255, 1, 1), (1, 0, 1, 1), (170/255, 170/255, 0, 1), (0, 170/255, 0, 1),
              (225/255, 110/255, 100/255, .25), (1, 170/255, 127/255, 1), (1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1),
              (40, 160, 1, 1), (1, 200/255, 0, 1), (1, 150/255, 0, 1)]
    init_organ_colors = dict(zip(organs.keys(), colors))


viz_config = parser.parse_args()
