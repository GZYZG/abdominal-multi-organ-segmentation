import enum


ORGAN_MAPPING = {0: '背景', 1: 'spleen 脾', 2: 'right kidney 右肾', 3: 'left kidney 左肾',
                 4: 'gallbladder 胆囊', 5: 'esophagus 食管', 6: 'liver 肝脏', 7: 'stomach 胃',
                 8: 'aorta 大动脉', 9: 'inferior vena cava 下腔静脉', 10: 'portal vein and splenic vein 门静脉和脾静脉', 
                 11: 'pancreas 胰腺', 12: 'right adrenal gland 右肾上腺', 13: 'left adrenal gland 左肾上腺'
}

# 并不一次性分割所有器官。将器官划分为几个组，为每个组训练一个模型
VISIBLES = [
    None,
    [1, 2, 3, 6, 7, 8, 9],
    [4, 5, 10, 11, 12, 13]
]