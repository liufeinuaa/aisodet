# 初始设置时候，对原始dotav1数据集进行的分割处理 (分割成重叠率为200pixle的1024x1024的img)
python tools/data/dota/img_split.py --base-json tools/data/dota/ss_trainval.json
python tools/data/dota/img_split.py --base-json tools/data/dota/ss_val.json
python tools/data/dota/img_split.py --base-json tools/data/dota/ss_test.json



