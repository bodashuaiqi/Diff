## Requirement 先装环境

``conda env create -f environment.yml``

## Datasets 
数据集HAM10000对应isic，APTOS2019就是aptos
如果需要下载，请参考原README

## Run
2. For Training! run: ``bash training_scripts/run_isic.sh`` where the first command line is used ``python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}``

训练用： 
python main.py --config 'path to aptos.yml' --doc '选择aptos or isic' --loss diffmic_conditional

3. For Testing! run: ``bash training_scripts/run_isic.sh`` where the second command line is used ``python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --test --eval_best``

测试用：
python main.py --config DiffMIC/exp/logs/ --doc aptos --loss diffmic_conditional --test --eval_best
同样，doc那里根据你实际用的数据集来

我写了个inference.py，便于你做展示。使用方法写在那个文件里了。请自行阅读

写的十分详细了，给个好评吧😉