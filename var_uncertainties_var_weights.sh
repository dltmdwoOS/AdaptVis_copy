export TEST_MODE=True

python3 main_aro.py \
    --dataset COCO_QA_one_obj \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties_var_weights \
    --weight1 0.5 \
    --weight2 1.2 \
    --threshold 0.3 \
    --option=four