export TEST_MODE=True

python3 main_aro.py \
    --dataset VG_QA_one_obj \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties_var_weights \
    --weight1 0.5 \
    --weight2 2.0 \
    --threshold 0.2 \
    --option=six