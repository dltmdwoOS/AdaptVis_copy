export TEST_MODE=True

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties \
    --weight1 0.5 \
    --weight2 1.5 \
    --threshold 0.4 \
    --option=four

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties \
    --weight1 0.5 \
    --weight2 2.0 \
    --threshold 0.4 \
    --option=four

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties \
    --weight1 0.8 \
    --weight2 1.2 \
    --threshold 0.4 \
    --option=four

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties \
    --weight1 0.8 \
    --weight2 1.5 \
    --threshold 0.4 \
    --option=four

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_var_uncertainties \
    --weight1 0.8 \
    --weight2 2.0 \
    --threshold 0.4 \
    --option=four

