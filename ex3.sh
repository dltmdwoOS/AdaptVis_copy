export TEST_MODE=False

python3 main_aro.py \
    --dataset VG_QA_two_obj \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_entropy \
    --weight1 0.5 \
    --weight2 1.2 \
    --threshold 0.9 \
    --option=six

# python3 main_aro.py \
#     --dataset VG_QA_two_obj \
#     --model-name='llava1.5' \
#     --download \
#     --method adapt_vis_entropy \
#     --weight1 0.5 \
#     --weight2 1.2 \
#     --threshold 0.96 \
#     --option=four