# Assignment 5

## Q1. Classification Model

overall accuracy with best model: 0.9769

***Matched:***

**Chair:**

GT of index 0:

![gt_cls_idx0_1](/home/haoyus/16825/PS/assignment5/output/gt_cls_idx0_1.gif)

Pred of index 0:

![pred_cls_idx0_1](/home/haoyus/16825/PS/assignment5/output/pred_cls_idx0_1.gif)

**Vase**:

GT of index 645:

![gt_cls_idx_645](/home/haoyus/16825/PS/assignment5/output/gt_cls_idx_645.gif)

Pred of index 645:

![pred_cls_idx_645](/home/haoyus/16825/PS/assignment5/output/pred_cls_idx_645.gif)

**Lamp:**

GT of index 900:

![gt_cls_idx900](/home/haoyus/16825/PS/assignment5/output/gt_cls_idx900.gif)

Pred of index 900:

![pred_cls_idx900](/home/haoyus/16825/PS/assignment5/output/pred_cls_idx900.gif)

***Failure:***

GT of index 406:

![mismatch_exp_406_gt_chair](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_406_gt_chair.gif)

Pred of index 406:

![mismatch_exp_406_pred_lamp](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_406_pred_lamp.gif)

The correct class should be **chair**, but wrongly predicted as **lamp**.



GT of index 650:

![mismatch_exp_650_gt_vase](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_650_gt_vase.gif)

Pred of index 650:

![mismatch_exp_650_pred_lamp](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_650_pred_lamp.gif)

The correct class should be **vase**, but wrongly predicted as **lamp**.



GT of index 916:

![mismatch_exp_916_gt_lamp](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_916_gt_lamp.gif)

Pred of index 916:

![mismatch_exp_916_pred_vase](/home/haoyus/16825/PS/assignment5/output/mismatch_exp_916_pred_vase.gif)

The correct class should be **lamp**, but wrongly predicted as **vase**.

## Q2. Segmentation Model 

overall accuracy with best model: 0.8979

idx 176: accuracy: 96.65%

GT:

![gt_seg_misc_176](/home/haoyus/16825/PS/assignment5/output/gt_seg_misc_176.gif)

Pred:

![pred_seg_misc_176](/home/haoyus/16825/PS/assignment5/output/pred_seg_misc_176.gif)

idx 281: accuracy: 98.39%

GT:

![gt_seg_misc_281](/home/haoyus/16825/PS/assignment5/output/gt_seg_misc_281.gif)

Pred:

![pred_seg_misc_281](/home/haoyus/16825/PS/assignment5/output/pred_seg_misc_281.gif)

idx 344: accuracy: 90.92%

GT:

![gt_seg_leg_344](/home/haoyus/16825/PS/assignment5/output/gt_seg_leg_344.gif)

Pred:

![pred_seg_leg_344](/home/haoyus/16825/PS/assignment5/output/pred_seg_leg_344.gif)

idx 386: accuracy: 62.32%

GT:

![gt_seg_misc_386](/home/haoyus/16825/PS/assignment5/output/gt_seg_misc_386.gif)

Pred:

![pred_seg_misc_386](/home/haoyus/16825/PS/assignment5/output/pred_seg_misc_386.gif)

idx 605: accuracy: 57.45%

GT:

![gt_seg_misc_605](/home/haoyus/16825/PS/assignment5/output/gt_seg_misc_605.gif)

Pred:

![pred_seg_misc_605](/home/haoyus/16825/PS/assignment5/output/pred_seg_misc_605.gif)

## Q3. Robustness Analysis

I first try with rotating the poindcloud and run eval_cls.py and eval_seg.py.

I tried with angle 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180.

Here is how accuracy changes with rotation angle on classification and segmentation tasks.

![plot](/home/haoyus/16825/PS/assignment5/output/plot.png)

idx 899:

angle 0:

GT class: lamp

![gt_cls_idx899_0](/home/haoyus/16825/PS/assignment5/output/gt_cls_idx899_0.gif)

Pred class: lamp

![pred_cls_idx899_0](/home/haoyus/16825/PS/assignment5/output/pred_cls_idx899_0.gif)

angle 15:

GT class: lamp

![gt_cls_idx899_15](/home/haoyus/16825/PS/assignment5/output/gt_cls_idx899_15.gif)

Pred class: vase

![pred_cls_idx899_15](/home/haoyus/16825/PS/assignment5/output/pred_cls_idx899_15.gif)

