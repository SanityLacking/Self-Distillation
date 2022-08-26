# Self Distillation
<hr>
Knowledge distillation within a branching model. <br>
The main exit acts as a teacher for the branch exits during branch training, improving the branch accuracy <br>

Source files are located in folder branching, with the class model defined in branching\core_v2.py


To run, go to notebooks and run jupyter notebook self_distillation.ipynb. This notebook contains a working example of the branching process and self distillation of a branched model. <br>
in the "branch the model" section, "replace models/resnet_CE_entropy_finetuned.hdf5" with the model you wish to branch, <br>
in the add_branch or add_distill function identify the branch structure in branch_layers (*_branch_Distill* and *_branch_conv1* are predefined)<br>
select the branch points in branch_points ("conv2_block1_out","conv2_block3_out" should work for any resNet model, otherwise change this name to the layer name reported by tensorflow using "model.summary()" ) <br>
Provide an individual loss for each exit, in our example this means 3 losses. the first loss is the main exit. <br>
Only one optimizer per model is possible <Br>


