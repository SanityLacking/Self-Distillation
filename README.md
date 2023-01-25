# Self Distillation
<hr>
This Repository contains the code for the Conference Paper: 
>**Self Distillation: Be your own Teacher** <br>

When building lightweight Deep learning models, The trade off between accuracy and efficiency is the paramount concern. This repository demonstrates how using a combination of early exiting and knowledge distillation methods, it is possible to improve both efficiency and accuracy at the same time.<br>

<img src="https://user-images.githubusercontent.com/4435648/214571697-2c8adfe1-361a-41ab-adfc-68a8af7a4824.png" width=50% height=50%><img src="https://user-images.githubusercontent.com/4435648/214572602-6aa12158-f820-4f3a-ae49-005afae719e0.png" width=50% height=50%>  <br>
<div align="center">

*Knowledge Distillation  Vs Self Distillation*<br></div>
<hr>
In Delf Distillation, The main exit of a model acts as a teacher for the branch exits that are added earlier to the model's structure and trained. By using the knowledge of the main exit as a teaching source, the branches accuracy improves, and the branch predictions can be used as prediction outputs, reducing the overall processing time needed per input.<br>

<div align="center"><img src="https://user-images.githubusercontent.com/4435648/214572200-62d118fd-f93d-4225-9114-ec2502ce4671.png" width=50% height=50%><br>

*Self Distillation improves accuracy across a range of DNN model structures, even already very lightweight model designs such as squeezenet and efficientnet*
<br></div>
<div align="center"><img src="https://user-images.githubusercontent.com/4435648/214572208-0529573c-2c1d-4eda-9d57-f4622d2d3cbe.png" width=50% height=50%> <br>


*Self Distilling also improves the accuracy of the original branch, meaning that this training process is beneficial even for non-branching models.*<br></div>

<hr>


Source files are located in folder branching, with the class model defined in branching\core_v2.py


To run, go to notebooks and run jupyter notebook self_distillation.ipynb. This notebook contains a working example of the branching process and self distillation of a branched model. <br>
in the "branch the model" section, "replace models/resnet_CE_entropy_finetuned.hdf5" with the model you wish to branch, <br>
in the add_branch or add_distill function identify the branch structure in branch_layers (*_branch_Distill* and *_branch_conv1* are predefined)<br>
select the branch points in branch_points ("conv2_block1_out","conv2_block3_out" should work for any resNet model, otherwise change this name to the layer name reported by tensorflow using "model.summary()" ) <br>
Provide an individual loss for each exit, in our example this means 3 losses. the first loss is the main exit. <br>
Only one optimizer per model is currently enabled <Br>

