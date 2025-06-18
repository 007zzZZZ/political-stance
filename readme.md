主要需要做的实验是



1 微调模型

2 调用微调模型定位神经元+评估微调模型政治倾向

3 做迁移神经元实验回答问题，评估政治倾向

4 使用抑制微调方法回答问题，包括做消融实验，评估政治倾向





## 1 微调模型

微调代码和脚本位置

finetune/ft.py

finetune/run_finetune.sh



## 2 定位神经元+获取和评估微调模型输出

一共三个数据集4个模型

**数据集1** 主要用于获取神经元，也要获取输出并评估

获取神经元、输出

political_stance/getNeuronDiff.py

political_stance/getNeuronDiff.sh

评估

在最后统一评估





**数据集2，3**, 只需要获取输出并评估即可

获取输出

political_stance/ftanswer.py

political_stance/ftanswer_dataset2_3.sh

评估：

在最后统一评估



## 3 activation patching实验

用激活修补拼接定位到的神经元到原始模型上，得到输出并进行评估

**数据集1** 

获取输出

political_stance/migrate.py

political_stance/migrate.sh

评估

在最后统一评估

**数据集2，3**

获取输出

political_stance/migrate2.py

political_stance/migrate2.sh

评估

在最后统一评估





## 4 re-fine-tune方法+消融实验+评估倾向

先使用re-fine-tune脚本进行抑制微调得到抑制微调模型，这一步同时做了消融实验，抑制不同比例的general neurons的梯度然后再进行微调

**微调代码+脚本**

finetune/inhibition_ft.py

finetune/run_inhibft.sh



然后调用微调模型**回答问题并评估**

获取输出

political_stance/inhibit_answer.py

political_stance/inhibit_answer.sh

评估

在最后统一评估



## 5 评估

所有产生的输出都位于

autodl-tmp/data2use/{model}/{adapter_topic}_right_lora/{topic}/目录下

评估脚本为

political_stance/eval_answer.py 评估数据集1的输出

political_stance/eval_answer2.py 评估数据集2和3的输出



但是需要把每个目录下的

specific_neurons.json都删掉，这个不是LLMs输出，可以把data2use文件夹复制一份，然后删掉所有的specific_neurons.json再进行评估









