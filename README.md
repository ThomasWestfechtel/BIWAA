# BIWAA
PyTorch Code for BIWAA - Backprop Induced Feature Weighting for Adversarial Domain Adaptation with Iterative Label Distribution Alignment (WACV 2023)


### Method overview
The requirement for large labeled datasets is one of the limiting factors for training accurate deep neural networks. Unsupervised domain adaptation tackles this problem of limited training data by transferring knowledge from one domain, which has many labeled data, to a different domain for which little to no labeled data is available. One common approach is to learn domain-invariant features for example with an adversarial approach. Previous methods often train the domain classifier and label classifier network separately, where both classification networks have little interaction with each other. In this paper, we introduce a classifier based backprop induced weighting of the feature space. This approach has two main advantages. Firstly, it lets the domain classifier focus on features that are important for the classification and, secondly, it couples the classification and adversarial branch more closely. Furthermore, we introduce an iterative label distribution alignment method, that employs results of previous runs to approximate a class-balanced dataloader. We conduct experiments and ablation studies on three benchmarks Office-31, OfficeHome and DomainNet to show the effectiveness of our proposed algorithm.

![Alt text](figs/attention.jpg?raw=true "Attention")  
Heatmap of the features that are aligned in the adversarial network. The first row shows the original image. The second row shows the heatmap for DANN, while the last row shows the heatmap for our proposed backprop induced weighting method. The network is adapted on the task Art to RealWorld of the OfficeHome dataset. Without using the weighting, the adversarial network focuses on large parts of the image, including the background. The weighting lets the adversarial network focus mainly on the foreground object.

![Alt text](figs/pipeline-wbg.png?raw=true "Pipeline of method")  
Pipeline of our proposed method. We employ three losses to train the network. For the adversarial loss, we weight the feature space based on the importance for the classifier. In particular, we backpropagate the classification loss to the feature layer, normalize the gradients and employ it as a weighting vector. Furthermore, after training the network for a single, the predicted labels of the target domain are used to initialize dataloader of the next run to achieve a class-balanced dataloader.

### Usage
To train the network run:

```shell
bash train_office31.sh 
bash train_officeHome.sh 
bash train_domainNet.sh 
```

To train different a task change the line 12 of the scripts


### Citation
If you use BIWAA code please cite:
```text
@article{biwaa2023, 
    title={Backprop Induced Feature Weighting for Adversarial Domain Adaptation with Iterative Label Distribution Alignment.}, 
    author={Thomas Westfechtel and Hao-Wei Yeh and Qier Meng and Yusuke Mukuta and Tatsuya Harada},
    journal={Winter Conference on Applications of Computer Vision (WACV)},
    year={2023}
}
```
