# BIWAA
PyTorch Code for BIWAA - Backprop Induced Feature Weighting for Adversarial Domain Adaptation with Iterative Label Distribution Alignment (WACV 2023)


### Method overview
<img src="figs/211111-attention.png alt="attention" width="800"/>

<img src="figs/211116_pipeline.png alt="pipeline" width="800"/>

The requirement for large labeled datasets is one of the limiting factors for training accurate deep neural networks. Unsupervised domain adaptation tackles this problem of limited training data by transferring knowledge from one domain, which has many labeled data, to a different domain for which little to no labeled data is available. One common approach is to learn domain-invariant features for example with an adversarial approach. Previous methods often train the domain classifier and label classifier network separately, where both classification networks have little interaction with each other. In this paper, we introduce a classifier based backprop induced weighting of the feature space. This approach has two main advantages. Firstly, it lets the domain classifier focus on features that are important for the classification and, secondly, it couples the classification and adversarial branch more closely. Furthermore, we introduce an iterative label distribution alignment method, that employs results of previous runs to approximate a class-balanced dataloader. We conduct experiments and ablation studies on three benchmarks Office-31, OfficeHome and DomainNet to show the effectiveness of our proposed algorithm.

To train the network run:

bash train_domainNet.sh
bash train_office31.sh
bash train_officeHome.sh

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
