# Cost-SensitiveACGAN

Dissertation by Athirah Hazwani for Master of Computer Science.

## Abstract

The class imbalance problem has been recognized in many real-world applications and negatively affects machine learning performance. Generative Adversarial Networks or GANs have been known to be the next best thing in image generation. However, most GANs do not consider classes and when they do, cannot perform well under the imbalance problem. Based on related works, the modification of the loss function and various resampling methods have been commonly applied to counter the problem of class imbalance. In this study, CostSensitive-ACGAN is introduced which is a variation of Auxiliary Classifier GAN (ACGAN) that can work better under the class imbalance condition. This method incorporated the idea of applying cost-sensitive learning in the loss function to further improve the classification of minority classes. Cost-sensitive parameters are determined adaptively according to the classification error of the class to improve minority classes presence. By applying higher misclassification costs for minority classes, these instances can be magnified and recognized by the discriminator thus improving image generation altogether. This method has shown comparatively competitive results with existing benchmark model.

https://www.jatit.org/volumes/Vol101No12/19Vol101No12.pdf
