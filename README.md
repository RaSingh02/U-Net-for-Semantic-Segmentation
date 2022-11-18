# U-Net-for-Semantic-Segmentation

Was tasked to add down, up, & bottleneck blocks to the UNet function. This model generates approximate masks for pets located in images sourced 
from the "oxford_iiit_pet" TensorFlow dataset. In the end, the model is not very great at generating masks compared to the "true mask" of the pet. 
Even with 50 epochs, the model hit a plateau at around 85% validation accuracy. However, with 50 epochs, testing accuracy was seen to be as high as 94%.
