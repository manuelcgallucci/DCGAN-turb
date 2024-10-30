# DCGAN-turb     
Implementation of dense convolutional GANs to generate 1D turbulent velocity signals using a multi discriminator approach.

<div style="display: flex; justify-content: space-between;">
    <a href="https://iopscience.iop.org/journal/2632-2153" target="_blank"><img src="https://img.shields.io/badge/Visit-Journal-informational?style=flat-square&logoColor=white&color=blue" alt="Visit Journal" height="20"></a>
    <a href="https://github.com/manuelcgallucci/DCGAN-turb" target="_blank"><img src="https://img.shields.io/badge/View-Publication-informational?style=flat-square&logoColor=white&color=green" alt="View Publication" height="20"></a>
</div>


# Model usage

# Results
Here are shown the structure function s2, flatness and skewness of the real and the generated samples. As we observe the results are really good in the inertial and integral domains but the results are not yet optimal in the dissipative domain. 
<p align="center">
  <img src="images/samples_fake.png" alt="My Image 1" style="display:inline-block;width:39%;">
  <img src="images/samples_real.png" alt="My Image 2" style="display:inline-block;width:40%;">
</p>
<p align="center">
  <em>Real samples (left) and fake generated samples (right)</em>
</p>
<p align="center">
  <img src="images/comparison_s2.png" alt="My Image 1" style="display:inline-block;width:40%;">
  <img src="images/comparison_flatness.png" alt="My Image 2" style="display:inline-block;width:40%;">
</p>
<p align="center">
  <img src="images/comparison_skewness.png" alt="My Image 1" style="display:inline-block;width:40%;">
  <img src="images/histogram.png" alt="My Image 2" style="display:inline-block;width:30%;">
</p>
<p align="center">
  <em>Structure functions and histogram of the increments</em>
</p>

# Training scheme 

The generator used is a dense convolutional Unet with 5 residual connections. The discriminators were 3 dense nets for the s2, skewness and flatness and one big CNN composed of three individual parts for different sections of the same sample for the scales discriminator. 
<p align="center">
  <img src="images/training_scheme.png" alt="My Image 1" style="display:inline-block;width:49%;">
</p>
<p align="center">
  <img src="images/discriminator_scales.png" alt="My Image 1" style="display:inline-block;width:50%;">
</p>

# Loss evolution

<p align="center">
  <img src="images/loss_generator.png" alt="My Image 1" style="display:inline-block;width:45%;">
  <img src="images/loss_discriminator.png" alt="My Image 2" style="display:inline-block;width:45%;">
</p>

<p align="center">
  <img src="images/loss_discriminator_fake.png" alt="My Image 1" style="display:inline-block;width:45%;">
  <img src="images/loss_discriminator_real.png" alt="My Image 2" style="display:inline-block;width:45%;">
</p>

<p align="center">
  <img src="images/loss_discriminator_structures.png" alt="My Image 1" style="display:inline-block;width:45%;">
  <img src="images/losses.png" alt="My Image 1" style="display:inline-block;width:45%;">
</p>

