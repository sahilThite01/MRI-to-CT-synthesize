# MRI to CT synthesize

This repository explores multiple deep learning pipelines (Pix2Pix, improved GAN variants) to **synthesize CT images from MRI scans**, with a focus on clinical-quality reconstruction and robust evaluation metrics.

Built a Deep Learning based model to synthesize clinical quality CT images from Brain MRI data. Applied image normalization, cropping, augmentation, train-test preparation, and contrast adjustments during preprocessing. Tracked training metrics using SSIM and GAN loss; used PatchGAN discriminator(16×16).

The work is organized as a collection of **Jupyter notebooks** that incrementally improve the model design, data preprocessing, and training strategy.

---

# Generative Adversarial Networks (GANs)

**Generative Adversarial Networks (GANs)** are a class of deep learning models used to generate *new, realistic-looking data*.  
They are widely used in image synthesis, medical imaging, face generation, super-resolution, and domain translation tasks such as **MRI → CT**. <br>



## GAN Architecture Overview

A GAN consists of two neural networks:

1. **Generator (G)**  
  Produces fake samples from random noise.

2. **Discriminator (D)**  
  Evaluates whether input samples are real or fake.
  
<br>

🔹 **Generator (G):**
<br>The Generator is the model responsible for creating synthetic images.  
It starts with random noise (or sometimes a latent vector) and learns how to produce outputs that resemble real data.

**Key Points:**
- Takes **noise** as input and outputs **fake images**
- Learns to mimic the distribution of real images
- Tries to **fool the discriminator**
- Improves through feedback from D's classification

**Goal:**  
Generate images that look real enough to trick the discriminator.

<br>

🔸 **Discriminator (D):**
<br>The Discriminator is a classifier that decides whether an input image is real or fake.

**Key Points:**
- Takes **real images and generated images** as input
- Outputs a **probability** (real vs fake)
- Penalizes the generator when fake images are detected
- Learns meaningful features for realism detection

**Goal:**  
Correctly distinguish real images from generator-produced fake ones.

<br>

⚔️ **Adversarial Learning (Competition):**  
- G learns to produce better (more realistic) images <br>
- D learns to become better at detection <br>
- This adversarial process continues until the generator produces highly realistic outputs

Together, they form the **Generative Adversarial Network (GAN)** —  a powerful framework for realistic synthetic data generation.


---

## System Architectures

- Model Architecture
<img width="607" height="259" alt="image" src="https://github.com/user-attachments/assets/63963502-95df-4204-9f96-adb2e97b5d0e" />


- Pix2Pix Architecture
<img width="464" height="308" alt="image" src="https://github.com/user-attachments/assets/39f04b14-de11-4d54-9368-4a57f232d396" />

---

## 📂 Repository Structure

Key components in this repo: (`./models/`)

- `CycleGAN_MK1.ipynb`, `CycleGAN_MK2.ipynb`  
  Initial experiments with unpaired MRI↔CT translation using CycleGAN.

- `MK1_MRI-CT.ipynb`, `MK2_MRI-CT.ipynb`  
  Early Pix2Pix-style paired translation baselines.

- `MK3_*_MRI_CT.ipynb`, `MK3.1.2_GOD_MRI_CT.ipynb`, `MK3.1.2_v4_*.ipynb`, `MK3.5_*`, `MK3.6_*`  
  Progressive improvements to the Pix2Pix-style model, loss functions, and training setup.

- `MK6.*_CP*.ipynb`  
  Later **checkpointed** versions with improved preprocessing, augmentation, and loss weighting – these yield the best-performing models so far (see metrics below).

- `Preprocess_&_mask.ipynb`, `Mask_creator.ipynb`  
  Notebooks for preprocessing MRI/CT slices, generating masks, and preparing data for training.

- `practice_brain_example/`  
  Example pipeline/notebooks on a brain MRI use case, useful as a small sandbox to understand the workflow.

---

## Dataset

We use a publicly available paired MRI–CT dataset:

- **Dataset link:**  
  https://zenodo.org/records/7260705  

The dataset provides **paired MRI and CT slices**, enabling supervised image-to-image translation.

---

## Pretrained Models

Pretrained models and checkpoints are provided on Google Drive:

- **Models link:**  
  [https://drive.google.com/drive/u/1/folders/1mtTygiVIjpb8gmSn3h2RKz272qI7S2-i ](https://drive.google.com/drive/u/0/folders/1W_7ok_IFitLeMfKiPBpRiSAXytxv2YxH) 

Each model folder typically contains:

- Generator + Discriminator weights  
- Training/evaluation metrics (MSE, SSIM, FID, VGG/Perceptual Loss, etc.)  
- Sample qualitative results for visual inspection  

---

## Getting Started

1. **Clone the repository**

    git clone https://github.com/RajatRajSharma/MRI_to_CT_scan.git

    cd MRI_to_CT_scan

3. **Set up a Python environment**

   Use your preferred environment manager (conda/venv).

   Example:

    conda create -n mri_to_ct python=3.10
    conda activate mri_to_ct

4. **Install dependencies**

   Install standard deep learning & image-processing libraries (PyTorch, torchvision, numpy, matplotlib, scikit-image, etc.):

    pip install torch torchvision numpy matplotlib scikit-image tqdm opencv-python

   (You can adjust this list based on the imports used in each notebook.)

5. **Download dataset**

   - Download the dataset from Zenodo:  
     https://zenodo.org/records/7260705  
   - Arrange the MRI–CT pairs as expected by the preprocessing notebooks (`Preprocess_&_mask*.ipynb`).  
   - Update any dataset paths inside the notebooks as needed.

6. **Download pretrained models (optional)**

   - Fetch checkpoints from the Google Drive link.  
   - Place them in a directory of your choice (e.g., `./checkpoints/`) and update the relevant paths in the notebooks for inference-only runs.

---

## 📈 Models – Code Flow & Metrics

Below is a summary of the **key notebook versions** and their performance metrics.

### Best Model: MK6.4.3_CP3
  - MSE: 0.008025
  - SSIM: 0.917649 (~92%)  
  - FID: 60.405193 
  - VGG Loss: 0.864183 

<br>

### MK3.1.x Series

- **MK3.1.2_GOD_MRI_CT.ipynb** (17/01/2025)  
  - MSE: 0.063136  
  - SSIM: 0.449328  
  - FID: 221.746994  
  - VGG Loss: 2.912386  

- **MK3.1.2_v4_50epochs_GOD.ipynb** (24/04/2025)  
  - MSE: 0.072253  
  - SSIM: 0.702594  
  - FID: 174.542770  
  - VGG Loss: 2.671596  


## ✅ Check Point 1 (26/05/2025)

- **MK6.0.1_CP1**  
  - MSE: 0.065860  
  - SSIM: 0.678714  
  - FID: 175.659180  
  - VGG Loss: 2.517590  

- **MK6.0.3_CP1**  
  - MSE: 0.056884  
  - SSIM: 0.591850  
  - FID: 166.765396  
  - VGG Loss: 2.724192  

- **MK6.1.0_CP1** (30/05/2025)  
  - MSE: 0.019255  
  - SSIM: 0.825634  
  - FID: 113.451118  
  - VGG Loss: 1.631517  

- **MK6.1.1_CP1** (31/05/2025)  
  - MSE: 0.019586  
  - SSIM: 0.813731  
  - FID: 118.909042  
  - VGG Loss: 1.640784  


## ✅ Check Point 2 (01/06/2025)

- **MK6.2.0_CP2**  
  - MSE: 0.018466  
  - SSIM: 0.823152  
  - FID: 113.933151  
  - VGG Loss: 1.602489  

- **MK6.2.1_CP2**  
  - MSE: 0.015199  
  - SSIM: 0.852237  
  - FID: 104.621910  
  - VGG Loss: 1.428501  


## ✅ Check Point 3 (05/06/2025)

- **MK6.3.0_CP3**  
  - MSE: 0.013473  
  - SSIM: 0.866290  
  - FID: 97.263969  
  - VGG Loss: 1.324233  

- **MK6.3.1_CP3**  
  - MSE: 0.012631  
  - SSIM: 0.875822  
  - FID: 90.494118  
  - VGG Loss: 1.250894  

These later checkpoints (`MK6.x`) incorporate improved preprocessing, better augmentation, refined loss balancing, and updated architectural tweaks, making them the preferred choices for inference and comparison.

---

## Design Choices & Ideas

- Two types of **data augmentation** specifically tailored for paired MRI–CT slices.  
- Updated transform functions that act **jointly** on MRI–CT pairs, ensuring spatial alignment is preserved:

    mri_image, ct_image = self.transform(mri_image, ct_image)

- Conversion from **3-channel to 1-channel** (since MRI/CT images are naturally grayscale).  
- Intentional limit on training data size (at most ~1000 paired images) when additional data does not add new variation or cases.  
- Data collected from **3 different centers**, aiming to improve generalization across scanners and acquisition protocols.

---

## Potential Applications

- Synthetic CT generation for radiotherapy planning from MRI-only workflows  
- Orthopedic analysis (e.g., knee joint evaluation) when CT is not available  
- Reducing radiation exposure by replacing CT acquisitions with MRI + synthetic CT  
- Multi-center robustness studies on MRI–CT domain translation  

---

## Contributions

- Feel free to open **issues** or **pull requests** for:
  - New architectures (e.g., diffusion models, 3D GANs)  
  - Better preprocessing or augmentation pipelines  
  - Additional metrics or evaluation scripts  
  - Documentation improvements (especially in `/docs`)

If you use this repository or its ideas in your research, please consider:

- Citing the **dataset** from Zenodo  
- Adding a reference to this GitHub repository in your work

---

# 📬 Contact

If you found this project helpful, feel free to connect!

- **LinkedIn:** [<img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/7f83a5c7-d664-41c4-8d3b-81522ba5e4de" />](https://www.linkedin.com/in/sahil-thite-2582a9231/)  
- **GitHub:** *[<img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/5dccb824-84f2-4f39-bbb4-9accff21662d" />
](https://github.com/sahilThite01)*  
- **Email:** [thitesahil100@gmail.com](mailto:thitesahil100@gmail.com)

---

### ⭐ If you like this project, don’t forget to star the repo!
