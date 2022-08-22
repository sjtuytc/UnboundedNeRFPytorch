
Weekly Classified Neural Radiance Fields - reconstruction ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
==========================================================================================================================================================================
## Filter by classes: 
 [all](../weekly_nerf.md) | [dynamic](./dynamic.md) | [editing](./editing.md) | [fast](./fast.md) | [generalization](./generalization.md) | [human](./human.md) | [video](./video.md) | [lighting](./lighting.md) | [reconstruction](./reconstruction.md) | [texture](./texture.md) | [semantic](./semantic.md) | [pose-slam](./pose-slam.md) | [others](./others.md) 
## Aug14 - Aug20, 2022
## Aug7 - Aug13, 2022
  - [OmniVoxel: A Fast and Precise Reconstruction Method of Omnidirectional Neural Radiance Field, GCCE 2022](https://arxiv.org/abs/2208.06335) | [code]
    > This paper proposes a method to reconstruct the neural radiance field with equirectangular omnidirectional images. Implicit neural scene representation with a radiance field can reconstruct the 3D shape of a scene continuously within a limited spatial area. However, training a fully implicit representation on commercial PC hardware requires a lot of time and computing resources (15 ∼ 20 hours per scene). Therefore, we propose a method to accelerate this process significantly (20 ∼ 40 minutes per scene). Instead of using a fully implicit representation of rays for radiance field reconstruction, we adopt feature voxels that contain density and color features in tensors. Considering omnidirectional equirectangular input and the camera layout, we use spherical voxelization for representation instead of cubic representation. Our voxelization method could balance the reconstruction quality of the inner scene and outer scene. In addition, we adopt the axis-aligned positional encoding method on the color features to increase the total image quality. Our method achieves satisfying empirical performance on synthetic datasets with random camera poses. Moreover, we test our method with real scenes which contain complex geometries and also achieve state-of-the-art performance. Our code and complete dataset will be released at the same time as the paper publication.
  - [Fast Gradient Descent for Surface Capture Via Differentiable Rendering, 3DV2022](https://hal.inria.fr/hal-03748662/) | [code]
    > Differential rendering has recently emerged as a powerful tool for image-based rendering or geometric reconstruction from multiple views, with very high quality. Up to now, such methods have been benchmarked on generic object databases and promisingly applied to some real data, but have yet to be applied to specific applications that may benefit. In this paper, we investigate how a differential rendering system can be crafted for raw multi-camera performance capture. We address several key issues in the way of practical usability and reproducibility, such as processing speed, explainability of the model, and general output model quality. This leads us to several contributions to the differential rendering framework. In particular we show that a unified view of differential rendering and classic optimization is possible, leading to a formulation and implementation where complete non-stochastic gradient steps can be analytically computed and the full perframe data stored in video memory, yielding a straightforward and efficient implementation. We also use a sparse storage and coarse-to-fine scheme to achieve extremely high resolution with contained memory and computation time. We show experimentally that results rivaling in quality with state of the art multi-view human surface capture methods are achievable in a fraction of the time, typically around a minute per frame.
  - [PlaneFormers: From Sparse View Planes to 3D Reconstruction, ECCV2022](https://arxiv.org/abs/2208.04307) | [code]
    > We present an approach for the planar surface reconstruction of a scene from images with limited overlap. This reconstruction task is challenging since it requires jointly reasoning about single image 3D reconstruction, correspondence between images, and the relative camera pose between images. Past work has proposed optimization-based approaches. We introduce a simpler approach, the PlaneFormer, that uses a transformer applied to 3D-aware plane tokens to perform 3D reasoning. Our experiments show that our approach is substantially more effective than prior work, and that several 3D-specific design decisions are crucial for its success.
  - [PS-NeRV: Patch-wise Stylized Neural Representations for Videos](https://arxiv.org/abs/2208.03742) | [code]
    > We study how to represent a video with implicit neural representations (INRs). Classical INRs methods generally utilize MLPs to map input coordinates to output pixels. While some recent works have tried to directly reconstruct the whole image with CNNs. However, we argue that both the above pixel-wise and image-wise strategies are not favorable to video data. Instead, we propose a patch-wise solution, PS-NeRV, which represents videos as a function of patches and the corresponding patch coordinate. It naturally inherits the advantages of image-wise methods, and achieves excellent reconstruction performance with fast decoding speed. The whole method includes conventional modules, like positional embedding, MLPs and CNNs, while also introduces AdaIN to enhance intermediate features. These simple yet essential changes could help the network easily fit high-frequency details. Extensive experiments have demonstrated its effectiveness in several video-related tasks, such as video compression and video inpainting.
## Jul31 - Aug6, 2022
  - [PRIF: Primary Ray-based Implicit Function](https://research.google/pubs/pub51556/) | [code]
    > We introduce a new implicit shape representation called Primary Ray-based Implicit Function (PRIF). In contrast to most existing approaches based on the signed distance function (SDF) which handles spatial locations, our representation operates on oriented rays. Specifically, PRIF is formulated to directly produce the surface hit point of a given input ray, without the expensive sphere-tracing operations, hence enabling efficient shape extraction and differentiable rendering. We demonstrate that neural networks trained to encode PRIF achieve successes in various tasks including single shape representation, category-wise shape generation, shape completion from sparse or noisy observations, inverse rendering for camera pose estimation, and neural rendering with color.
## Jul24 - Jul30, 2022
  - [Going Off-Grid: Continuous Implicit Neural Representations for 3D Vascular Modeling, MICCAI STACOM 2022](https://arxiv.org/abs/2207.14663) | [code]
    > Personalised 3D vascular models are valuable for diagnosis, prognosis and treatment planning in patients with cardiovascular disease. Traditionally, such models have been constructed with explicit representations such as meshes and voxel masks, or implicit representations such as radial basis functions or atomic (tubular) shapes. Here, we propose to represent surfaces by the zero level set of their signed distance function (SDF) in a differentiable implicit neural representation (INR). This allows us to model complex vascular structures with a representation that is implicit, continuous, light-weight, and easy to integrate with deep learning algorithms. We here demonstrate the potential of this approach with three practical examples. First, we obtain an accurate and watertight surface for an abdominal aortic aneurysm (AAA) from CT images and show robust fitting from as little as 200 points on the surface. Second, we simultaneously fit nested vessel walls in a single INR without intersections. Third, we show how 3D models of individual arteries can be smoothly blended into a single watertight surface. Our results show that INRs are a flexible representation with potential for minimally interactive annotation and manipulation of complex vascular structures.
  - [GAUDI: A Neural Architect for Immersive 3D Scene Generation](https://arxiv.org/abs/2207.13751) | [***``[code]``***](https://github.com/apple/ml-gaudi)
    > We introduce GAUDI, a generative model capable of capturing the distribution of complex and realistic 3D scenes that can be rendered immersively from a moving camera. We tackle this challenging problem with a scalable yet powerful approach, where we first optimize a latent representation that disentangles radiance fields and camera poses. This latent representation is then used to learn a generative model that enables both unconditional and conditional generation of 3D scenes. Our model generalizes previous works that focus on single objects by removing the assumption that the camera pose distribution can be shared across samples. We show that GAUDI obtains state-of-the-art performance in the unconditional generative setting across multiple datasets and allows for conditional generation of 3D scenes given conditioning variables like sparse image observations or text that describes the scene.
  - [AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction, ECCV2022](https://arxiv.org/abs/2207.12909) | [***``[code]``***](https://zerchen.github.io/projects/alignsdf.html)
    > Recent work achieved impressive progress towards joint reconstruction of hands and manipulated objects from monocular color images. Existing methods focus on two alternative representations in terms of either parametric meshes or signed distance fields (SDFs). On one side, parametric models can benefit from prior knowledge at the cost of limited shape deformations and mesh resolutions. Mesh models, hence, may fail to precisely reconstruct details such as contact surfaces of hands and objects. SDF-based methods, on the other side, can represent arbitrary details but are lacking explicit priors. In this work we aim to improve SDF models using priors provided by parametric representations. In particular, we propose a joint learning framework that disentangles the pose and the shape. We obtain hand and object poses from parametric models and use them to align SDFs in 3D space. We show that such aligned SDFs better focus on reconstructing shape details and improve reconstruction accuracy both for hands and objects. We evaluate our method and demonstrate significant improvements over the state of the art on the challenging ObMan and DexYCB benchmarks.
  - [NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing, ECCV2022(oral)](https://arxiv.org/abs/2207.11911) | [code]
    > Very recently neural implicit rendering techniques have been rapidly evolved and shown great advantages in novel view synthesis and 3D scene reconstruction. However, existing neural rendering methods for editing purposes offer limited functionality, e.g., rigid transformation, or not applicable for fine-grained editing for general objects from daily lives. In this paper, we present a novel mesh-based representation by encoding the neural implicit field with disentangled geometry and texture codes on mesh vertices, which facilitates a set of editing functionalities, including mesh-guided geometry editing, designated texture editing with texture swapping, filling and painting operations. To this end, we develop several techniques including learnable sign indicators to magnify spatial distinguishability of mesh-based representation, distillation and fine-tuning mechanism to make a steady convergence, and the spatial-aware optimization strategy to realize precise texture editing. Extensive experiments and editing examples on both real and synthetic data demonstrate the superiority of our method on representation quality and editing ability. Code is available on the project webpage: this https URL.
## Previous weeks
  - [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video,, ICCV2021](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/) | [***``[code]``***](https://github.com/facebookresearch/nonrigid_nerf)
    > We present Non-Rigid Neural Radiance Fields (NR-NeRF), a reconstruction and novel view synthesis approach for general non-rigid dynamic scenes. Our approach takes RGB images of a dynamic scene as input (e.g., from a monocular video recording), and creates a high-quality space-time geometry and appearance representation. We show that a single handheld consumer-grade camera is sufficient to synthesize sophisticated renderings of a dynamic scene from novel virtual camera views, e.g. a `bullet-time' video effect. NR-NeRF disentangles the dynamic scene into a canonical volume and its deformation. Scene deformation is implemented as ray bending, where straight rays are deformed non-rigidly. We also propose a novel rigidity network to better constrain rigid regions of the scene, leading to more stable results. The ray bending and rigidity network are trained without explicit supervision. Our formulation enables dense correspondence estimation across views and time, and compelling video editing applications such as motion exaggeration. Our code will be open sourced.
  - [Neural Articulated Radiance Field, ICCV2021](https://arxiv.org/abs/2104.03110) | [***``[code]``***](https://github.com/nogu-atsu/NARF#code)
    > We present Neural Articulated Radiance Field (NARF), a novel deformable 3D representation for articulated objects learned from images. While recent advances in 3D implicit representation have made it possible to learn models of complex objects, learning pose-controllable representations of articulated objects remains a challenge, as current methods require 3D shape supervision and are unable to render appearance. In formulating an implicit representation of 3D articulated objects, our method considers only the rigid transformation of the most relevant object part in solving for the radiance field at each 3D location. In this way, the proposed method represents pose-dependent changes without significantly increasing the computational complexity. NARF is fully differentiable and can be trained from images with pose annotations. Moreover, through the use of an autoencoder, it can learn appearance variations over multiple instances of an object class. Experiments show that the proposed method is efficient and can generalize well to novel poses.
  - [GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering, ICCV2021(oral)](https://arxiv.org/abs/2010.04595) | [***``[code]``***](https://github.com/alextrevithick/GRF)
    > We present a simple yet powerful neural network that implicitly represents and renders 3D objects and scenes only from 2D observations. The network models 3D geometries as a general radiance field, which takes a set of 2D images with camera poses and intrinsics as input, constructs an internal representation for each point of the 3D space, and then renders the corresponding appearance and geometry of that point viewed from an arbitrary position. The key to our approach is to learn local features for each pixel in 2D images and to then project these features to 3D points, thus yielding general and rich point representations. We additionally integrate an attention mechanism to aggregate pixel features from multiple 2D views, such that visual occlusions are implicitly taken into account. Extensive experiments demonstrate that our method can generate high-quality and realistic novel views for novel objects, unseen categories and challenging real-world scenes.
  - [MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo, ICCV2021](https://apchenstu.github.io/mvsnerf/) | [***``[code]``***](https://github.com/apchenstu/mvsnerf)
    > We present MVSNeRF, a novel neural rendering approach that can efficiently reconstruct neural radiance fields for view synthesis. Unlike prior works on neural radiance fields that consider per-scene optimization on densely captured images, we propose a generic deep neural network that can reconstruct radiance fields from only three nearby input views via fast network inference. Our approach leverages plane-swept cost volumes (widely used in multi-view stereo) for geometry-aware scene reasoning, and combines this with physically based volume rendering for neural radiance field reconstruction. We train our network on real objects in the DTU dataset, and test it on three different datasets to evaluate its effectiveness and generalizability. Our approach can generalize across scenes (even indoor scenes, completely different from our training scenes of objects) and generate realistic view synthesis results using only three input images, significantly outperforming concurrent works on generalizable radiance field reconstruction. Moreover, if dense images are captured, our estimated radiance field representation can be easily fine-tuned; this leads to fast per-scene reconstruction with higher rendering quality and substantially less optimization time than NeRF.
  - [Towards Continuous Depth MPI with NeRF for Novel View Synthesis, ICCV2021](https://arxiv.org/abs/2103.14910) | [***``[code]``***](https://github.com/vincentfung13/MINE)
    > In this paper, we propose MINE to perform novel view synthesis and depth estimation via dense 3D reconstruction from a single image. Our approach is a continuous depth generalization of the Multiplane Images (MPI) by introducing the NEural radiance fields (NeRF). Given a single image as input, MINE predicts a 4-channel image (RGB and volume density) at arbitrary depth values to jointly reconstruct the camera frustum and fill in occluded contents. The reconstructed and inpainted frustum can then be easily rendered into novel RGB or depth views using differentiable rendering. Extensive experiments on RealEstate10K, KITTI and Flowers Light Fields show that our MINE outperforms state-of-the-art by a large margin in novel view synthesis. We also achieve competitive results in depth estimation on iBims-1 and NYU-v2 without annotated depth supervision. Our source code is available at this https URL
  - [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction, ICCV2021(oral)](https://arxiv.org/abs/2104.10078) | [***``[code]``***](https://github.com/autonomousvision/unisurf)
    > Neural implicit 3D representations have emerged as a powerful paradigm for reconstructing surfaces from multi-view images and synthesizing novel views. Unfortunately, existing methods such as DVR or IDR require accurate per-pixel object masks as supervision. At the same time, neural radiance fields have revolutionized novel view synthesis. However, NeRF's estimated volume density does not admit accurate surface reconstruction. Our key insight is that implicit surface models and radiance fields can be formulated in a unified way, enabling both surface and volume rendering using the same model. This unified perspective enables novel, more efficient sampling procedures and the ability to reconstruct accurate surfaces without input masks. We compare our method on the DTU, BlendedMVS, and a synthetic indoor dataset. Our experiments demonstrate that we outperform NeRF in terms of reconstruction quality while performing on par with IDR without requiring masks.
  - [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction, NeurIPS2021](https://arxiv.org/abs/2106.10689) | [***``[code]``***](https://github.com/Totoro97/NeuS)
    > We present a novel neural surface reconstruction method, called NeuS, for reconstructing objects and scenes with high fidelity from 2D image inputs. Existing neural surface reconstruction approaches, such as DVR and IDR, require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures. Meanwhile, recent neural methods for novel view synthesis, such as NeRF and its variants, use volume rendering to produce a neural scene representation with robustness of optimization, even for highly complex objects. However, extracting high-quality surfaces from this learned implicit representation is difficult because there are not sufficient surface constraints in the representation. In NeuS, we propose to represent a surface as the zero-level set of a signed distance function (SDF) and develop a new volume rendering method to train a neural SDF representation. We observe that the conventional volume rendering method causes inherent geometric errors (i.e. bias) for surface reconstruction, and therefore propose a new formulation that is free of bias in the first order of approximation, thus leading to more accurate surface reconstruction even without the mask supervision. Experiments on the DTU dataset and the BlendedMVS dataset show that NeuS outperforms the state-of-the-arts in high-quality surface reconstruction, especially for objects and scenes with complex structures and self-occlusion.
  - [Volume Rendering of Neural Implicit Surfaces, NeurIPS2021](https://arxiv.org/abs/2106.12052) | [code]
    > Neural volume rendering became increasingly popular recently due to its success in synthesizing novel views of a scene from a sparse set of input images. So far, the geometry learned by neural volume rendering techniques was modeled using a generic density function. Furthermore, the geometry itself was extracted using an arbitrary level set of the density function leading to a noisy, often low fidelity reconstruction. The goal of this paper is to improve geometry representation and reconstruction in neural volume rendering. We achieve that by modeling the volume density as a function of the geometry. This is in contrast to previous work modeling the geometry as a function of the volume density. In more detail, we define the volume density function as Laplace's cumulative distribution function (CDF) applied to a signed distance function (SDF) representation. This simple density representation has three benefits: (i) it provides a useful inductive bias to the geometry learned in the neural volume rendering process; (ii) it facilitates a bound on the opacity approximation error, leading to an accurate sampling of the viewing ray. Accurate sampling is important to provide a precise coupling of geometry and radiance; and (iii) it allows efficient unsupervised disentanglement of shape and appearance in volume rendering. Applying this new density representation to challenging scene multiview datasets produced high quality geometry reconstructions, outperforming relevant baselines. Furthermore, switching shape and appearance between scenes is possible due to the disentanglement of the two.