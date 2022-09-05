# Download and process Mega datasets.

You can download the mega dataset via the "data_preprocess/download\_mega.sh" or from [our Google drive](https://drive.google.com/drive/folders/1zzvGWhrbx2_XuK_6mBYpkGngHoL9QGMR?usp=sharing). You can download selected data from this table:

| Dataset name | Images & poses | Masks | Pretrained models |
|---|---|---|---|
| Waymo | [waymo\_image\_poses](https://drive.google.com/file/d/1U7wcE5r-kWtUBscljjTn6q18E8E8kJTd/view?usp=sharing) | - | - |
| Building | [building-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz) | [building-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-grid-8.tgz) | [building-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-8.pt) |
| Rubble | [rubble-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz) | [rubble-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-grid-8.tgz) | [rubble-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-8.pt) |
| Quad | [ArtsQuad_dataset](http://vision.soic.indiana.edu/disco_files/ArtsQuad_dataset.tar) - [quad-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm.tgz) | [quad-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-grid-8.tgz) | [quad-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-8.pt) |
| Residence | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [residence-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm.tgz) | [residence-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-grid-8.tgz) | [residence-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-8.pt) |
| Sci-Art | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [sci-art-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz) | [sci-art-pixsfm-grid-25](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-grid-25.tgz) | [sci-art-pixsfm-25-w-512.pt](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-25-w-512.pt) |
| Campus | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [campus](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm.tgz) | [campus-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-grid-8.tgz) | [campus-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-8.pt) |

We provide detailed explanations with examples for the Mega-NeRF data structure in [this doc](docs/mega_format_explained.md). After downloading the data, unzip the files and make folders via the following commands:

```bash
bash data_preprocess/process_mega.sh
```
