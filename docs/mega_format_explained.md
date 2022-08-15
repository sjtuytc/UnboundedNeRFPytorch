# The Mega standard format (MSF) explained.

> The MSF is a data format for large-scale NeRFs, which is proposed by Mega-NeRF.

The format of the final data with trained weights should be like:
```
data
  |——————building
  |         |——————building-pixsfm // the source data folder
  |         |        └——————train
  |         |        |        └——————metadata
  |         |        |        |         └——————000001.pt   
  |         |        |        └——————rgbs
  |         |        |        |         └——————000001.jpg   
  |         |        └——————val
  |         |        |        └——————metadata
  |         |        |        |         └——————000001.pt   
  |         |        |        └——————rgbs
  |         |        |                  └——————000001.jpg    
  |         |        └——————coordinates.pt
  |         |        └——————mappings.txt
  |         └——————building-pixsfm-grid-8 // masks
  |         |        |        └——————0
  |         |        |        |         └——————000001.pt  
  |         |        |        └——————params.pt
  |         └——————building-pixsfm-grid-8.pt  //final model
```