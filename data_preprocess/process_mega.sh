cd data/mega/
tar xopf ArtsQuad_dataset.tar
mkdir ArtsQuad
mv ArtsQuad_* ArtsQuad
tar -zxvf building-pixsfm.tgz
tar -zxvf building-pixsfm-grid-8.tgz
tar -zxvf rubble-pixsfm.tgz
tar -zxvf rubble-pixsfm-grid-8.tgz
tar -zxvf quad-pixsfm.tgz
tar -zxvf quad-pixsfm-grid-8.tgz
tar -zxvf residence-pixsfm.tgz
tar -zxvf residence-pixsfm-grid-8.tgz
tar -zxvf sci-art-pixsfm.tgz
tar -zxvf sci-art-pixsfm-grid-25.tgz
tar -zxvf campus-pixsfm.tgz
tar -zxvf campus-pixsfm-grid-8.tgz
mkdir building
mv building-* building
mkdir campus
mv campus-* campus
mkdir rubble
mv rubble-* rubble
mkdir residence
mv residence-* residence
mkdir campus
mv campus-* campus
mkdir sci-art
mv sci-art-* sci-art
mkdir quad
mv quad-* quad
cd ../../