openDTAM
========

An open source implementation of DTAM. 

Currently only the dense mapper is implemented. 

## Build Instructions On Ubuntu 14.04 LTS

### Install dependencies

- opencv-2.4.9
- CUDA 3.0+

### Build openDTAM

    cd openDTAM/
    mkdir build
	cd build
    cmake ..
    make

### Run openDTAM
/path/to/openDTAM/build$ optirun -b none ./test-mapping-orb-slam ../data/openDTAM-settings.yaml

### Trouble Shooting
Tested with opencv-2.4.9. Other versions might not work. 
