# Cardiac geometries core

Gmsh files for caridac geometries

Source code : https://github.com/ComputationalPhysiology/cardiac-geometries-core


## Install
You can install the package with pip
```
python3 -m pip install cardiac-geometries-core
```
or you can just run a container without any installation, i.e
```
docker run --rm -w /home/shared -v $PWD:/home/shared -it ghcr.io/computationalphysiology/cardiac-geometries-core:latest
```
For example the following command will create an lv ellipsoid and save it to a file called `lv-mesh.msh`
```
docker run --rm -w /home/shared -v $PWD:/home/shared -it ghcr.io/computationalphysiology/cardiac-geometries-core:latest lv-ellipsoid lv-mesh.msh
```

## Authors
Henrik Finsberg (henriknf@simula.no)

## License
MIT
