This guide describes how to process and open ortho maps in Paraview.

#### Why:

Opening an Ortho map in Paraview is useful for multiple reasons:

- Better contextual information than point clouds
- Better trajectory background than satellite imagery
- Possible to do animations
- Possible to blend 3D model with point clouds

###### Requirements:

Paraview

#### How to:

1. Export an `.obj` zip file from ortho map software
2. The zip file contains a folder with:
   1. `.obj` file
   2. multiple `.png` files describing the textures
   3. `.mtl` file describing the relation between the object and textures
3. Use the `obj-mtl-importer.py` script. The script differs in several ways from the one on the internet, since the one there relies on Paraview 5.8 and some functions are depricated. The main (and actually the only difference) is replacing:

```python
from paraview import servermanager
texture = servermanager._getPyProxy(servermanager.CreateProxy('textures', 'ImageTexture'))
texture.FileName = os.path.join(self.baseDir, material['map_Kd'][0])
self.textures[name] = texture
servermanager.Register(texture)
```

with

```python
from paraview.simple import CreateTexture
texture = CreateTexture(os.path.join(self.baseDir, material['map_Kd'][0]))
self.textures[name] = texture
```

4. Call it with `path_to_paraview_python obj-mtl-importer.py path_to_obj_file`. `path_to_paraview_python` is a python executable called `pvpython` that comes with Paraview. On MacOS, it is a file located in `/Applications/ParaView-5.13.3.app/Contents/bin/`.

#### Related links:

- [Creating render view in Python](https://discourse.paraview.org/t/python-documentation-for-createview/3507)
- Paraview python [docs](https://www.paraview.org/paraview-docs/v5.13.3/python/)
- The original obj-mtl-importer.py [script](https://github.com/Kitware/vtk-js/blob/master/Utilities/ParaView/obj-mtl-importer.py)
- Some random links related to `.obj` and `.gltf` file imports:
  - [1](https://discourse.paraview.org/t/obj-mtl-importer-py-errors/5309)
  - [2](https://discourse.paraview.org/t/gltf-textures/7264)
  - [3](https://discourse.paraview.org/t/paraview-5-9-and-gltf2-0/6591)
  - [4](https://discourse.paraview.org/t/gltf-textures/7264)
  - [5](https://discourse.paraview.org/t/5-10-how-to-render-a-gltf-file/9160)
  - [6](https://discourse.paraview.org/t/how-to-load-a-obj-model-with-multiple-textures/9939)
  - [7](https://discourse.paraview.org/t/wavefront-obj-with-materials-mtl/1012)
  - [8](https://discourse.paraview.org/t/show-obj-with-mtl/2932/10)
  -
- [pygltflib](https://pypi.org/project/pygltflib/) allows conversion between glb and gltf
- online gltf [visualization](https://www.gltfeditor.com/)
-
