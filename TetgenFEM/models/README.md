# Models

Put all model-related assets in this folder:

- Surface meshes: `*.stl`
- Tetrahedral meshes (direct loading): `*.node` + `*.ele` (+ optional `*.poly`)
- Any auxiliary mesh files (e.g. `*.geo`, `*.msh`, `*.db`, `*.mtl`)

In `TetgenFEM/parameters*.txt` you can switch models by editing only:

```
modelDir=models
stlFile=some_model.stl
nodeFile=some_mesh.node
eleFile=some_mesh.ele
```

If `stlFile/nodeFile/eleFile` already contain a path (e.g. `models/foo.stl`), it will be used as-is.

