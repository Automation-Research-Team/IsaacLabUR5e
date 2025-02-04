import os
from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np

LOCAL_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
USD_DIR = os.path.join(LOCAL_ASSETS_DIR, "usd")
TEXTURES_DIR = os.path.join(LOCAL_ASSETS_DIR, "textures")

try:
    from torchvision.io import read_image
except ModuleNotFoundError as e:
    print(f"{e} - Running the mesh generator requires torchvision")
import shutil

try:
    import kaolin as kal
    from kaolin.rep import SurfaceMesh
    from kaolin.render.materials import PBRMaterial
except ModuleNotFoundError as e:
    print(f"{e} - Running the mesh generator requires kaolin")
try:
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrnp
except ModuleNotFoundError as e:
    print(f"{e} - Running the mesh generator requires meshlib")
try:
    from pxr import Usd, Sdf, UsdShade
except ModuleNotFoundError as e:
    print(f"{e} - Running the mesh generator requires usd-core")

def imshow(img_path: str | os.PathLike, is_rgb: bool=True) -> None:
    img = read_image(str(img_path))
    if is_rgb:
        img = torch.einsum("cwh->whc", img)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    
def print_tensor(t, name='', **kwargs):
    print(kal.utils.testing.tensor_info(t, name=name, **kwargs))

def predict_image_depth(img: torch.Tensor)-> torch.Tensor:

    img_device = img.device

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas: nn.Module = torch.hub.load("intel-isl/MiDaS", "DPT_Large") # type: ignore
    midas.to(model_device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform # type: ignore

    untransformed_input = torch.einsum("cwh->whc", img).cpu().numpy()
    input = transform(untransformed_input).to(model_device)

    with torch.no_grad():
        pred = midas(input)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
    pred = pred.to(img_device)
    return pred

def sample_mesh_path(rel_path: str | os.PathLike) -> str | os.PathLike:
    return os.path.join(USD_DIR, rel_path)

def import_mesh(path: str | os.PathLike, cuda:bool = True) -> SurfaceMesh:
    mesh = kal.io.import_mesh(path)
    # Fix material import (material property convention mismatch between omniverse usd assets and kaolin expected format)
    if mesh.materials is not None and len(mesh.materials) == 1 and type(mesh.materials[0]) == dict:
        mesh.materials = [PBRMaterial(
            material_name=mesh.materials[0].get("material_name"),
            diffuse_color=None if "diffuse_color_constant" not in mesh.materials[0] else mesh.materials[0].get("diffuse_color_constant").get("value"),
            diffuse_texture=None if "diffuse_texture" not in mesh.materials[0] else (read_image(mesh.materials[0].get("diffuse_texture").get("value").resolvedPath)/255.0)
        )]
    if cuda:
        mesh = mesh.cuda()
    return mesh

def export_mesh(path: str | os.PathLike, mesh: SurfaceMesh, omni_template_path: str | os.PathLike | None = None, texture_path: str | os.PathLike | None = None) -> None:
    dir = os.path.split(path)[0]
    if not os.path.isdir(dir):
        os.mkdir(dir)
    fname = str(path).split(".")[-2]
    if omni_template_path:
        omni_template_dir, omni_template_fname = os.path.split(omni_template_path)
        omni_template_fname = omni_template_fname.split(".")[0]
        omni_template_root = "_".join(omni_template_fname.split("_")[:2])

        omni_instanceables_fname = f"{omni_template_fname}.usd"
        omni_instanceables_path = os.path.join(omni_template_dir, omni_instanceables_fname)
        instanceables_path = os.path.join(dir, omni_instanceables_fname)
        instanceables_scene_path = f"/{omni_template_fname}_loose"
        instanceables_scene_mesh_path = f"{instanceables_scene_path}/{omni_template_fname[:-4]}_loose"
        shutil.copyfile(omni_instanceables_path, instanceables_path)

        kal.io.usd.export_mesh(file_path=instanceables_path,
            scene_path=os.path.join(instanceables_scene_mesh_path, "collisions"),
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            uvs=mesh.uvs,
            face_uvs_idx=mesh.face_uvs_idx,
            materials=None,
            material_assignments=mesh.material_assignments)

        if texture_path and mesh.materials is not None:
            stage_ref = Usd.Stage.Open(instanceables_path) # type: ignore
            mat_prim = stage_ref.GetPrimAtPath(os.path.join(instanceables_scene_path, "Looks", f"{os.path.split(mesh.materials[0].material_name)[-1]}", "Shader"))
            mat_prim = UsdShade.Shader(mat_prim) # type: ignore
            mat_prim.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(os.path.relpath(texture_path, start=os.path.dirname(instanceables_path))) # type: ignore

        kal.io.usd.export_mesh(file_path=instanceables_path,
            scene_path=os.path.join(instanceables_scene_mesh_path, "visuals"),
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            uvs=mesh.uvs,
            face_uvs_idx=mesh.face_uvs_idx,
            materials=None,
            material_assignments=mesh.material_assignments)

    else:
        scene_path = f"/{fname}"
        materials=mesh.materials
        kal.io.usd.export_mesh(file_path=path,
                            scene_path=scene_path,
                            vertices=mesh.vertices,
                            faces=mesh.faces,
                            face_normals=mesh.face_normals,
                            materials=materials,
                            material_assignments=mesh.material_assignments)
        
def get_applied_texture_deformations(mesh: SurfaceMesh, deformation_texture: torch.Tensor, conversion: str = "mean") -> torch.Tensor:
    # Set uvs
    # Random uniform uvs
    mesh.uvs = torch.zeros((len(mesh.vertices), 2)).to(mesh.vertices.device) # type: ignore
    unq_vertices, unq_vertex_indices = torch.unique(mesh.vertices, return_inverse=True, dim=0)
    rand_uvs = torch.rand(size=(len(unq_vertices), 2)).to(mesh.vertices.device) # type: ignore
    for v, v_idx in zip(unq_vertices, unq_vertex_indices):
        x,y,z = v
        v_mask = torch.where(((mesh.vertices[:, 0] == x) & (mesh.vertices[:, 1] == y) & (mesh.vertices[:, 2] == z)), True, False) # type: ignore
        mesh.uvs[v_mask, :] = rand_uvs[v_idx, :]
    mesh.face_uvs_idx = mesh.faces

    if conversion == "depth":
        deformation_texture = predict_image_depth(deformation_texture)
        deformation_texture = torch.tile(deformation_texture, (3, 1, 1))
    vertex_texture_vals = (kal.render.mesh.texture_mapping(mesh.uvs[unq_vertex_indices].unsqueeze(0), deformation_texture.unsqueeze(0))).squeeze(0)
    if conversion == "mean":
        # Heuristic for corrosion textures:
        # corrosion pixels (which we want to use to deform the corresponding vertices more)
        # usually have high r but low g and b so the mean across these is low, whereas grey steel typically has closer rgb vals so a higher mean
        # Could be improved, e.g. might not handle 'blue-ish' steel well
        scales = 1-torch.mean(vertex_texture_vals, dim=1)
    elif conversion == "depth":
        scales = vertex_texture_vals[:, 0]
    else:
        scales = vertex_texture_vals
    return scales

def deform_mesh(mesh: SurfaceMesh, scales: torch.Tensor | None = None, limits: torch.Tensor | None = None) -> SurfaceMesh:

    unq_vertices, unq_vertex_indices = torch.unique(mesh.vertices, return_inverse=True, dim=0)

    if limits is not None:
        limits = limits.to(mesh.vertices.device) # type: ignore
    else:
        limits = torch.ones((len(unq_vertex_indices),)).to(mesh.vertices.device) # type: ignore
    if scales is not None:
        scales = scales.to(mesh.vertices.device) # type: ignore
    else:
        scales = torch.rand((len(unq_vertex_indices),)).to(mesh.vertices.device) # type: ignore
    scales *= limits

    for v, v_idx in zip(unq_vertices, unq_vertex_indices):
        x,y,z = v
        v_mask = torch.where(((mesh.vertices[:, 0] == x) & (mesh.vertices[:, 1] == y) & (mesh.vertices[:, 2] == z)), True, False) # type: ignore
        v_normal = mesh.vertex_normals[v_idx] # type: ignore
        mesh.vertices[v_mask, :] += scales[v_idx] * v_normal # type: ignore
    return mesh

# Create a new usd with a deformed mesh
def create_deformed_mesh(src_mesh_path: str | os.PathLike, new_mesh_path: str | os.PathLike, deformation_texture_path: str | os.PathLike | None = None, limits: torch.Tensor | None = None, texture_conversion: str="mean") -> SurfaceMesh:
    mesh = import_mesh(src_mesh_path)
    scales = None
    if deformation_texture_path:
        deformation_texture = (read_image(str(deformation_texture_path))/255.0).to(mesh.vertices.device) # type: ignore
        scales = get_applied_texture_deformations(mesh, deformation_texture, conversion=texture_conversion)
    deform_mesh(mesh, scales=scales, limits=limits)
    export_mesh(new_mesh_path, mesh, omni_template_path=src_mesh_path, texture_path=deformation_texture_path)
    return mesh

def normalize_mesh(mesh: SurfaceMesh) -> SurfaceMesh:
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0) # type: ignore
    return mesh

def get_mesh_centre(mesh: SurfaceMesh) -> torch.Tensor:
    return (torch.min(mesh.vertices, dim=0).values + torch.max(mesh.vertices, dim=0).values) / 2 # type: ignore

def centre_mesh_at_point(mesh: SurfaceMesh, point: torch.Tensor) -> SurfaceMesh:
    mesh.vertices -= point # type: ignore
    return mesh

def get_paired_mesh_proximities(first_mesh_path: str | os.PathLike, second_mesh_path: str | os.PathLike) -> tuple[torch.Tensor, torch.Tensor]:

    first_mesh = import_mesh(first_mesh_path, cuda=False)
    second_mesh = import_mesh(second_mesh_path, cuda=False)

    mr_first_mesh = mrnp.meshFromFacesVerts(first_mesh.faces.numpy(), first_mesh.vertices.numpy()) # type: ignore
    mr_second_mesh = mrnp.meshFromFacesVerts(second_mesh.faces.numpy(), second_mesh.vertices.numpy()) # type: ignore

    first_mesh_vertex_distances = []
    second_mesh_vertex_distances = []

    for v in first_mesh.vertices.numpy(): # type: ignore
        v = mr.Vector3f(v[0], v[1], v[2])
        dist = mr.findSignedDistance(v, mr_second_mesh).dist # type: ignore
        first_mesh_vertex_distances.append(dist)
    first_mesh_vertex_distances = torch.Tensor(first_mesh_vertex_distances)

    for v in second_mesh.vertices:
        v = mr.Vector3f(v[0], v[1], v[2])
        dist = mr.findSignedDistance(v, mr_first_mesh).dist # type: ignore
        second_mesh_vertex_distances.append(dist)
    second_mesh_vertex_distances = torch.Tensor(second_mesh_vertex_distances)

    return first_mesh_vertex_distances, second_mesh_vertex_distances


def main():

    # Calculate nut and bolt deformation limits based on their proximities in an assembly
    nut_limit_eps = -0.0025
    bolt_limit_eps = 0.0025
    bolt_mesh_path = sample_mesh_path(os.path.join("bolt", "default", "factory_bolt_m16.usd"))
    nut_mesh_path = sample_mesh_path(os.path.join("nut", "default", "factory_nut_m16.usd"))
    nut_vertex_scale_limits, bolt_vertex_scale_limits = get_paired_mesh_proximities(first_mesh_path=nut_mesh_path, second_mesh_path=bolt_mesh_path)
    nut_vertex_scale_limits *= nut_limit_eps
    bolt_vertex_scale_limits *= bolt_limit_eps

    corrosion_texture_path = os.path.join(TEXTURES_DIR, "steel_corrosion.jpg")

    damaged_nut_mesh_path = sample_mesh_path(os.path.join("nut", "damaged", "factory_nut_m16_damaged.usd"))
    damaged_nut_mesh = create_deformed_mesh(nut_mesh_path, damaged_nut_mesh_path, deformation_texture_path=corrosion_texture_path, limits=nut_vertex_scale_limits, texture_conversion="mean")

    bolt_mesh_path = sample_mesh_path(os.path.join("bolt", "default", "factory_bolt_m16.usd"))
    damaged_bolt_mesh_path = sample_mesh_path(os.path.join("bolt", "damaged", "factory_bolt_m16_damaged.usd"))
    damaged_bolt_mesh = create_deformed_mesh(bolt_mesh_path, damaged_bolt_mesh_path, deformation_texture_path=corrosion_texture_path, limits=bolt_vertex_scale_limits, texture_conversion="mean")

if __name__ == "__main__":
    main()