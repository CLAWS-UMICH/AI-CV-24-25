import trimesh

def ply_to_glb(ply_file, glb_file):
    # Load the PLY file
    scene = trimesh.load(ply_file, force='scene')
    
    # Check if the scene contains geometry
    if scene.is_empty:
        scene = trimesh.load(ply_file.replace("meshed-poisson", "fused"), force='scene')
        print(f"Converting fused to glb")
    else:
        print(f"Converting meshed-poisson to glb")
    # Export the scene as GLB
    scene.export(glb_file, file_type='glb')
