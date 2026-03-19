import trimesh
import numpy as np
import os
import argparse
from tqdm import tqdm

def read_off_file(off_file_path, num_points):
    """
    Read OFF file and sample point cloud from mesh surface
    
    Args:
        off_file_path:  Path to the OFF file
        num_points:     Number of points to sample
    
    Returns:
        pointcloud: Sampled point cloud array
    """
    mesh            = trimesh.load(off_file_path, process=False)
    face_count      = len(mesh.faces)
    dynamic_points  = min(num_points, max(256, face_count // 10))
    pointcloud, _   = trimesh.sample.sample_surface(mesh, dynamic_points)
    # print(f"Mesh face count: {face_count}, Dynamic sampling points: {dynamic_points}")
    return pointcloud

def make_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_name(file_path):
    """Extract filename without extension"""
    filename    = os.path.basename(file_path)
    name, _     = os.path.splitext(filename)
    return name

def traversal_dir(path):
    """
    Traverse directory and return subdirectories and files
    
    Args:
        path: Directory path to traverse
    
    Returns:
        dirs:   List of subdirectory paths
        files:  List of file paths
    """
    dirs    = []
    files   = []
    for entry in os.scandir(path):
        if entry.is_dir():
            dirs.append(entry.path)
        elif entry.is_file():
            files.append(entry.path)
    return dirs, files

def run(source_root, target_root, num_points=4096):
    """
    Batch convert OFF files to dense pointcloud
    
    Args:
        source_root:    ModelNet40 root directory
        target_root:    Output directory for pointcloud
        num_points:     Number of points to sample per mesh
    """

    print(f"INFO: source root: {source_root}")
    print(f"INFO: target root: {target_root}")

    class_dirs, _ = traversal_dir(source_root)
    
    for class_dir in tqdm(class_dirs, desc="Processing Classes"):
        class_name = os.path.basename(class_dir)
        
        src_train_dir   = os.path.join(class_dir, 'train')
        src_test_dir    = os.path.join(class_dir, 'test')

        trg_train_dir   = os.path.join(target_root, class_name, 'train')
        trg_test_dir    = os.path.join(target_root, class_name, 'test')

        make_dir(trg_train_dir)
        make_dir(trg_test_dir)

        _, train_off_files  = traversal_dir(src_train_dir)
        _, test_off_files   = traversal_dir(src_test_dir)

        # Process training files
        for off_file in train_off_files:
            if not off_file.endswith('.off'):
                continue
            base_name   = get_file_name(off_file)
            pointcloud  = read_off_file(off_file, num_points)
            save_path   = os.path.join(trg_train_dir, f"{base_name}.txt")
            np.savetxt(save_path, pointcloud, fmt="%.6f", delimiter=',')
        
        # Process test files
        for off_file in test_off_files:
            if not off_file.endswith('.off'):
                continue
            base_name   = get_file_name(off_file)
            pointcloud  = read_off_file(off_file, num_points)
            save_path   = os.path.join(trg_test_dir, f"{base_name}.txt")
            np.savetxt(save_path, pointcloud, fmt="%.6f", delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ModelNet40 OFF files to dense pointcloud')

    parser.add_argument('--source', '-s', type=str, 
                       default='/workspace/data/ModelNet40',
                       help='ModelNet40 source directory path')
    parser.add_argument('--target', '-t', type=str,
                       default='/workspace/data/ModelNet40_txt',
                       help='Pointcloud output directory path')
    parser.add_argument('--num-points', '-n', type=int, default=4096,
                       help='Number of points to sample per point cloud (default: 4096)')
    args = parser.parse_args()

    print('=' * 50)
    print('ModelNet40 OFF to Point Cloud Converter')
    print('=' * 50)

    print('Processing .off file convert to .txt pointcloud file')
    if not os.path.exists(args.source):
        print(f"Error: {args.source} does not exist")
    else:
        run(args.source, args.target, args.num_points)
