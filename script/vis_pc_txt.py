import open3d as o3d
import numpy as np
import argparse

def read_off_file(off_file_path):
    """
    Read OFF file and return mesh with normals and color
    
    Args:
        off_file_path: Path to the OFF file
    
    Returns:
        mesh: Open3D mesh object with vertex normals and green color
    """
    mesh = o3d.io.read_triangle_mesh(off_file_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0, 1, 0])
    return mesh

def read_txt_file(txt_file_path):
    """
    Read TXT point cloud file and return colored pointcloud
    
    Args:
        txt_file_path: Path to the TXT file (points in CSV format)
    
    Returns:
        pcd: Open3D point cloud object with red color
    """
    points      = np.loadtxt(txt_file_path, delimiter=',')
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])
    return pcd

if __name__ == "__main__":
    OFF_FILE_PATH = '/workspace/data/ModelNet40/airplane/train/airplane_0001.off'
    TXT_FILE_PATH = '/workspace/data/ModelNet40_txt/airplane/train/airplane_0001.txt'
    parser = argparse.ArgumentParser(description='Visualize ModelNet40 mesh (green) and converted pointcloud (red)')

    parser.add_argument('--off', '-o', type=str, 
                       default=OFF_FILE_PATH,
                       help='Path to the OFF mesh file for visualization (default: airplane sample)')
    parser.add_argument('--txt', '-x', type=str,
                       default=TXT_FILE_PATH,
                       help='Path to the TXT point cloud file for visualization (default: airplane sample)')
    parser.add_argument('--num-points', '-n', type=int, default=4096,
                       help='Number of points to sample per point cloud (default: 4096)')
    args = parser.parse_args()
    

    mesh    = read_off_file(args.off)
    pcd     = read_txt_file(args.txt)
    o3d.visualization.draw_geometries([pcd])