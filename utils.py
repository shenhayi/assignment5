import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np  

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    # Ensure all tensors are on the correct device
    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(device)
    sample_labels = labels.unsqueeze(0).to(device)
    sample_colors = torch.zeros((1,10000,3), device=device)

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i], device=device)

    sample_colors = sample_colors.repeat(30,1,1)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)

def viz_cls (args, verts, labels, path, device):
    """
    visualize classification result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    point_color = torch.tensor([0.7, 0.0, 0.0]).unsqueeze(0).to(device)  # Single color for all points

    # Create point cloud with single color
    points = verts.unsqueeze(0)
    rgba = torch.ones_like(points).to(device) * point_color
    point_cloud = pytorch3d.structures.Pointclouds(
        points=points,
        features=rgba,
    ).to(device)

    # Setup renderer with lights
    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Generate multiple viewpoints 
    rends = []
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    for theta in azim:
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=theta)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(point_cloud, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rends.append((rend * 255).astype(np.uint8))

    imageio.mimsave(path, rends, fps=15, loop=0)
