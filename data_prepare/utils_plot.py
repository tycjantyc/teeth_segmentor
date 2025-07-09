from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

def plot_3d_fast(image, threshold=1):
    # Position the scan upright
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]
    
    # Extract mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(p, threshold)
    
    # Convert to plotly format
    x, y, z = verts.T
    i, j, k = faces.T
    
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightblue',
        opacity=0.5
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
    )
    
    fig = go.Figure(data=[mesh], layout=layout)
    fig.show()