"""Plotly views shared by the workbench and self-contained HTML exports."""
import html
import numpy as np
import plotly.graph_objects as go
from .mesh import TetMesh


def mesh_trace(points,faces,*,name,color='#38bdf8',opacity=1.,visible=True):
    return go.Mesh3d(x=points[:,0],y=points[:,1],z=points[:,2],
                     i=faces[:,0],j=faces[:,1],k=faces[:,2],name=name,color=color,
                     opacity=opacity,visible=visible,showlegend=True)


def layout(fig):
    fig.update_layout(scene=dict(aspectmode='data',xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)'),
                      height=650,margin=dict(l=0,r=0,b=30,t=30),uirevision='s3-view',
                      legend=dict(orientation='h'))
    return fig


def path_trace(curves,kind,name,visible=True):
    points=[]
    for curve in curves:
        if curve.kind==kind:points.extend(curve.points.tolist()+[[None,None,None]])
    xyz=np.asarray(points,dtype=object).reshape(-1,3)
    return go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',name=name,
                        line=dict(color='#f59e0b' if kind=='contour' else '#ec4899',width=4),
                        visible=visible,connectgaps=False,hoverinfo='name')


def toolpath_figure(result,layers,paths,*,index=None,cumulative=False,show_surface=True,show_normals=False):
    mesh=TetMesh(result['points'],result['cells'])
    fig=go.Figure(mesh_trace(mesh.points,mesh.faces[mesh.boundary],name='Part',opacity=.10,color='#94a3b8'))
    selected=list(range(len(layers))) if index is None else (list(range(index+1)) if cumulative else [index])
    curves=[curve for i in selected for curve in paths[i]]
    for kind in ('contour','stress'):fig.add_trace(path_trace(curves,kind,f'{kind.title()} strokes'))
    if show_surface:
        displayed=selected if index is None else [index]
        pts=[];faces=[];offset=0
        for i in displayed:
            pts.append(layers[i].points);faces.append(layers[i].faces+offset);offset+=len(layers[i].points)
        if pts:fig.add_trace(mesh_trace(np.vstack(pts),np.vstack(faces),name='Curved layers',opacity=.25))
    if show_normals and curves:
        pts=np.vstack([p.points for p in curves]);ns=np.vstack([p.normals for p in curves])
        stride=max(1,int(np.ceil(len(pts)/150)))
        pts=pts[::stride];ns=ns[::stride]
        fig.add_trace(go.Cone(x=pts[:,0],y=pts[:,1],z=pts[:,2],u=ns[:,0],v=ns[:,1],w=ns[:,2],
                             sizemode='absolute',sizeref=max(np.ptp(mesh.points,axis=0))*.025,
                             anchor='tail',showscale=False,name='Surface normals',colorscale='Blues'))
    return layout(fig)


def deformation_figure(result,amount=1.):
    mesh=TetMesh(result['points'],result['cells']);faces=mesh.faces[mesh.boundary]
    points=(1-amount)*result['points']+amount*result['deformed']
    fig=go.Figure(mesh_trace(result['points'],faces,name='Original reference',opacity=.12,color='#94a3b8'))
    fig.add_trace(mesh_trace(points,faces,name='Deformation',opacity=.9))
    return layout(fig)


def write_html(path,result,layers,paths,report):
    deformation=deformation_figure(result)
    steps=[]
    for amount in np.linspace(0,1,11):
        points=(1-amount)*result['points']+amount*result['deformed']
        steps.append(dict(method='restyle',args=[dict(x=[points[:,0].tolist()],y=[points[:,1].tolist()],z=[points[:,2].tolist()]),[1]],
                          label=f'{amount:.0%}'))
    deformation.update_layout(sliders=[dict(steps=steps,active=10,currentvalue=dict(prefix='Deformation: '))])
    mesh=TetMesh(result['points'],result['cells'])
    fig=go.Figure(mesh_trace(mesh.points,mesh.faces[mesh.boundary],name='Part',opacity=.10,color='#94a3b8'))
    for i,(layer,curves) in enumerate(zip(layers,paths)):
        fig.add_trace(mesh_trace(layer.points,layer.faces,name=f'Layer {i+1}',opacity=.2,visible=i==len(layers)-1))
        for kind in ('contour','stress'):fig.add_trace(path_trace(curves,kind,f'{i+1} · {kind}'))
    steps=[]
    for i in range(len(layers)):
        visible=[True]
        for j in range(len(layers)):visible.extend([j==i,j<=i,j<=i])
        steps.append(dict(method='update',args=[dict(visible=visible)],label=str(i+1)))
    layout(fig)
    fig.update_layout(showlegend=False,sliders=[dict(steps=steps,active=len(layers)-1,currentvalue=dict(prefix='Layers through: '))])
    content='<!doctype html><html><head><meta charset="utf-8"><title>S³ deformation and toolpaths</title></head>'
    content+='<body style="font-family:system-ui;max-width:1400px;margin:24px auto;padding:0 20px">'
    content+='<h1>S³ deformation and toolpaths</h1><p>Drag to rotate; scroll to zoom. Geometry and path coordinates are in millimeters.</p>'
    build=report.get('build_plate')
    if not build or not build.get('fixed'):
        content+='<p><strong>Unconstrained build base:</strong> this run may tilt the part away from its intended starting surface. It is not a validated build sequence.</p>'
    elif build.get('floating_minimum_vertices') or build.get('below_plate_vertices'):
        content+='<p><strong>Invalid build start:</strong> the field contains off-base starting regions or material below the plate.</p>'
    else:
        content+='<p>Build-plate contact is fixed. No below-plate vertices or off-base scalar minima were detected; overhang and coverage checks remain separate.</p>'
    content+='<h2>Deformation</h2><p>The slider interpolates the original and final geometry for inspection.</p>'
    content+=deformation.to_html(full_html=False,include_plotlyjs=True)
    content+='<h2>Curved toolpaths</h2><p>Orange: boundary contours. Pink: stress-aligned interiors. Separate strokes have no implied travel connection.</p>'
    content+=fig.to_html(full_html=False,include_plotlyjs=False)
    content+=f'<p>{report["total_curves"]:,} strokes · {report["total_waypoints"]:,} waypoints · {report["total_length"]:.2f} mm deposition length.</p>'
    content+=f'<p>Layer mode: {html.escape(report["layers"]["mode"])}. Physical coverage and collision clearance are not certified.</p></body></html>'
    path.write_text(content)
