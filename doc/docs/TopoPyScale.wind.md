<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/wind.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.wind`
Wind downscaling options: blending with the ERA5 10 m wind, and the terrain modification of MicroMet. 

TopoSCALE takes the wind of the pressure levels at the elevation of the point: free-atmosphere wind, without the drag of the surface. Two options change it, alone or together (config toposcale.wind_downscaling): 


- surface_blend: u = u_pl + w (u10 - u_pl), and the same for v, with u10, v10 the ERA5 10 m wind interpolated to  the point. w is a constant in [0, 1], or 'surface_effect': the terrain weight of the surface-effect correction of  temperature (toposcale.surface_effect, REDCAPP or DReaMIT), clipped to [0, 1]: points coupled to the ERA5 surface  (valleys) get its wind, exposed points (ridges) keep the free-atmosphere wind. This reuse of the temperature  weight is a TopoPyScale assumption, not a published method. 
- terrain: liston_elder, the wind weighting and diverting factors of MicroMet (Liston and Elder, 2006, J.  Hydrometeorol. 7, 217-234), applied after the blending: 

 Ws = W (1 + gs Os + gc Oc),       direction + Od,     Od = -0.5 Os sin(2 (xi - theta)) 

 with theta the direction the wind blows from, xi the azimuth the slope faces (both clockwise from north),  Os = beta cos(theta - xi) the slope in the direction of the wind (beta: slope), and Oc the curvature of the terrain  over the length scale eta, both scaled to [-0.5, 0.5] over the DEM: Os by twice the steepest slope of the DEM, Oc  by twice its largest absolute curvature. gs = 0.58, gc = 0.42 and eta = 500 m are the values of MicroMet and  SnowModel. 

Config:  toposcale:  wind_downscaling:  surface_blend: 0            # weight of the ERA5 10 m wind: 0 (off), a value in [0, 1], or surface_effect  terrain: none               # none or liston_elder  slope_weight: 0.58          # gs  curvature_weight: 0.42      # gc  curvature_length: 500       # eta [m] 

**Global Variables**
---------------
- **TERRAIN_METHODS**
- **DEFAULTS**
- **U10**
- **V10**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/wind.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `config`

```python
config(cfg)
```

Options of toposcale.wind_downscaling with their defaults; checks the values. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/wind.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `needs_surface_wind`

```python
needs_surface_wind(cfg)
```






---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/wind.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `curvature`

```python
curvature(z, dx, eta)
```

Terrain curvature of MicroMet on a grid: the mean over the four directions (S-N, W-E, SW-NE, NW-SE) of the elevation of the cell minus the mean of its two neighbours at distance eta, divided by 2 eta (2 sqrt(2) eta on the diagonals). Neighbours beyond the edge take the value of the edge. 



**Args:**
 
 - <b>`z`</b> (array):  elevation [m], (y, x), NaN for no data 
 - <b>`dx`</b> (float):  cell size [m] 
 - <b>`eta`</b> (float):  length scale [m], about half the wavelength of the topographic features 



**Returns:**
 
 - <b>`array`</b>:  curvature [m-1 times m / m], (y, x) 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/wind.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup`

```python
setup(df, cfg, ds_param, per_cluster, surface=None)
```

Terrain parameters of the points for the wind options, and the argument wind of toposcale_core.downscale_block. 



**Args:**
 
 - <b>`df`</b> (dataframe):  df_centroids (point_name, x, y, slope, aspect); the column wind_curvature is added in place  for liston_elder 
 - <b>`cfg`</b> (dict):  config toposcale.wind_downscaling 
 - <b>`ds_param`</b> (dataset):  DEM parameters (elevation, slope on the DEM grid, and point_name for clusters) 
 - <b>`per_cluster`</b> (bool):  curvature of a point = mean over the cells of its cluster (toposub), else the cell of  the point 
 - <b>`surface`</b> (dict):  the surface-effect setup (surface_effect.setup), for surface_blend: surface_effect 



**Returns:**
 dict for downscale_block(wind=...), or None when both options are off 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
