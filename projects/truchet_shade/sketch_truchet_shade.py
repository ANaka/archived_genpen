import vsketch
import itertools
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
import vsketch
import shapely.geometry as sg
from shapely.geometry import box, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon, LineString
import shapely.affinity as sa
import shapely.ops as so
import matplotlib.pyplot as plt
import pandas as pd

import vpype_cli
from typing import List, Generic
from genpen import genpen as gp, utils as utils
from scipy import stats as ss
import geopandas
from shapely.errors import TopologicalError
import functools
import vpype
from skimage import io
from pathlib import Path

import bezier
import vpype as vp
from sklearn.preprocessing import minmax_scale
from skimage import feature
from genpen.utils import Paper
from genpen.genpen import *



class TruchetShadeSketch(vsketch.SketchClass):
    # Sketch parameters:
    
    page_size = vsketch.Param("11x14in", choices=list(vp.PAGE_SIZES.keys())+['11x14in', '11x17in'])
    landscape = vsketch.Param(False)
    show_grid = vsketch.Param(False)
    xmargins = vsketch.Param(30., 0, unit="mm")
    ymargins = vsketch.Param(30., 0, unit="mm")
    xgen = vsketch.Param(0.5, min_value=0., max_value=1.)
    p_continue=vsketch.Param(0.85, min_value=0., max_value=1.)
    depth_limit=vsketch.Param(1)
    degrees = vsketch.Param(45.)
    linemerge_tol = vsketch.Param(0., decimals=2)
    delay_hatch =  vsketch.Param(True)
    
    spacing = vsketch.Param(3., min_value=0., decimals=2)
    
    def get_drawbox(self, vsk):
        return box(self.xmargins, self.ymargins, vsk.width-self.xmargins, vsk.height-self.ymargins)
    
    def subdivide(self, poly):
        xgen = make_callable(self.xgen)
        split_func = functools.partial(split_along_longest_side_of_min_rectangle, xgen=xgen)
        splits = recursive_split_frac_buffer(
            poly, 
            split_func=split_func,
            p_continue=self.p_continue, 
            depth=0, 
            depth_limit=self.depth_limit,
            buffer_frac=-0.0
        )

        bps = MultiPolygon([p for p in splits])
        return bps
    
    def fill_polys(self, polys):
        fills = []
        for p in polys:
            xjitter_func = 0
            # yjitter_func = ss.norm(loc=0, scale=np.random.uniform(0.01, 0.3)).rvs
            yjitter_func = 0
            bhf = BezierHatchFill(
                spacing=self.spacing,
                degrees=np.random.choice([self.degrees, self.degrees+90]),
                poly_to_fill=p, 
                xjitter_func=xjitter_func, 
                yjitter_func=yjitter_func,
                fill_inscribe_buffer=1.4,
                n_nodes_per_line=5,
                n_eval_points=20,
            )
            fills.append(bhf.p)

        fills = [f for f in fills if f.length > 0]
        return gp.merge_LineStrings(fills)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size(self.page_size, landscape=self.landscape)
        vsk.penWidth("0.1mm")
        drawbox = self.get_drawbox(vsk)
        # implement your sketch here
        self.bps = self.subdivide(drawbox)
        if self.show_grid:
            vsk.stroke(2)
            vsk.geometry(self.bps)
        if not self.delay_hatch:
            fills = self.fill_polys(self.bps)
            vsk.stroke(1)
            vsk.geometry(fills)
            vsk.vpype(f"splitall linemerge -t {self.linemerge_tol}mm linesort")
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        if self.delay_hatch:
            fills = self.fill_polys(self.bps)
            vsk.stroke(1)
            vsk.geometry(fills)
            vsk.vpype(f"splitall linemerge -t {self.linemerge_tol}mm linesort")


if __name__ == "__main__":
    TruchetShadeSketch.display()
