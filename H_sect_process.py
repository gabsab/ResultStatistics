import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pandas as pd
import os

class Specimen:
    """
    The specimen contains information for the name of the section, load, LPF, top vertical displacement,
    horizontal displacement at mid-length, end-rotations and the deformed shape at different load conditions

    """
    def __init__(self,
                 name=None,
                 comment=None,
                 geometry=None,
                 test_data=None,
                 numerical_data=None):
        self.name = name
        self.comment = comment
        self.geometry = geometry
        self.test_data = test_data
        self.numerical_data = numerical_data
        self.load = load
        self.lpf = lpf
        self.v_displ = v_displ
        self.h_displ = h_displ
        self.e_rot = e_rot
        self.path_data = path_data
        self.file = file


class Geometry:
    """
    The specimen contains information for the name of the section, load, LPF, top vertical displacement,
    horizontal displacement at mid-length, end-rotations and the deformed shape at different load conditions

    """

    def __init__(self,
                 length=None,
                 cross_section=None,
                 imperfection=None,
                 tolerance_class=None,):
        self.length = length
        self.cross_section = cross_section
        self.tolerance_class = tolerance_class


class Test_data:
    """
    The specimen contains information for the name of the section, load, LPF, top vertical displacement,
    horizontal displacement at mid-length, end-rotations and the deformed shape at different load conditions

    """

    def __init__(self,
                 load=None,
                 lpf=None,
                 deformation=None,
                 v_displ=None,
                 h_displ=None,
                 e_rot=None,
                 path_data=None,
                 imperfections=None,
                 file=None):
        self.load = load
        self.lpf = lpf
        self.deformation = deformation
        self.v_displ = v_displ
        self.h_displ = h_displ
        self.e_rot = e_rot
        self.path_data = path_data
        self.imperfections = imperfections
        self.file = file



File_name = 'C:\\Users\\gabsab\\Documents\\Personal\\Test_data\\Test_db.xlsx'
Sheet_name = 'Nominal'    # sheet name 'All'
Sheet_name = 'Average'
col_data_nominal = pd.read_excel(File_name, Sheet_name, header=0)
col_data_nominal = col_data_nominal.drop(index=0)
A_all = []
Iy_all = []
Iz_all = []
for h, bf, tf, tw, fy in zip(col_data_nominal.H, col_data_nominal.B, col_data_nominal.tf, col_data_nominal.tw, col_data_nominal.ReH):
    [A, Aeff, Iy, Iz, Iyeff, Izeff, iy, iz, f_ratio, class_f, lambda_f, ro_f, w_ratio, class_w, lambda_w, ro_w] = \
        calc_prop_I(h, bf, tf, tw, aw=3, fy=355, YModulus=2.1e5, nu=0.3)
    A_all.append(A)
    Iy_all.append(Iy)
    Iz_all.append(Iz)
for A, Iy, Iz, Lcr, fy in zip(A_all, Iy_all, Iz_all, col_data_nominal.Lcr,col_data_nominal.ReH):
    [lambda_y1, lambda_y, lambda_z1, lambda_z, Npl, Neff, Nby, Nbz, csi_y, csi_z, ro, csi_ty, csi_tz] = \
        flexural_buckling(A=A, Iy=Iy, Iz=Iz, Lcr=Lcr, fy=fy, alpha_y=0.34, alpha_z=0.49, )
    print(lambda_z)