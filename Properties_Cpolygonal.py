# ********************************************************************************
# Calculate Cross section properties for closed polygonal cross-sections

# Created for the Aeolus4future project
# Author: Gabriel Sabau
# ********************************************************************************
import numpy as np


def calc_prop_c(r, t, xc_c, yc_c, theta, phi):
    # A - Area of the bent arc
    # r - radius of the bent arc
    # t - thickness of the arc
    # theta - angle between 2 facets
    A = theta * r * t
    # ro - outside radius of the bend
    # ri - inside radius of the bend
    ro = r + t / 2
    ri = r - t / 2

    # Ix,Iy,Ixy - second moment of area of the bend relative to center of rotation, principal axis (xc,yc)
    Ix = (ro ** 4 - ri ** 4) / 16 * (2 * theta - np.sin(2 * theta))
    Sx = (ro ** 3 - ri ** 3) * np.sin(theta) / 3
    Iy = Ix
    Sy = (ro ** 3 - ri ** 3) * (1 - np.cos(theta)) / 3
    Ixy = (ro ** 4 - ri ** 4) / 16 * (np.cos(2 * theta) - 1)

    # Ix1,Iy1 - second moment of area of the bend relative to center of rotation, axis paralel to global x,y
    Ix1 = Ix * np.cos(phi) ** 2 + Iy * np.sin(phi) ** 2 - Ixy * np.sin(2 * phi)
    Iy1 = Iy * np.cos(phi) ** 2 + Ix * np.sin(phi) ** 2 + Ixy * np.sin(2 * phi)

    # xcg,ycg - coordinates of the centroid of the bend relative to the center of rotation (xc,yc)
    # theta -
    ycg = Sy / A
    xcg = Sx / A
    ycg_gl = xcg * np.cos(phi) + ycg * np.sin(phi)
    xcg_gl = ycg * np.cos(phi) - xcg * np.sin(phi)
    # Ix0,Iy0 - second moment of area of the bend relative to origin and main axis
    Ix0 = Ix1 + 2 * yc_c * ycg_gl * A + yc_c ** 2 * A
    Iy0 = Iy1 + 2 * xc_c * xcg_gl * A + xc_c ** 2 * A
    I = [A, Ix0, Iy0]
    return I


def calc_prop_p(xstart, ystart, xend, yend, t, cp, phi, ro):
    # A - Area of plate
    # b - width of the plate
    # t - thickness of the plate
    b = np.sqrt((xend - xstart) ** 2 + (yend - ystart) ** 2)
    A = b * t
    Aeff = b * t * ro
    beff = b * ro

    # xcg,ycg - coordinates of the centroid of the plate relative to center of section (xc,yc)
    xcg = cp[0]
    ycg = cp[1]
    # Ix,Iy,Ixy - second moment of area of the plate relative to center of rotation, principal axis (xc,yc)
    Ixeff = (t ** 3) * beff / 12
    Ix = (t ** 3) * b / 12
    Iyeff = t * (beff ** 3) / 12
    Iy = t * (b ** 3) / 12
    Ixy = 0

    # Ix1,Iy1 - second moment of area of the plate relative to center of rotation, axis paralel to global x,y
    Ix1 = Ix * np.cos(phi) ** 2 + Iy * np.sin(phi) ** 2
    Iy1 = Iy * np.cos(phi) ** 2 + Ix * np.sin(phi) ** 2
    Ix1eff = Ixeff * np.cos(phi) ** 2 + Iyeff * np.sin(phi) ** 2
    Iy1eff = Iyeff * np.cos(phi) ** 2 + Ixeff * np.sin(phi) ** 2

    # Ix0,Iy0 - second moment of area of the plate relative to origin and main axis
    Ix0eff = Ix1 + ycg ** 2 * Aeff
    Iy0eff = Iy1 + xcg ** 2 * Aeff
    Ix0 = Ix1 + ycg ** 2 * A
    Iy0 = Iy1 + xcg ** 2 * A
    Ixyp = 0
    I = [A, Aeff, Ix0, Iy0, Ix0eff, Iy0eff]
    return I


def calc_prop_p_EC(xstart, ystart, xend, yend, t, cp, phi, ro):
    # A - Area of plate
    # b - width of the plate
    # t - thickness of the plate
    b = np.sqrt((xend - xstart) ** 2 + (yend - ystart) ** 2)
    A = b * t
    Aeff = b * t * ro
    beff = b * ro

    # xcg,ycg - coordinates of the centroid of the plate relative to center of section (xc,yc)
    xcg = cp[0]
    ycg = cp[1]
    # Ix,Iy,Ixy - second moment of area of the plate relative to center of rotation, principal axis (xc,yc)
    Ixeff = (t ** 3) * beff / 12
    Ix = (t ** 3) * b / 12
    Iyeff = t * (beff ** 3) / 12
    Iy = t * (b ** 3) / 12
    Ixy = 0

    # Ix1,Iy1 - second moment of area of the plate relative to center of rotation, axis paralel to global x,y
    Ix1 = Ix * np.cos(phi) ** 2 + Iy * np.sin(phi) ** 2
    Iy1 = Iy * np.cos(phi) ** 2 + Ix * np.sin(phi) ** 2
    Ix1eff = Ixeff * np.cos(phi) ** 2 + Iyeff * np.sin(phi) ** 2
    Iy1eff = Iyeff * np.cos(phi) ** 2 + Ixeff * np.sin(phi) ** 2

    # Ix0,Iy0 - second moment of area of the plate relative to origin and main axis
    Ix0eff = Ix1 + ycg ** 2 * Aeff
    Iy0eff = Iy1 + xcg ** 2 * Aeff
    Ix0 = Ix1 + ycg ** 2 * A
    Iy0 = Iy1 + xcg ** 2 * A
    Ixyp = 0
    I = [A, Aeff, Ix0, Iy0, Ix0eff, Iy0eff]
    return I


# Input: r,t,xc,yc,theta,phi


i = 0
kN = 1.0
m = 1.0
N = kN / 1000
mm = m / 1000
MPa = N / (mm ** 2)
Area = 8000 * (mm ** 2)
R_S355 = [120 * mm, 170 * mm, 180 * mm]
R_S690 = [120 * mm, 140 * mm, 150 * mm]
Nf = 4  # number of facets
Nfacets = Nf * np.ones(6)  # 9 6 12 number of facets vector

#t = map(lambda R: Area / (2 * R * Nf * np.sin(np.pi / Nf)), R_S355)  # vary thickness to keep area constant

YModulus = 2.1e5 * MPa  # ,2.1e5*MPa,2.1e5*MPa,2.1e5*MPa,2.1e5*MPa,2.1e5*MPa
SteelG = [355 * MPa, 690 * MPa]  # , 'S460' ,460*MPa,690*MPa 355*MPa,
rcoef = 2.5  # t/r thickness over radius of the bending coefficient.
nu = 0.3
lambda_gl = np.linspace(0.5, 1.2, 8)
alpha = 0.49  # buckling curve coefficient
filename = 'Polygons_%s.txt' % Nf
Table_heading = ['Name', 'Length', 'Facets', 'R', 'A', 'Aeff', 'ro', 'c', 'ratio', 'lambdap', 'lambda_gl', 'Npl',
                 'Neff', 'Nbrd',
                 'epsilon', 'class_ic', 'class_t', 'chi', 'chi_tot']
# Table_heading='Name'+'\t'+'Facets'+'\t'+'R'+'\t'+'A'+'\t'+'Aeff'+'\t'+'ro'+'\t'+'c'+'\t'+'ratio'+'\t'+'lambdap'+'\t'+
# 'lambda_gl'+'\t'+'Npl'+'\t'+'Neff'+'\t'+'Nbrd'+'\t'+'epsilon'+'\t'+'class_ic'+'\t'+'class_t'+'\t'+'chi'+'\t'+'chi_tot'
file = open(filename, 'w')

for text in Table_heading:
    text2wr = text + '\t'
    file.write(text2wr)
# print text2wr
# file.write(Table_heading)
file.write('\n')
file.close()
file = open(filename, 'a')
for fy in SteelG:
    if fy == 355 * MPa:
        R = R_S355
    if fy == 690 * MPa:
        R = R_S690
    for Radius in R:
        for c_lambda in lambda_gl:

            i = (Area + 2 * (Radius * 2) ** 2) / (4 * np.sqrt(3) * 2 * Radius)  # only for square hollow sections
            L = np.pi * c_lambda * i * np.sqrt(YModulus / fy)
            # ----------------------------------------------------
            # Create the models

            MName = 'Polyg_S%s_%s_%s' % (int(fy / 1000), int(Radius * 1000), int(Nf))
            # -----------------------------------------------------

            # Create the array for defining the polygon x and y coordinates
            theta = 2 * np.pi / Nf;
            t = Area / (2 * Radius * Nf * np.sin(theta / 2))
            if t < 3 * mm:
                rcoef = 2
            else:
                rcoef = 2.5
            f = np.linspace(0.0, 2 * np.pi, Nf + 1)
            x = []
            y = []
            points = []
            filletpoints = []
            line_points = []
            fillet_start = []
            fillet_end = []
            x_plate_points = []
            y_plate_points = []
            for j in range(len(f) - 1):
                x.append(Radius * np.cos(f[j]))
                y.append(Radius * np.sin(f[j]))
            # ## Bends
            # # Bending radius
            rbend = rcoef * t
            # rbend=3.6*mm
            # # Distance between bending centre and corner
            lc = rbend / np.cos(theta / 2)
            # # Centers of bending arcs
            xc = x - lc * np.cos(f[:-1])
            yc = y - lc * np.sin(f[:-1])

            # Angles of the edges' midlines (measured from x-axis)
            phi_mids = f[:-1] - theta / 2

            # xy coords of the start and end fillet points
            for j in range(len(x)):
                xarc_end = xc[j] + rbend * np.cos(phi_mids[j] + theta)
                yarc_end = yc[j] + rbend * np.sin(phi_mids[j] + theta)
                xarc_start = xc[j] + rbend * np.cos(phi_mids[j])
                yarc_start = yc[j] + rbend * np.sin(phi_mids[j])
                fillet_start.append((xarc_start, yarc_start), )
                fillet_end.append((xarc_end, yarc_end), )
            # xy coordinates of the plate centers
            for j in range(len(x)):
                points.append((x[j], y[j]))
                x_line = (x[j - 1] + x[j]) / 2
                y_line = (y[j - 1] + y[j]) / 2
                line_points.append((x_line, y_line), )
            ## Calculate sigmcr according to EN 1993-1-5
            # aplate - length of the plate between the bending corners
            aplate_r = np.sqrt(
                (fillet_start[2][0] - fillet_end[1][0]) ** 2 + (fillet_start[2][1] - fillet_end[1][1]) ** 2)
            aplate_EC = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)

            # sigmcr - critical load of a plate considering simple supports on the sides
            sigmcr = 4 * (np.pi ** 2) * YModulus * t ** 2 / (12 * (1 - nu ** 2) * aplate_EC ** 2)

            # lambdap - plate slenderness
            lambdap = np.sqrt(fy / sigmcr)

            # verify if reduction is neccessary based on plate slenderness 1993-1-5
            if lambdap < 0.673:
                ro = 1
            else:
                ro = (lambdap - 0.055 * 4) / (lambdap ** 2)
            # Calculate cross-section properties
            Ixt = 0.0
            Iyt = 0.0
            At = 0.0
            Iyteff = 0.0
            Ixteff = 0.0
            Ateff = 0.0
            # # Properties generated by the curved corners

            for xi, yi, phi in zip(xc, yc, phi_mids):
                [A, Ixc, Iyc] = calc_prop_c(rbend, t, xi, yi, theta, phi)  # t-0.05*t
                Ixt = Ixt + Ixc
                Iyt = Iyt + Iyc
                At = At + A
                line = 'Corner angle %s' % phi
                print(line, Ixc, Iyc, Ixt, Iyt)
                Iyteff = 0.0 + Iyt
            Ixteff = 0.0 + Ixt
            Ateff = 0.0 + At
            x_start = []
            y_start = []
            x_end = []
            y_end = []
            for xi, yi in fillet_end:
                x_start.append(xi)
                y_start.append(yi)

            for xi, yi in fillet_start[1:]:
                x_end.append(xi)
                y_end.append(yi)

            x_end.append(fillet_start[0][0])
            y_end.append(fillet_start[0][1])

            # #Calculate according to EN 1993-1-3
            # #------------------------------------------------
            # x_start=[]
            # y_start=[]
            # x_end=[]
            # y_end=[]
            # for xi,yi in zip(x,y):
            # x_start.append(xi)
            # y_start.append(yi)

            # for xi,yi in zip(x[1:],y[1:]):
            # x_end.append(xi)
            # y_end.append(yi)

            # x_end.append(x[0])
            # y_end.append(y[0])
            # #-----------------------------------------------------------------
            # # Properties generated by the plates
            for center_point, phi, xstart, ystart, xend, yend in zip(line_points, phi_mids, x_start, y_start, x_end,
                                                                     y_end):
                [A, Aeff, Ixp, Iyp, Ixeffp, Iyeffp] = calc_prop_p(xstart, ystart, xend, yend, t, center_point, phi, ro)
                Ixt = Ixt + Ixp
                Iyt = Iyt + Iyp
                Iyteff = Iyteff + Iyeffp
                Ixteff = Ixteff + Ixeffp
                At = At + A
                Ateff = Ateff + Aeff
                line = 'Plate angle: %s' % phi
                print(line, Ixp, Iyp, Iyt, Ixt)
            # # Reduction due to corners
            # delta=0.43*(rbend-t/2)*2*theta/np.pi/aplate_EC
            # At=At*(1-delta)
            # Ateff=Ateff*(1-delta)
            # Ixt=Ixt*(1-delta*2)
            # Iyt=Iyt*(1-delta*2)
            # Iyteff=Iyteff*(1-delta*2)
            # Ixteff=Ixteff*(1-delta*2)

            # Calculate resistance based on EN 1993-1-1
            Npl = fy * At
            Neff = fy * Ateff
            Ncr = np.pi ** 2 * YModulus * min(Iyt, Ixt) / L ** 2
            rg = np.sqrt(min(Iyt, Ixt) / At)
            # lambda_gl=(L/rg)/(np.pi*np.sqrt(YModulus[i]/fy))
            # lambda_gl_eff=(L/rg)*np.sqrt(Ateff/At)/(np.pi*np.sqrt(YModulus[i]/fy))
            lambda_gl_eff = np.sqrt(Ateff * fy / Ncr)

            # if Ateff < At:
            fi = 0.5 * (1 + alpha * (lambda_gl_eff - 0.2) + lambda_gl_eff ** 2)
            csi = 1 / (fi + np.sqrt(fi ** 2 - lambda_gl_eff ** 2))
            Nbrd = Neff * csi
            csi_tot = Nbrd / Npl
            # if Ateff == At:
            # fi=0.5*(1+alpha*(lambda_gl-0.2)+lambda_gl**2)
            # csi = 1/(fi+np.sqrt(fi**2-lambda_gl**2))

            # Determine class for internal compressed parts
            ratio = aplate_EC / t
            epsilon = np.sqrt(235 * MPa / fy)
            if ratio <= 33 * epsilon:
                clasa_ic = 1
            elif ratio <= 38 * epsilon:
                clasa_ic = 2
            elif ratio <= 42 * epsilon:
                clasa_ic = 3
            else:
                clasa_ic = 4

            # Determine class for tubuluar sections
            dot = Radius * 2 / t
            if dot <= 50 * (epsilon ** 2):
                clasa_t = 1
            elif dot <= 70 * (epsilon ** 2):
                clasa_t = 2
            elif dot <= 90 * (epsilon ** 2):
                clasa_t = 3
            else:
                clasa_t = 4
            line = [MName, round(L, 3), round(Nf), Radius, int(At*10**6), int(Ateff*10**6), round(ro, 3),
                    round(aplate_EC*10**3), round(ratio, 3), round(lambdap, 3), round(lambda_gl_eff, 3), int(Npl),
                    int(Neff), int(Nbrd), round(epsilon, 2), int(clasa_ic), int(clasa_t), round(csi, 3),
                    round(csi_tot, 3)]
            for text in line:
                text = str(text) + '\t'
                file.write(text)
            file.write('\n')
file.close()
