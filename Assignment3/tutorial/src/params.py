"""Tutorial data (Stability lobes) - finish milling of a thin 6061-T6 wall."""
import numpy as np

D     = 20e-3            # tool diameter [m]
Z     = 3                # number of teeth
ETA   = np.radians(45)   # helix angle [rad]
AE    = 1e-3             # radial depth of cut [m]

W1    = 1068.1           # natural pulsation of the wall [rad/s]
M1    = 2.63             # modal mass [kg]
XI    = 0.04             # damping ratio [-]

KT    = 550e6            # tangential cutting coefficient [N/m^2]
KR    = 150e6            # radial cutting coefficient [N/m^2]

MODE  = "down"           # milling mode used in both models (slide 20 gives the down-milling angles)

# derived
F1    = W1 / (2 * np.pi)         # natural frequency [Hz]
K1    = M1 * W1**2               # stiffness [N/m]
KR_KT = KR / KT                  # reduced radial coefficient [-]
R     = D / 2
AE_D  = AE / D
