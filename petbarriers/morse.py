import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

sns.set_style('white')
sns.set_palette(sns.hls_palette(6, h=0.5,l=0.4,s=0.5))
font = {'size': 16}
plt.rc('font',**font)

num_dict = {'0': '$_{0}$', '1': '$_{1}$', '2': '$_{2}$', '3': '$_{3}$', '4': '$_{4}$',
            '5': '$_{5}$', '6': '$_{6}$', '7': '$_{7}$', '8': '$_{8}$', '9': '$_{9}$',
            '+': '$^{+}$'}
# usage
SUB = str.maketrans(num_dict)
# print("H2SO4+".translate(SUB))

avg_dist = 2.5432441257325
dproton = avg_dist
# total energy, d(H-metal), d(water-metal), d(H2O-metal), E_el_ads_H
energy_dict = {'H2' : [-32.9419840999],
            'H' : [-13.1625208331],
            'Pt' : [-27082.9269032, 1.567, avg_dist, 3.0095288764558106, -0.25],
            'Ag' : [-35323.6277998, 1.687, avg_dist, 2.801355592387848, 0.49],
            'Au' : [-31772.4252654, 1.613, avg_dist, 2.6987522336206533, 0.52],
            'Rh' : [-86064.3177733, 1.594, avg_dist, 3.2374964553755508, -0.31]}


def morse_norm(r, a):
    return((1-np.exp(-a*r))**2)

def morse_diff(a,b,r):
    y1 = (1-np.exp(-a*r))**2
    y2 = (1-np.exp(-b*r))**2
    return(y1-y2)


class PES:
    """Handle data from a proton donor.
    Args:
        filepath (str) : path to data.
        plot: Whether to plot data after read in.

    Attributes:
        donor: 'left' or right, determines curve position.
        df: dataframe with data

    """

    def __init__(self, filepath, donor='left', dHeq=1., gH=0., plot=False):
        self.filepath = filepath
        self.donor = donor
        self.dH = dHeq
        self.preprocess()
        self.normalize()
        self.fit_morse()
        self.gH = gH - self.De

        if plot:
            # Plot proton donor data.
            fig, ax = plt.subplots()
            ax.plot(self.df.distance, self.df.E_tot)
            ax.set_xlabel('distance')
            ax.set_ylabel('Energy')
            ax.legend()
            ax.set_title(self.donor.translate(SUB) + ' normalized PES')
            plt.show()

    def preprocess(self):
        # Read in and preprocess data
        df = pd.read_csv(self.filepath, sep='\t', header=None)
        df.columns = ['distance', 'E_tot']

        # H2O ata are noisy at large d.
        if not self.donor == 'left':
            for i in range(10):
                df = self._smoothen(df)

        # Get dissociation energy and set Emin = 0
        df.E_tot = df.E_tot - df.E_tot.min()
        self.De = df.E_tot.max()

        self.df = df
        return None

    def normalize(self):
        df = self.df.copy(deep=True)

        # Min-Max scaling to 0-->1
        df.E_tot = df.E_tot / self.De

        # Get distance at minimum energy and set to 0.
        dmin = df[df.E_tot == df.E_tot.min()].distance.get_values()[0]
        if not self.donor == 'left':
            df.distance = dmin - df.distance
            dmax = df.distance.max() / 1.1
            df = df[df.distance < dmax]
        else:
            df.distance = df.distance - dmin

        self.df = df
        return None

    def fit_morse(self):
        # Fit to get a
        popt, pcov = curve_fit(morse_norm, self.df.distance, self.df.E_tot)
        perr = np.sqrt(np.diag(pcov))
        self.a = popt
        self.leftrror = perr

    def morse(self, r=None):
        # Define the morse potential for this proton donor.
        if r is None:
            r = self.df.distance
        if not self.donor == 'left':
            return (self.De * (1 - np.exp(-self.a * (self.dH - r)))**2 + self.gH)
        else:
            return (self.De * (1 - np.exp(-self.a * (r - self.dH)))**2 + self.gH)

    def morse_norm(self, r):
        return (1 - np.exp(-self.a * r))**2

    def plot_morse(self):
        fig, ax = plt.subplots()
        if self.donor == 'left':
            ax.plot(self.df.distance + self.dH, self.df.E_tot * self.De + self.gH)
        else:
            ax.plot(self.dH - self.df.distance, self.df.E_tot * self.De + self.gH)
        ax.plot(self.df.distance, self.morse(), '--',
                label='Morse: a=%5.3f' % tuple(self.a))
        ax.set_xlabel('distance')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.set_title(self.donor.translate(SUB) + ' morse fit')
        ax.set_ylim((-7, 7))
        plt.show()
        return None

    def _smoothen(self, df):
        df_new = df.copy(deep=True)
        for idx, entry in enumerate(df.E_tot):
            if idx == 0:
                previos_entry = entry
            else:
                dE = entry - previos_entry
                if dE < 0:
                    df_new = df_new.drop([idx], axis=0)
                previos_entry = entry
        df_new = df_new.reset_index(drop=True)
        return (df_new)

#%%
class Energy:
    """Compute diabatic and adiabatic energy curves.
    Args:
        pes1 and pes2 : PES objects.
    """

    def __init__(self, left, right):
        self.x = np.linspace(-10., 10., 1000)
        self.left = left
        self.right = right
        self.r_corr = np.linspace(self.left.dH, self.right.dH, 1000)
        self._adiabatic_correction()

    def morse_left(self, r=None):
        if r is None:
            r = self.x
        return (self.left.De * (1 - np.exp(-self.left.a * (r - self.left.dH))) ** 2 + self.left.gH)

    def morse_right(self, r=None):
        if r is None:
            r = self.x
        return (self.right.De * (1 - np.exp(-self.right.a * (self.right.dH - r))) ** 2 + self.right.gH)

    def interception(self, adiabatic=False, plot=False):
        a = self.morse_left()
        b = self.morse_right()

        # Find intercept
        xint_list = list(self.x[np.argwhere(np.diff(np.sign(a - b))).flatten()])
        yint_list = []
        for xi in xint_list:
            yint_list.append(self.morse_left(r=xi)[0])
        val, idx = min((val, idx) for (idx, val) in enumerate(yint_list))
        self.xint = xint_list[idx]
        self.yint = val
        self.Ea_left = val + self.left.De
        self.Ea_right = val + self.right.De

        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot diabatic curves and intercept
            ax.plot(self.x, a, label='dia-left')
            ax.plot(self.x, b, label='dia-right')
            ax.plot([self.xint], [self.yint], marker='o', markersize=6.0,
                    c='grey', markeredgecolor='k', ls='',
                    label='E$^{a}_{left}$ = %5.2f eV' % self.Ea_left)
            ax.plot([self.xint], [self.yint], marker='o', markersize=6.0,
                    c='grey', markeredgecolor='k', ls='',
                    label='E$^{a}_{right}$ = %5.2f eV' % self.Ea_right)

            # Plot adiabatic curves and intercept
            if adiabatic:
                ax.plot(self.r_corr, self.adia_left, label='adia-left')
                ax.plot(self.r_corr, self.adia_right, label='adia-right')
                ax.plot([self.xint_ad], [self.yint_ad], marker='o', markersize=6.0,
                        c='w', markeredgecolor='b', ls='',
                        label='E$^{a}_{left}$ = %5.2f eV' % (self.Ea_ad_left))
                ax.plot([self.xint_ad], [self.yint_ad], marker='o', markersize=6.0,
                        c='w', markeredgecolor='b', ls='',
                        label='E$^{a}_{right}$ = %5.2f eV' % (self.Ea_ad_right))

            ax.set_xlabel('Distance to metal surface (Å)')
            ax.set_ylabel('Energy (eV)')

            ax.set_xlim((0.5, 4.))
            ax.set_ylim((-8, 4))

            ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
            plt.show()
        return (self.xint, self.yint)

    def _adiabatic_correction(self):
        # Get Gamma values between 0 and 1 for all distances of interest
        gamma_left = self.left.morse_norm(self.r_corr - self.left.dH)       # 0-->0.66
        gamma_right = self.right.morse_norm(self.right.dH - self.r_corr)    # 0.66-->0

        # Morse values
        E_stretch_left = gamma_left * self.left.De + self.left.gH
        E_stretch_right = gamma_right * self.right.De + self.right.gH

        # E_hyb
        self.adia_left = E_stretch_left + gamma_left * (1-gamma_right) * (self.right.gH - E_stretch_left)
        self.adia_right = E_stretch_right + gamma_right * (1-gamma_left) * (self.left.gH - E_stretch_right)

        # Find transition state
        xts_ad = list(self.r_corr[np.argwhere(np.diff(np.sign(self.adia_left - self.adia_right))).flatten()])
        yts_ad = list(self.adia_left[np.argwhere(np.diff(np.sign(self.adia_left - self.adia_right))).flatten()])
        val, idx = min((val, idx) for (idx, val) in enumerate(yts_ad))
        self.xint_ad = xts_ad[idx]
        self.yint_ad = val

        self.Ea_ad_left = self.yint_ad + self.left.De
        self.Ea_ad_right = self.yint_ad + self.right.De

        return None


if __name__ == '__main__':
    print('Executed without errors.')



