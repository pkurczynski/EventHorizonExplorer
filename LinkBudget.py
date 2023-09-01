# #!/usr/bin/env python
"""
LinkBudget.py                                       Python3 script

A link budget analysis tool for the Event Horizon Explorer mission concept,
optical communications architecture. Based on Wang, J. et al. 2023 SPIE


usage: LinkBudget.py [-h] [-v]

Computes optical communications link budget analysis for Event Horizon
Explorer

options:
  -h, --help     show this help message and exit
  -v, --verbose  Print descriptive text output to console.

    
    
Version 1

pk 8/23/2023
"""
import argparse
import astropy
import astropy.units as u
import numpy as np
from datetime import datetime
import sys
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def LinkEquation(D_rx, D_tx, eta_rx, eta_tx, eta_atm, eta_misc, CarrierWavelength, LinkDistance, P_tx, verbose = False):
    """
    Computes received power of optical communications link as a function
    of transmitter, receiver and link properties
    
    Implements Equation [1] from Wang, J. et al. 2023, SPIE 12413
    

    Parameters
    ----------
    D_rx : Quantity
        Receiver aperture, cm.
    D_tx : Quantity
        Transmit aperture, cm.
    eta_rx : Quantity
        Receiver coupling efficiency, dB.
    eta_tx : Quantity
        Transmit efficiency, dB.
    eta_atm: Quantity
        Atmospheric loss, dB.
    eta_misc: Quantity
        Miscellaneous additional loss, dB
    CarrierWavelength : Quantity
        carrier wavelength, micron.
    LinkDistance : Quantity
        Distance between transmitter and receiver, km.
    P_tx : Quantity
        Transmitter power, dBm.
    verbose : boolean, optional
        Print diagnostics to console. The default is False.

    Returns
    -------
    P_rx : float
        Receiver power, dBm.

    """

    """
    Total loss in dB
    """
    eta_total = eta_rx + eta_tx + eta_atm + eta_misc


    """
    Compute receiver power using Equation [1] from Wang et al. 2023 Vol. 12413
    
    P_rx = eta_rx * eta_tx * 
        ( ( pi * D_rx * D_tx ) / (4 * Wavelength * LinkDistance) )^2 * P_tx
    
    Quantities above are in physical units. Convert to dB by 
    taking logarithm of both sides, and multiply by 10, using
    GeometryFactor = ( pi * D_rx * D_tx ) / (4 * Wavelength * LinkDistance).
    P_rx[dBm] = 10 * log( P_rx[mW] )
    P_tx[dBm] = 10 * log( P_tx[mW] )
    etc.
    """
    
    dBm = u.dB(u.mW)
    P_rx = -9999 * dBm
    
    GeometryFactor = float(np.pi * D_rx * D_tx / (4 * CarrierWavelength * LinkDistance))    
    P_rx = P_tx + eta_total + 20 * np.log10( GeometryFactor) * dBm
    P_rx = P_rx.value * dBm
       
    
    if verbose:
        print("LinkEquation():")
        print("\tINPUT")
        print("\tReceiver aperture            : ",D_rx)
        print("\tTransmitter aperture         : ",D_tx)
        print("\t\tSources of loss")
        print("\tReceiver coupling efficiency : ",eta_rx)
        print("\tTransmit efficiency          : ",eta_tx)
        print("\tAtmospheric loss             : ",eta_atm)
        print("\tMiscellaneous loss           : ",eta_misc)
        print("\t    TOTAL Loss               : ",eta_total)
        print("\tCarrier wavelength           : ",CarrierWavelength)
        print("\tLink Distance                : ",LinkDistance)
        print("\tTransmit Power               : ",P_tx)
        print("\tTransmit Power               : ",f"{P_tx.to(u.W):.2f}")
        print("\n")
        print("\tOUTPUT")
        print("\tReceiver power               :",f"{P_rx:.3f}")
        print("\tReceiver power               :",f"{P_rx.to(u.nW):.2f}")
        print("\n")
    
    return P_rx


def DataRateFromReceivedPower(P_rx, Frequency, Sensitivity, verbose = False):
    """
    Computes data rate of an optical communications link as a function
    of received power and other parameters.
    
    Implements Equation [2] from Wang et al 2023 SPIE Vol 12413

    Parameters
    ----------
    P_rx : Quantity
        Receiver Power, dBm.
    Frequency : Quantity
        Carrier Frequency, Hz.
    Sensitivity : Float
        Sensitivity, photons per bit.
    verbose : boolean, optional
        Print diagnostics to console. The default is False.

    Returns
    -------
    Data rate, Gigabits per second.

    """

    R = P_rx.to(u.W)/(Frequency.to(u.J, equivalencies = u.spectral()) * Sensitivity)

    R_Gbps = R.value / 1e9
    
    if verbose:
        print("DataRateFromReceivedPower():")
        print("\tINPUT")
        print("\tReceiver power               : ",f"{P_rx:.3f}")
        print("\tCarrier frequency            : ",f"{Frequency:.3E}")
        print("\tSensitivity, ppb             : ",Sensitivity)
        print("\n")
        print("\tOUTPUT")
        print("\tData rate, Gbps              : ",f"{R_Gbps:.3f}")
        print("\n")
    
    return R_Gbps


def DataRate(D_rx, D_tx, eta_rx, eta_tx, eta_atm, eta_misc, CarrierWavelength, LinkDistance, P_tx, Sensitivity, verbose = False):
    """
    Computes data rate as a function of input parameters by utlizing
    functions LinkEquation() and DataRateFromReceivedPower()
    

    Parameters
    ----------
    D_rx : Quantity
        Receiver aperture, cm.
    D_tx : Quantity
        Transmit aperture, cm.
    eta_rx : Quantity
        Receiver coupling efficiency, dB.
    eta_tx : Quantity
        Transmit efficiency, dB.
    eta_atm: Quantity
        Atmospheric loss, dB.
    eta_misc: Quantity
        Miscellaneous additional loss, dB
    CarrierWavelength : Quantity
        carrier wavelength, micron.
    LinkDistance : Quantity
        Distance between transmitter and receiver, km.
    P_tx : Quantity
        Transmitter power, dBm
    Sensitivity : Float
        Sensitivity, photons per bit.
    verbose : boolean, optional
        Print diagnostics to console. The default is False.

    Returns
    -------
    None.

    """
    
    
    P_rx = LinkEquation(D_rx, 
                        D_tx, 
                        eta_rx, 
                        eta_tx,
                        eta_atm,
                        eta_misc,
                        CarrierWavelength,
                        LinkDistance, 
                        P_tx, 
                        verbose)
    
    
    R = DataRateFromReceivedPower(P_rx, 
                 CarrierWavelength.to(u.Hz,equivalencies = u.spectral()), 
                 Sensitivity, 
                 verbose)

    return R


def DataRateGrid(D_rx, D_tx, eta_rx, eta_tx, eta_atm, eta_misc, CarrierWavelength, LinkDistance, P_tx, Sensitivity, verbose = False):
    """
    Computes data rate for a multi-dimensional grid of values based
    on the input parameter list

    Parameters
    ----------
    D_rx : Quantity
        Receiver aperture, cm.
    D_tx : Quantity
        Transmit aperture, cm.
    eta_rx : Quantity
        Receiver coupling efficiency, dB.
    eta_tx : Quantity
        Transmit efficiency, dB.
    eta_atm: Quantity
        Atmospheric loss, dB.
    eta_misc: Quantity
        Miscellaneous additional loss, dB
    CarrierWavelength : Quantity
        carrier wavelength, micron.
    LinkDistance : Quantity
        Distance between transmitter and receiver, km.
    P_tx : Quantity
        Transmitter power, dBm
    Sensitivity : Float
        Photons per bit (eg 5)
    verbose : TYPE, optional
        The default is False.

    Returns
    -------
    DataRateArray : Float array
        Data rate in Gbps for each set of parameters of the tradespace

    """

    DataRateArray = np.empty(shape = [len(D_rx), 
                              len(D_tx), 
                              len(eta_rx),
                              len(eta_tx),
                              len(eta_atm),
                              len(eta_misc),
                              len(CarrierWavelength),
                              len(LinkDistance), 
                              len(P_tx), 
                              len(Sensitivity)])
 

    for j1, D_rx_ in enumerate(D_rx):
        for j2, D_tx_ in enumerate(D_tx):
            for j3, eta_rx_ in enumerate(eta_rx):
                for j4, eta_tx_ in enumerate(eta_tx):
                    for j5, eta_atm_ in enumerate(eta_atm):
                        for j6,eta_misc_ in enumerate(eta_misc):
                            for j7, CarrierWavelength_ in enumerate(CarrierWavelength):
                                for j8, LinkDistance_ in enumerate(LinkDistance):
                                    for j9, P_tx_ in enumerate(P_tx):
                                        for j10, Sensitivity_ in enumerate(Sensitivity) :
                                            
                                            DataRateArray[j1,j2,j3,j4,j5,j6,j7,j8,j9,j10] = DataRate(D_rx_,
                                                         D_tx_,
                                                         eta_rx_, 
                                                         eta_tx_,
                                                         eta_atm_,
                                                         eta_misc_,
                                                         CarrierWavelength_,
                                                         LinkDistance_,
                                                         P_tx_,
                                                         Sensitivity_,
                                                         verbose = False)
                        
    return DataRateArray


if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description="Computes optical communications link budget analysis for Event Horizon Explorer")
    parser.add_argument('-v','--verbose', action = 'store_true', help='Print descriptive text output to console.')
    
    args = parser.parse_args()
    
    if (args.verbose):
        print("LinkBudget.py \n")
        
        print("Run date and time :  ", str(datetime.now()))
        print("Python Version    :  ", sys.version)
        print("Astropy version   :  ", astropy.__version__)
        print("\n")
        
    """
    Code validation:
    
    Wang et al 2023 SPIE 
    
    Figure 3: Downlink from GEO (40,000 km)

    Parameter         Wang+2013 A   THIS CODE   Wang+2013 B    THIS CODE
        
    LinkDistance      40,000 km     40,000 km   40,000 km      40,000 km
    
    Space Aperture    10 cm         10 cm       2 cm           2 cm
    Power             10 W          10 W        10 W           10 W
    Data rate         100 Gbps      244.897     100 Gbps       244.897 Gbps
    Ground Aperture   60 cm         60 cm       3 m            3 m
                
    
    NB: This code produces results that are discrepant from Wang+2023
    by about a factor of two (this code: 244.897 Gbps vs 100 Gbps Wang+2023)
    
    Figure 4 (see paragraph above Figure 4)
    
    Parameter         Wang+2023 C  THIS CODE    Wang+2023 D    THIS CODE
    
    Ground Aperture   60 cm        60 cm        60 cm          60 cm
    Total losses      -15 dB       -15.66 dB    -15 dB         -15.66 dB
    Transmit Power    39.5 dBm     39.5 dBm     42.5 dBm       42.5 dBm
    Sensitivity (ppb)   5            5            5              5
    
    Data rate (Gbps)  100 Gbps     218.265      200            435.496
    

    """
    
    """
    Set of parameters for computing data rate
    """
    D_rx = 60.0 * u.centimeter
    D_tx = 10.0 * u.centimeter
    eta_rx = -8.9 * u.dB
    eta_tx = -5.1 * u.dB
    eta_atm = -1.0 * u.dB
    eta_misc = -0.66 * u.dB
    CarrierWavelength = 1.55 * u.micron
    LinkDistance = 40000.0 * u.km
    
    dBm = u.dB(u.mW)
    P_tx = 40.0 * dBm
    
    Sensitivity = 5.0

    """
    Compute data rate for a single set of parameters
    """
    R = DataRate(D_rx, 
                 D_tx, 
                 eta_rx, 
                 eta_tx,
                 eta_atm,
                 eta_misc,
                 CarrierWavelength,
                 LinkDistance, 
                 P_tx,
                 Sensitivity, 
                 verbose = args.verbose)
    
    """
    Define the tradespace to explore for all relevant parameters
    """
    D_rx_array = np.linspace(60.0,100,9) * u.centimeter
    D_tx_array = np.linspace(10.0,15,9) * u.centimeter
    eta_rx_array = [-8.9] *u.dB
    eta_tx_array = [-5.1] * u.dB
    eta_atm_array = [-1.0] * u.dB
    eta_misc_array = [-0.66] * u.dB
    CarrierWavelengthArray = [1.55] * u.micron
    LinkDistanceArray = np.linspace(40000,50000,9) * u.km
    P_tx_array = np.linspace(40.0,43,9) * u.dB(u.mW)
    SensitivityArray = [5.0]
    
    """
    Compute DataRate for the entire tradespace
    """
    if (args.verbose):
        print("computing DataRateArray...\n")
    
    DataRateArray = DataRateGrid(D_rx_array,
                                 D_tx_array,
                                 eta_rx_array, 
                                 eta_tx_array,
                                 eta_atm_array,
                                 eta_misc_array,
                                 CarrierWavelengthArray,
                                 LinkDistanceArray,
                                 P_tx_array,
                                 SensitivityArray,
                                 verbose = False)

    """
    Plot the data rate tradespace
    """
    
    Z = DataRateArray[:,:,0,0,0,0,0,0,0,0]
    X = np.tile(D_rx_array.value,(len(D_rx_array),1))
    Yt = np.tile(D_tx_array.value,(len(D_tx_array),1))
    Y=Yt.transpose()
    
 
    ax = plt.figure().add_subplot(projection='3d', computed_zorder = False)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    ax.contourf(X, Y, Z, zdir='z', offset=Z.min(), cmap='coolwarm', zorder = 4)
    ax.contourf(X, Y, Z, zdir='x', offset=X.min(), cmap='coolwarm', zorder = 5)
    ax.contourf(X, Y, Z, zdir='y', offset=Y.max(), cmap='coolwarm', zorder = 6)

    #ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
    #   xlabel='X', ylabel='Y', zlabel='Z')

    # Plot the 3D surface
    ax.plot_surface(X,Y,Z, 
                   edgecolor='royalblue', 
                   lw=0.5, 
                   rstride=8, 
                   cstride=8,
                   alpha=0.5,
                   zorder = 7)


    ax.set(xlabel = 'Rx Aperture, cm', 
           ylabel = 'Tx Aperture, cm', 
           zlabel = 'Data Rate, Gbps')
    
    plt.show()
    
    
    if (args.verbose):
        print("Done!")