import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from FiniteVolume2D import do_simulation

def check_global_conservation():
    problem = 2
    L = 1
    Pe = 10
    # N = [5, 10, 20, 40, 80, 160, 320, 640]
    # N = [40, 80, 160, 320, 640]
    N = np.round(np.logspace(1, 3, 13)).astype(int)

    flux_cds = []
    flux_uds = []


    for n in N:
        print(n)
        _, _, _, f_cds, solve_time_cds = do_simulation(n, L, Pe, problem, 'cds', plot=False)
        _, _, _, f_uds, solve_time_uds = do_simulation(n, L, Pe, problem, 'uds', plot=False)
        flux_cds.append(f_cds)
        flux_uds.append(f_uds)

    flux_cds = np.array(np.abs(flux_cds))
    flux_uds = np.array(np.abs(flux_uds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ax1.plot(N, flux_cds, marker = '.', label='CDS')
    ax1.loglog(N, flux_cds, marker = '.', label='CDS')
    ax1.set_xlabel('Number of cells')
    ax1.set_ylabel('Flux')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    ax1.minorticks_on()  # Show minor ticks

    # ax2.plot(N, flux_uds, marker = '.', label='UDS')
    ax2.loglog(N, flux_uds, marker = '.', label='UDS')
    ax2.set_xlabel('Number of cells')
    ax2.set_ylabel('Flux')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    ax2.minorticks_on()  # Show minor ticks

    plt.tight_layout()
    plt.show()
    

def check_flux_west_wall():
    problem = 2
    L = 1
    Pe = 7
    N = [40, 80, 160, 320]

    dT_w_cds = []
    dT_w_uds = []

    for n in N:
        print(n)
        _, _, dT_cds, _, solve_time_cds = do_simulation(n, L, Pe, problem, 'cds', plot=False)
        DF_cds = [np.mean(dT_cds[1:-1, 0]), np.mean(dT_cds[1:-1, -1]), np.mean(dT_cds[0, 1:-1]), np.mean(dT_cds[-1, 1:-1])]
        _, _, dT_uds, _, solve_time_uds = do_simulation(n, L, Pe, problem, 'uds', plot=False)
        DF_uds = [np.mean(dT_uds[1:-1, 0]), np.mean(dT_uds[1:-1, -1]), np.mean(dT_uds[0, 1:-1]), np.mean(dT_uds[-1, 1:-1])]
        dT_w_cds.append(DF_cds[0])
        dT_w_uds.append(DF_uds[0])

    dT_w_cds = np.array(dT_w_cds)
    dT_w_uds = np.array(dT_w_uds)

    plt.figure(figsize=(10, 5))
    plt.semilogx(N, dT_w_cds, marker = '.', label='CDS')
    plt.semilogx(N, dT_w_uds, marker = '.', label='UDS')
    plt.xlabel('Number of cells')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    plt.minorticks_on()  # Show minor ticks
    plt.tight_layout()
    plt.show()






if __name__=="__main__":
    check_global_conservation()
    # check_flux_west_wall()
    print("Done")