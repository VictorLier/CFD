import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from FiniteVolume2D import do_simulation

def check_global_conservation():
    problem = 2
    L = 1
    Pe = 10
    # N = [5, 10, 20, 40, 80, 160, 320, 640]
    N = [40, 80, 160, 320]
    # N = np.round(np.logspace(1, 3, 13)).astype(int)

    flux_cds = []
    flux_uds = []


    for n in N:
        print(n)
        _, _, _, f_cds, solve_time_cds, _ = do_simulation(n, L, Pe, problem, 'cds', plot=False)
        _, _, _, f_uds, solve_time_uds, _ = do_simulation(n, L, Pe, problem, 'uds', plot=False)
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
    

def check_flux_west_wall():
    problem = 2
    L = 1
    Pe = 10
    N = [5, 10, 20, 40, 80, 160, 320, 640]

    dT_w_cds = []
    dT_w_uds = []

    for n in N:
        print(n)
        _, _, dT_cds, _, solve_time_cds, _ = do_simulation(n, L, Pe, problem, 'cds', plot=False)
        DF_cds = [np.mean(dT_cds[1:-1, 0]), np.mean(dT_cds[1:-1, -1]), np.mean(dT_cds[0, 1:-1]), np.mean(dT_cds[-1, 1:-1])]
        _, _, dT_uds, _, solve_time_uds, _ = do_simulation(n, L, Pe, problem, 'uds', plot=False)
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
    #plt.legend()
    plt.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    plt.minorticks_on()  # Show minor ticks
    plt.tight_layout()


def get_west_flux_uds():
    problem = 2
    L = 1
    Pe = np.round(np.logspace(0, 8, num=20))
    N = [50, 100, 150, 200, 250]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Adding lines at flux = 100, 200, 300, 400
    for i in range(1, len(N) + 1):
        ax.axhline(y=100*i, color='k', linestyle='-', linewidth=1)

    for n in N:
        DF_w = []
        for _Pe in Pe:
            print(n, _Pe)
            _, _, dT_uds, _, _, _ = do_simulation(n, L, _Pe, problem, 'uds', plot=False)
            _DF_w = np.mean(dT_uds[1:-1, 0])
            DF_w.append(_DF_w)

        ax.semilogx(Pe, DF_w, marker='.', label=f'N = {n}')
    
    ax.set_xlabel('Peclet number')
    ax.set_ylabel('Flux')
    #ax.legend()
    ax.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    ax.minorticks_on()  # Show minor ticks
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))  # Set minor tick locator

    

    plt.tight_layout()

        

            



if __name__=="__main__":
    # check_global_conservation()
    check_flux_west_wall()
    # get_west_flux_uds()
    import tikzplotlib
    tikzplotlib.save("WestFlux.tex")

    plt.show()