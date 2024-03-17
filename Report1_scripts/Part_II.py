import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from FiniteVolume2D import do_simulation

def convergence_rate():
    # Corrosponds to Part II 1a
    L = 1
    Pe = 10
    problem = 1
    scheme = ['cds', 'uds']
    Texact = lambda x: (np.exp(Pe*x)-1)/(np.exp(Pe)-1)

    n = np.array([10, 20, 40, 80, 160, 320])
    ERROR = np.zeros((len(n), len(scheme)))

    for j, fvscheme in enumerate(scheme):
        for i, N in enumerate(n):
        
            T, TT, solve_time, Flux, solve_time, xc = do_simulation(N, L, Pe, problem, fvscheme, plot = False)
            T_exact = Texact(xc)

            ERROR[i,j] = np.max(abs(T[2,:]-T_exact))
            
            #x_exact = np.linspace(np.min(xc), np.max(xc), 100)
            #if j == 1:
            #    plt.plot(x_exact, Texact(x_exact), label = 'Exact solution')

            plt.plot(xc, T[2,:], marker = '.',  label = 'T and n = ' + str(N) + ' and fvscheme = ' + fvscheme)
    
    slope_cds = np.polyfit(np.log(n), np.log(ERROR[:,0]), 1)[0]
    slope_uds = np.polyfit(np.log(n), np.log(ERROR[:,1]), 1)[0]

    print('The slope for the CDS scheme is: ', slope_cds)
    print('The slope for the UDS scheme is: ', slope_uds)
    plt.legend()
    plt.figure()
    plt.loglog(n, ERROR[:,0], label = 'cds')
    plt.loglog(n, ERROR[:,1], label = 'uds')
    plt.grid()
    plt.legend()
    return TT

def check_global_conservation():
    # Corrosponds to Part II 2a
    problem = 2
    L = 1
    Pe = 10
    N = [5, 10, 20, 40, 80, 160, 320]
    # N = np.round(np.logspace(1, 3, 10)).astype(int)

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
    for n, f in zip(N, flux_cds):
        print(n, f)
    for n, f in zip(N, flux_uds):
        print(n, f)

    # Plot for CDS
    plt.figure(figsize=(7, 5))
    plt.loglog(N, flux_cds, marker='.', label='CDS')
    plt.xlabel('Number of cells')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, which='both', linestyle='dashed')
    plt.minorticks_on()
    plt.tight_layout()

    # Plot for UDS
    plt.figure(figsize=(7, 5))
    plt.loglog(N, flux_uds, marker='.', label='UDS')
    plt.xlabel('Number of cells')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, which='both', linestyle='dashed')
    plt.minorticks_on()
    plt.tight_layout()
    
def check_flux_west_wall():
    # Corrosponds to Part II 2a
    problem = 2
    L = 1
    Pe = 10
    N = [40, 80, 160, 320]
    # N = np.round(np.logspace(1, 3, 10)).astype(int)

    dT_w_cds = []
    dT_w_uds = []

    for n in N:
        print(n)
        _, _, dT_cds, _, solve_time_cds, _ = do_simulation(n, L, Pe, problem, 'cds', plot=False)
        DF_cds = [np.sum(dT_cds[1:-1, 0])*L/n, np.sum(dT_cds[1:-1, -1])*L/n, np.sum(dT_cds[0, 1:-1])*L/n, np.sum(dT_cds[-1, 1:-1])*L/n]
        _, _, dT_uds, _, solve_time_uds, _ = do_simulation(n, L, Pe, problem, 'uds', plot=False)
        DF_uds = [np.sum(dT_uds[1:-1, 0])*L/n, np.sum(dT_uds[1:-1, -1])*L/n, np.sum(dT_uds[0, 1:-1])*L/n, np.sum(dT_uds[-1, 1:-1])*L/n]
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

def get_west_flux_uds():
    # Corrosponds to Part II 2b
    problem = 2
    L = 1
    Pe = np.round(np.logspace(0, 8, num=10))
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
            _DF_w = np.sum(dT_uds[1:-1, 0])*L/n
            DF_w.append(_DF_w)

        ax.semilogx(Pe, DF_w, marker='.', label=f'N = {n}')
    
    ax.set_xlabel('Peclet number')
    ax.set_ylabel('Flux')
    ax.legend()
    ax.grid(True, which='both', linestyle='dashed')  # Add dashed gridlines for both major and minor ticks
    ax.minorticks_on()  # Show minor ticks
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))  # Set minor tick locator

    

    plt.tight_layout()



if __name__=="__main__":
    convergence_rate()
    check_global_conservation()
    check_flux_west_wall()
    get_west_flux_uds()
    plt.show()