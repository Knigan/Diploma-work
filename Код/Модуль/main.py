import time
import chaospy
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

import metrics
import oscillator

sns.set_context('notebook', font_scale=1.2)
sns.set_style('ticks')
sns.set_palette('bright')
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/content-creation/main/nma.mplstyle")
warnings.filterwarnings("ignore", category=UserWarning)


def example1(**kwargs):
    """
    Example 1.1: Damped linear oscillator model. PC
    expansion using the least squares method.
    """

    # Define default values
    default_values = {'train_size': 750, 'test_size':
     250, 'mu': 0.8, 'sigma': 0.2, 't_min': 0,
    't_max': 30, 'coord_num': 1000, 'max_order': 2,
    'plot': True, 'quantiles': True}

    # Define and initialize local variables
    for key, value in default_values.items():
        if key not in kwargs:
            kwargs[key] = default_values[key]

    # Define the damped linear oscillator model with default settings
    model = oscillator.DampedOscillator()
    model.solve()

    # Define the distribution of the damping factor
    c_dist = chaospy.Normal(kwargs['mu'], kwargs['sigma'])

    # Sample from the distribution using latin hypercube experimental design
    c_sample = c_dist.sample(kwargs['train_size'] +
                             kwargs['test_size'],
                             rule='latin_hypercube')

    # Define the time point sequence
    coordinates = np.linspace(kwargs['t_min'],
                              kwargs['t_max'],
                              kwargs['coord_num'])

    # Evaluate the model at the experimental design points
    model_eval = np.array([model.compute_displacement(p,
                 coordinates) for p in c_sample])

    # Create train and test sets
    train_sample, test_sample, model_train_eval, model_test_eval = train_test_split(c_sample,
                                       model_eval,
                      test_size=kwargs['test_size'],
                      train_size=kwargs['train_size'])

    # Define the PC basis
    chaos_basis = chaospy.generate_expansion(
                  kwargs['max_order'], c_dist,
                  normed=True)
    # Define the PC metamodel
    metamodel = chaospy.fit_regression(chaos_basis,
                train_sample.reshape(1, -1),
                model_train_eval)

    # Evaluate the metamodel at the testing points
    metamodel_test_eval = metamodel(test_sample).T

    # Make comparative plots
    if kwargs['plot']:
        fig, ax = plt.subplots(nrows=3, ncols=1,
                  figsize=(10, 13))
        ax[0].plot(coordinates, model_test_eval.T,
              alpha=0.01, color='red')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('$\\mathcal{M}(t, c)$')
        ax[0].set_title('Оценки модели при случайном параметре $c$')

        ax[1].plot(coordinates, metamodel_test_eval.T,
                   alpha=0.01, color='blue')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('$\\mathcal{M}_{PC}(t, c)$')
        ax[1].set_title('Оценки метамодели ПХ при случайном параметре $c$')

        ax[2].plot(coordinates, model_test_eval.T,
                   alpha=0.01, color='red')
        ax[2].plot(coordinates, metamodel_test_eval.T,
                   alpha=0.01, color='blue')
        ax[2].set_xlabel('t')
        ax[2].set_ylabel('$y$')
        ax[2].set_title('Совмещённые оценки')
        plt.savefig('results/Oscillator evaluation.jpg',
                    dpi=500)

    # Compute the errors
    e_loo = metrics.leave_one_out_error(test_sample,
            chaos_basis, model_test_eval,
            metamodel_test_eval)
    r_sqr = metrics.r_squared(model_test_eval,
            metamodel_test_eval)
    e_mae = metrics.mean_absolute_error(model_test_eval,
            metamodel_test_eval)

    e_rmae = metrics.relative_mean_absolute_error(
             model_test_eval, metamodel_test_eval)

    # Make comparative plots for the errors
    if kwargs['plot']:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax[0, 0].plot(coordinates, e_loo, linewidth=2,
                      color='forestgreen')
        ax[0, 0].set_xlabel('t')
        ax[0, 0].set_ylabel('$E_{LOO}$')
        ax[0, 0].set_title('Ошибка leave-one-out')

        ax[0, 1].plot(coordinates, r_sqr, linewidth=2,
                      color='forestgreen')
        ax[0, 1].set_xlabel('t')
        ax[0, 1].set_ylabel('$R^2$')
        ax[0, 1].set_title('Коэффициент детерминации')
        ax[0, 1].set_yticklabels(
                 [0.8, 0.85, 0.9, 0.95, 1.0])

        ax[1, 0].plot(coordinates, e_mae, linewidth=2,
                      color='forestgreen')
        ax[1, 0].set_xlabel('t')
        ax[1, 0].set_ylabel('$MAE$')
        ax[1, 0].set_title('Средняя абсолютная ошибка')

        ax[1, 1].plot(coordinates, e_rmae, linewidth=2,
                      color='forestgreen')
        ax[1, 1].set_xlabel('t')
        ax[1, 1].set_ylabel('$rMAE$')
        ax[1, 1].set_title('Относительная средняя абсолютная ошибка')
        plt.savefig('results/Oscillator errors.jpg',
                    dpi=500)

    # Compute the mean and the standard deviation of the metamodel output
    metamodel_eval_mean = chaospy.E(metamodel, c_dist)
    metamodel_eval_std = chaospy.Std(metamodel, c_dist)

    # Plot the mean and the standard deviation in a demonstrative form
    if kwargs['plot']:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.fill_between(coordinates, metamodel_eval_mean
                        - metamodel_eval_std,
                        metamodel_eval_mean +
                        metamodel_eval_std,
                        alpha=0.4,
                        label='Стандартное отклонение')
        ax.plot(coordinates, metamodel_eval_mean,
                label='Математическое ожидание')
        ax.set_title('Математическое ожидание и стандартное отклонение')
        ax.set_xlabel('t')
        ax.set_ylabel('$\\mathcal{M}_{PC}(t, c)$')
        plt.legend()
        plt.savefig('results/Oscillator expectation.jpg',
                    dpi=500)

    # Measure the time for computing the model and metamodel responses
    if kwargs['quantiles']:
        model_time = np.array([])
        metamodel_time = np.array([])
        for p in c_dist.sample(1000,
                               rule='latin_hypercube'):
            start1 = time.time()
            model.compute_displacement(p, coordinates)
            finish1 = time.time()
            model_time = np.append(model_time, finish1 -
                                               start1)
            start2 = time.time()
            metamodel(p)
            finish2 = time.time()
            metamodel_time = np.append(metamodel_time,
                                       finish2 - start2)

        # Estimate the 95% quantiles
        with open('results/Oscillator quantiles.txt',
                  'w') as res_file:
            print(np.quantile(model_time,
                  q=0.025).round(10),
                  np.quantile(model_time,
                  q=0.975).round(10), file=res_file)
            print(np.quantile(metamodel_time,
                  q=0.025).round(10),
                  np.quantile(metamodel_time,
                  q=0.975).round(10), file=res_file)

    """
    Example 1.2: Damped linear oscillator model. PC
    expansion using the Galerkin method.
    """

    # Define the mean and the standard deviation of the damping factor
    c_mu = kwargs['mu']
    c_sigma = kwargs['sigma']

    # Define the initial condition for the Galerkin method
    ic = oscillator.galerkin_ic(model.y_0, model.y_1,
         chaos_basis)

    # Compute the expansion coefficients using the Galerkin method
    galerkin_coeffs = solve_ivp(
                      fun=oscillator.galerkin_system,
                      t_span=(kwargs['t_min'],
                              kwargs['t_max']),
                      y0=ic, method='RK23',
                      t_eval=coordinates,
                      args=(c_dist, c_mu, c_sigma,
                            chaos_basis, model))['y']

    # Define the intrusive metamodel from the result
    intrusive_metamodel = chaospy.sum(chaos_basis *
                          galerkin_coeffs.T[:, :3], -1)

    res = {'model': model, 'chaos_basis': chaos_basis,
           'metamodel': metamodel, 'intrusive_metamodel':
           intrusive_metamodel, 'e_loo': e_loo,
           'r_sqr': r_sqr, 'e_mae': e_mae,
           'e_rmae': e_rmae}
    return res

np.random.seed(118848)
res = example1()
with open("results/galerkin_coefficients.txt", "w") as f:
    print(res, file=f)
    
print("Completed!")