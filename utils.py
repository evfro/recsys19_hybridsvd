import numpy as np
import pandas as pd

try:
    from telepyth import TelepythClient
except ImportError:
    TelepythClient = None

from polara.evaluation.pipelines import set_config

def print_data_stats(data_labels, all_data):
    raw_data, similarities, sim_indices, item_meta = all_data
    for lbl in data_labels:
        if raw_data[lbl] is not None:
            itemid = item_meta[lbl].index.name
            uniques = raw_data[lbl].apply('nunique')[:2].to_dict()
            print(lbl)
            print(uniques)
            print('density',
                  100*raw_data[lbl].shape[0] / 
                  np.prod(list(uniques.values()), dtype='i8'), end='\n')
            print('similarity matrix density',
                  100*similarities[lbl][itemid].nnz / 
                  np.prod(similarities[lbl][itemid].shape, dtype='i8'))


def apply_config(models, config, label):
    for model in models:
        model.verbose = False
        if not isinstance(config, (list, tuple)):
            config = [config]
        for cfg in config:
            model_config = cfg[label].get(model.method, None)
            if model_config:
                model_config = {k: v if v==v else None # handle NaNs if present
                                for k, v in model_config.items()}
                set_config(model, *list(zip(*model_config.items())))


def plot_rank_results(results, **kwargs):
    return [pd.DataFrame(scores)
              .plot(logx=True, title=label, ylim=(0, None), **kwargs)
            for label, scores in results.items()]


def plot_topn_results(results, metric, **kwargs):
    return [scores.xs(metric, level='metric', axis='columns')
                  .squeeze().unstack('model') # remove extra levels and reshape
                  .mean(level='top-n')
                  .plot.bar(rot=0, title=label, **kwargs)
            for label, scores in results.items()]


def report_results(plot_func, *args, **kwargs):
    if isinstance(plot_func, str):
        lbl = plot_func
        if (plot_func == 'tuning') or (plot_func == 'rank'):
            plot_func = plot_rank_results
            msg = ''
        if (plot_func == 'cv') or (plot_func == 'topn'):
            plot_func = plot_topn_results
            msg = args[1] 
    else:
        lbl = plot_func.__name__
    
    axs = plot_func(*args, **kwargs)
    if TelepythClient:
        tp = TelepythClient()
        for ax in axs:
            try: # send figure to chat with @telepyth_bot in Telegram
                tp.send_figure(ax.get_figure(), f'{msg} {lbl}')
            except: # don't stop running python if fails
                pass
    return axs


def save_results(name, tuning=None, config=None, cv=None):
    if config:
        [np.save(f'results/{lbl}/{name}_best_config_{lbl}.npy', cfg) for lbl, cfg in config.items()];
    if tuning:
        [pd.DataFrame(scr).to_csv(f'results/{lbl}/{name}_tuning_scores_{lbl}.csv') for lbl, scr in tuning.items()];
    if cv:
        [res.to_csv(f'results/{lbl}/{name}_cv_scores_{lbl}.csv') for lbl, res in cv.items()];

def save_training_time(name, model, index, label):
    pd.DataFrame(index=index, data={model.method: model.training_time}).to_csv(f'results/{label}/{name}_training_time_{label}.csv')

def save_cv_training_time(name, models, label):
    for model in models:
        index = pd.Index(range(1, len(model.training_time)+1), name='fold')
        save_training_time(f'{name}_cv', model, index, label)

def read_config(experiment_name, data_label):
    return np.load(f'results/{data_label}/{experiment_name}_best_config_{data_label}.npy').item()


def read_cv_data(experiment_name, data_label):
    return pd.read_csv(f'results/{data_label}/{experiment_name}_cv_scores_{data_label}.csv', index_col=[0, 1, 2], header=[0, 1])