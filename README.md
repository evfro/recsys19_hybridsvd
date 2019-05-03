# Reproducing HybridSVD paper
This repository contains full source code for reproducing results from the HybridSVD paper. If you want to run it on your own machine, make sure to prepare [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) according to [this](binder/environment.yml) configuation file, which contains the list of all required packages (including their versions).  

You can also **interactively run experiments directly in your browser** with the help of [Binder](https://mybinder.readthedocs.io) cloud technologies. Simply click on the badge below to get started:

[![badge](https://img.shields.io/badge/reproduce%20results-in%20your%20browser-F5A252.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC)](https://mybinder.org/v2/gh/recsys19/hybridsvd/master?urlpath=lab/tree/experiments/HybridSVD.ipynb)

This will launch interactive JupyterLab environment with access to all repository files. By default it starts with the `HybridSVD.ipynb` notebook that contains the code for HybridSVD model evaluated on the Movielens and Bookcrossing datasets.  

## Mind cloud environment restrictions
Due to restrictions on Binder's cloud resources only small datasets, e.g., `Movielens-1M` or `Amazon Video Games`, allow performing full experiments without interruption. Attempts to work with larger files will likely crash the environment. Originally all experiments were conducted on HPC servers with much larger amount of hardware resources. It is, therefore, advised to make the following modifications to run jupyter notebooks safely in the Binder cloud:

### Working with Movielens-1M data
Experiments with this dataset are available in the following files:
* Baselines.ipynb
* HybridSVD.ipynb
* FactorizationMachines.ipynb
* LCE.ipynb
* ScaledSVD.ipynb
* ScaledHybridSVD.ipynb

You need to change the `data_labels` variable in the `Experiment setup` section of each notebook from
```python
data_labels = ['ML1M', 'ML10M', 'BX']
```
to
```python
data_labels = ['ML1M']
```
Accordingly, do not run cells under `Movielens10M` and `BookCrossing` headers (these datasets are not provided in the cloud environment). Also make sure that the first argument to the `get_movielens_data` is *../datasets/movielens/ml-1m.zip* (originally the notebooks were executed on several machines that's why the path may vary), e.g., it should start as:
```python
data_dict[lbl], meta_dict[lbl] = get_movielens_data('../datasets/movielens/ml-1m.zip',
                                                     <other arguments>
```

### Working with Amazon Video Games data
Experiments with this dataset are available in the following files:
* Baselines_AMZ.ipynb
* HybridSVD_AMZ.ipynb
* FactorizationMachines_AMZ.ipynb
* LCE_AMZ.ipynb
* ScaledSVD_AMZ.ipynb
* ScaledHybridSVD_AMZ.ipynb

You need to change the `data_labels` variable in the `Experiment setup` section from
```python
data_labels = ['AMZe', 'AMZvg']
```
to
```python
data_labels = ['AMZvg']
```
Accordingly, do not run cells under `AMZe` header. Again, make sure to provide correct input arguments to the `get_amazon_data`. In this case they are:
```python
data_dict[lbl], meta_dict[lbl] = get_amazon_data('../datasets/amazon/ratings_Video_Games.csv',
                                                 meta_path='../datasets/amazon/meta/meta_Video_Games.json.gz',
                                                 <other arguments>
```

### Reducing training time
Keep in mind that some models require much longer training time than others. For example, the whole experiment for `HybridSVD` in both standard and cold start scenarios on the Movielens-1M dataset completes even before the initial tuning of `Factorization Machines` is done for standard scenario. As Binder automatically shuts down long running tasks you may not be able to perform all computations before the timeout. To reduce the risk of such shutdown you may want to run different notebooks (different models) in independent Binder sessions. You may also want to reduce the number of points to consider in the random grid search for tuning non SVD-based models. For example, in the FM case you can change the `ntrial=60` input to `ntrials=30` in the `fine_tune_fm(model, params, label, ntrials=60)` function calls. This may, however, slightly decrease the resulting quality of FM.

Alternatively, you can skip parameter tuning sections for long-running models and reuse previously found set of nearly optimal hyper-parameters. They are printed in the end of each section with model tuning. You can also find them in the [View optimal parameters](View_optimal_parameters.ipynb) notebook.
