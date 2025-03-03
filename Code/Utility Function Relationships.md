# Function Relationships

## Basic Functions

These functions perform fundamental operations and do not use other functions.

1. **dist_calc**
   - **Description**: Calculates distribution metrics.
   - **Used By**: `dist_combiner`, `plot_bin`

2. **plot_lift**
   - **Description**: Plots lift chart.
   - **Used By**: `plot_all`

3. **plot_ks**
   - **Description**: Plots KS curve.
   - **Used By**: `plot_all`
4. **get_psi**
   - **Description**: Calculates Population Stability Index (PSI).
   - **Used By**: `get_psi_vif`

5. **get_vif**
   - **Description**: Calculates Variance Inflation Factor (VIF).
   - **Used By**: `get_psi_vif`
6. **bin_set**
   - **Description**: Sets bins and transforms data.
   - **Uses**: `toad.transform.Combiner`, `toad.transform.WOETransformer`
   - **Used By**: `monolize`
7. **cross_calc**
   - **Description**: Calculates bad rate trends for groups.
   - **Used By**: `plot_badrate`, `cross_vars`

## Mid-Level Functions

These functions use basic functions and are also used by other functions.

1. **dist_combiner**
   - **Description**: Combines distributions using specified bins.
   - **Uses**: `dist_calc`
   - **Used By**: `bins2mono`, `monolize`
2. **bins2mono**
   - **Description**: Ensures bins are monotonic.
   - **Uses**: `dist_combiner`
   - **Used By**: `monolize`
4. **monolize**
   - **Description**: Performs monotonic binning.
   - **Uses**: `dist_combiner`, `bins2mono`, `bin_set`
   - **Used By**: `discrete_type`
6. **plot_bin**
   - **Description**: Plots bin distribution and bad rate.
   - **Uses**: `dist_calc`
   - **Used By**: `bin_update_show_plt`
8. **val_describe**
   - **Description**: Describes variable statistics.
   - **Used By**: `val_describe_tot`
9. **val_describe_tot**
   - **Description**: Describes and outputs variable statistics.
   - **Uses**: `val_describe`
11. **model_verify**
    - **Description**: Verifies model performance.
    - **Uses**: `toad.metrics.KS`, `toad.metrics.AUC`, `toad.metrics.PSI`
    - **Used By**: `rst_print`

## Higher-Level Functions

These functions rely on other functions but are not used by any other function.

1. **get_psi_vif**
   - **Description**: Calculates PSI and VIF.
   - **Uses**: `get_psi`, `get_vif`
2. **discrete_type**
   - **Description**: Discretizes and transforms data.
   - **Uses**: `monolize`
3. **bin_update_show_plt**
   - **Description**: Updates and shows bin plots.
   - **Uses**: `plot_bin`
4. **sample_select**
   - **Description**: Selects training, testing, and validation samples.
5. **rst_print**
   - **Description**: Prints model verification results.
   - **Uses**: `model_verify`
6. **plot_all**
   - **Description**: Plots ROC, KS, and lift charts.
   - **Uses**: `plot_lift`, `plot_ks`
7. **plot_badrate**
   - **Description**: Plots bad rate trends for groups.
   - **Uses**: `cross_calc`
8. **cross_vars**
   - **Description**: Checks bad rate trends for multiple variables.
   - **Uses**: `cross_calc`, `plot_badrate`