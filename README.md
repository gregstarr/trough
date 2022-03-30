[![Python Package using Conda](https://github.com/gregstarr/trough/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/gregstarr/trough/actions/workflows/python-package-conda.yml)

# trough

### Example

![Example](example.png)

### Features
- Download Madrigal TEC, OMNI and DMSP SSUSI data
- Process datasets into more convenient `xarray` data structures and save as NetCDF
- Automatically label main ionospheric trough

# Usage

1. Clone Repo
2. create conda environment using `environment.yml` (if you have trouble with apexpy, install it first)
3. install trough with `pip install -e .`
4. copy `config.json.example` --> `config.json` and change any options you want
5. run with `python -m trough config.json`
6. wait for it to finish (can take several days if you are running 5+ years)
7. add `import trough` in your code and access the data using `trough.get_data`

### Config
| Config Option | Definition                                                     |
| --- |----------------------------------------------------------------|
| base_dir | base directory of trough downloads and processing, directories |
| madrigal_user_name |                                                                |
| madrigal_user_email |                                                                |
| madrigal_user_affil |                                                                |
| nasa_spdf_download_method |                                                                |
| lat_res |                                                                |
| lon_res |                                                                |
| time_res_unit |                                                                |
| time_res_n |                                                                |
| script_name |                                                                |
| start_date |                                                                |
| end_date |                                                                |
| keep_download |                                                                |
| trough_id_params |                                                                |

"trough_id_params": {
"bg_est_shape": [
  1,
  19,
  17
],
"model_weight_max": 15,
"rbf_bw": 1,
"tv_hw": 2,
"tv_vw": 1,
"l2_weight": 0.09,
"tv_weight": 0.15,
"perimeter_th": 30,
"area_th": 30,
"threshold": 1,
"closing_rad": 0
},