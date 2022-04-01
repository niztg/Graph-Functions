# Graph-Functions
graphz

This module allows you to make graphs easily

example usage:

```py
dg_stats2 = {
    "10/19": [9.5, 3.5],
    "11/19": [11.3, 3.0],
    "12/19": [11.8, 2.7],
    "01/20": [14.4, 5.3],
    "02/20": [12.6, 4.8],
    "12/20": [19.0, 7.2],
    "01/21": [12.7, 3.6],
    "02/21": [15.9, 6.5],
    "03/21": [17.5, 5.4],
    "04/21": [20.5, 7.3],
    "10/21": [13.4, 8.2],
    "11/21": [20.5, 6.8],
    "12/21": [20.8, 7.5],
    "01/22": [20.4, 10.2],
    "02/22": [28.0, 5.0],
    "03/22": [25.3, 10.7]
}

prepare_manylines(
    plt_values=dg_stats2,
    x_ticks=True,
    y_ticks=True,
    del_spines=True,
    annotate=True,
    title="Darius Garland's Statistical Improvements by Month",
    color=['#005bff', '#00c8ff'],
    legend=["PPG", "APG"]
)
```

<img src=https://i.imgur.com/NsdDLWh.png height=500></img>


made for my son 💙xox
