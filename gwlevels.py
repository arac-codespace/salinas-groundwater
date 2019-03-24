import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

print("Running gwlevels script...")


class GWLevels():
    def __init__(self):
        self.field_data = self.get_gw_data()

    def get_gw_data(self):
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))
        DATA_PATH = os.path.join(CURR_DIR, "salinas-gwlevels/usgs_field_measurements/")

        df = pd.read_csv(
                os.path.join(DATA_PATH, "gwlevels-3-21.tsv"), sep='\t', header=110
            )
        df.lev_dt = pd.to_datetime(df["lev_dt"])
        df['month'] = df["lev_dt"].dt.strftime('%b')

        return(df)

    # filter by approval status code (A for appoved, B for provisional)
    def filter(self):
        df = self.field_data
        return df.query('lev_age_cd=="A"')


def frequency(df, height=5, aspect=1, col_wrap=None, palette=None, style=None):
    # sns.set(style="whitegrid", font_scale=1.0)
    # Extract pertinent columns, groupby month and reset index...
    df2 = df[["site_no", "lev_va", "lev_dt"]]
    site_no_list = df.site_no.unique().tolist()
    # Group so each month-year will have the first record only representing
    # that a measurement was observed in that period
    df2 = df2.groupby(["site_no", df2.lev_dt.dt.month, df2.lev_dt.dt.year]).first()
    # Grouping by site and month and counting the number of year records under a month
    df2 = df2.groupby(["site_no", df2.lev_dt.dt.month]).lev_va.count()
    # col_wrap = math.ceil((df2.site_no.unique().size)/2)
    print(df2)
    # get site_no, lev_dt, lev_va as columns again
    df2 = df2.reset_index()
    print(df2)

    def num_to_month(num):
        switcher = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec"
        }

        return switcher.get(num, "Invalid month")

    df2["lev_dt"] = df2.lev_dt.apply(num_to_month)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Recommended... (df, height=1.5, col_wrap=7, style="darkgrid")
    sns.set(style=style, font_scale=0.8)

    g = sns.catplot(
        x="lev_dt", y="lev_va", col="site_no",
        data=df2, kind="bar", col_wrap=col_wrap,
        height=height, aspect=aspect, order=months,
        saturation=.5, ci=None, palette=palette
    )
    # https://stackoverflow.com/questions/43669229/increase-space-between-rows-on-facetgrid-plot
    g.set_axis_labels("Month", "Count").set_titles("site_no: {col_name}").set_xticklabels(rotation=75)
    g.fig.suptitle("Water-level: months measured throughout the years", size=16, weight="bold")

    index = 0
    for ax in g.axes.flatten():
        for _, spine in ax.spines.items():
            spine.set_visible(False)
            # spine.set_color('black')
            spine.set_linewidth(0)

        min_year = df[df["site_no"] == site_no_list[index]].lev_dt.dt.year.min()
        max_year = df[df["site_no"] == site_no_list[index]].lev_dt.dt.year.max()
        year_range = f'{min_year} - {max_year}'
        # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html
        ax.annotate(year_range, xy=(0, 0.94), xytext=(0, 0.94),
                    xycoords='axes fraction', style="italic")
        index += 1

    # .set_xticklabels("")
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()
