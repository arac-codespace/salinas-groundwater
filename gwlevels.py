import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates
import glob

print("Running gwlevels script...")


class GWLevels():
    def __init__(self):
        self.field_data = self.get_gw_data()

    # https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    def get_gw_data(self):
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))
        DATA_PATH = os.path.join(
            CURR_DIR, "salinas-gwlevels/usgs_field_measurements/")
        all_tsv_files = glob.glob(DATA_PATH + "/*.tsv")
        li = []
        for filename in all_tsv_files:
            print(filename)
            df = pd.read_csv(filename, index_col=None, sep='\t', header=110)
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)

        df.lev_dt = pd.to_datetime(df["lev_dt"])
        df['month'] = df["lev_dt"].dt.strftime('%b')

        return(df)

    # filter by approval status code (A for appoved, B for provisional)
    def filter(self):
        df = self.field_data
        return df.query('lev_age_cd=="A"')


def frequency(df, height=1.5, aspect=2, col_wrap=5, palette=None, style="darkgrid", filtered=True):
    # gw.frequency(df, height=1.5, aspect=2, col_wrap=5, style="darkgrid")
    # Extract pertinent columns, groupby month and reset index...
    df2 = df[["site_no", "lev_va", "lev_dt"]]
    # Group so each month-year will have the first record only representing
    # that a measurement was observed in that period
    df2 = df2.groupby(
        ["site_no", df2.lev_dt.dt.month, df2.lev_dt.dt.year]).first()
    # Grouping by site and month and counting the number of year records under
    # a month
    df2 = df2.groupby(["site_no", df2.lev_dt.dt.month]).lev_va.count()
    # get site_no, lev_dt, lev_va as columns again
    df2 = df2.reset_index()

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

    # Filtering out sites by site_no...
    filter_sites = [
        175708066163000,
        175748066160400,
        175921066144500,
        175933066161800
    ]
    # ~ indicates is not in.  Check pandas doc
    if filtered:
        df2 = df2[~df2["site_no"].isin(filter_sites)]

    # This is here to take filtering into account
    site_no_list = df2.site_no.unique().tolist()

    print(df2)
    loop = math.ceil(len(site_no_list)/(col_wrap*2))
    sns.set(style=style, font_scale=0.85)
    for i in range(loop):
            sliced_list = site_no_list[:col_wrap*2]
            g = sns.catplot(
                x="lev_dt", y="lev_va", col="site_no",
                data=df2[df2["site_no"].isin(sliced_list)], kind="bar", col_wrap=col_wrap,
                height=height, aspect=aspect, order=months,
                saturation=.5, ci=None, palette=palette
            )
            # https://stackoverflow.com/questions/43669229/increase-space-between-rows-on-facetgrid-plot
            g.set_axis_labels("Month", "Count").set_titles(
                "site_no: {col_name}").set_xticklabels(rotation=75)

            g.fig.suptitle(
                "Water-level: months measured throughout the years",
                size=16
            )

            index = 0
            for ax in g.axes.flatten():
                for _, spine in ax.spines.items():
                    spine.set_visible(False)
                    # spine.set_color('black')
                    spine.set_linewidth(0)

                min_year = df[df["site_no"] == sliced_list[index]].lev_dt.dt.year.min()

                max_year = df[df["site_no"] == sliced_list[index]].lev_dt.dt.year.max()

                year_range = f'{min_year} - {max_year}'
                # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html
                ax.annotate(year_range, xy=(0, 0.94), xytext=(0, 0.94),
                            xycoords='axes fraction', style="italic")
                index += 1

            plt.subplots_adjust(hspace=0.2, wspace=0.1)
            del site_no_list[:col_wrap*2]
    plt.show()


def year_avg_timeseries(df, height=1.5, aspect=1, col_wrap=None):
    df2 = df.groupby(["site_no", df.lev_dt.dt.year]).lev_va.mean()
    df2 = df2.reset_index()
    g = sns.FacetGrid(df2, col="site_no", col_wrap=col_wrap, height=height, aspect=aspect, sharex=False, sharey=False)
    g = g.map(plt.plot, "lev_dt", "lev_va", marker=".")
    g.set_xticklabels(rotation=75)

    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(plt.LinearLocator(5))
    plt.show()


# gw.month_year_timeseries(df, height=1.5, aspect=2, col_wrap=5, style="whitegrid")
def month_year_timeseries(df, height=1.5, aspect=1, col_wrap=None, style=None, filtered=True, dfprint=False):
    df2 = df.copy()
    df2.index = pd.to_datetime(df["lev_dt"])
    df2 = df2.groupby(["site_no"]).resample("M").lev_va.mean()
    df2 = df2.reset_index()

    # Filtering out sites by site_no...
    filter_sites = [
        175708066163000,
        175748066160400,
        175921066144500,
        175933066161800
    ]
    # ~ indicates is not in.  Check pandas doc
    if filtered:
        df2 = df2[~df2["site_no"].isin(filter_sites)]
    if dfprint:
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))

        df2.to_csv(os.path.join(
                    CURR_DIR, "gwlevels_by_month.csv"), index=False)

    sns.set(style=style, font_scale=0.85)
    g = sns.FacetGrid(df2, col="site_no", col_wrap=col_wrap, height=height, aspect=aspect, sharex=True, sharey=False)
    # seaborn doesn't handle dates in axis. Must use matplotlib's base capabilities
    g = g.map(plt.plot_date, "lev_dt", "lev_va", marker=None, fmt='r-')

    g.fig.suptitle(
        "Water-level: Monthly mean throughout the years", size=16
    )

    g.set_axis_labels("Year", "Depth to water-level (ft.)").set_titles(
        "site_no: {col_name}").set_xticklabels(rotation=75)

    for ax in g.axes.flatten():
        ax.invert_yaxis()
        # formater and locator are necessary to get the axes to behave when
        # interacted with
        # https://stackoverflow.com/questions/31255815/seaborn-tsplot-does-not-show-datetimes-on-x-axis-well
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
