import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates
import glob
from scipy import stats
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

    def avg_per_month(self):
        df = self.field_data
        df2 = df.groupby(["site_no",df.lev_dt.dt.month]).mean().reset_index()
        return df2

    def print_mean_per_month(self):
        df = self.avg_per_month()
        filter_sites = [
            175708066163000,
            175748066160400,
            175837066165400,
            175921066144500,
            175933066161800,
            175950066125200,
            180000066125200,
            180006066123700,
            180012066125500,
            180017066132100,
            180023066175400
        ]
        df2 = df[~df["site_no"].isin(filter_sites)]
        df2 = df2.pivot(index='site_no', columns='lev_dt', values='lev_va').reset_index()

        num_to_month = {
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

        df2.rename(columns=num_to_month, inplace=True)
        print(df2)
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))
        df2.to_csv(os.path.join(
                    CURR_DIR, "mean_gwlevels_per_month.csv"), index=False)


def frequency(df, height=1.5, aspect=2, col_wrap=5, palette=None, style="darkgrid", filtered=False):
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
        175837066165400,
        175921066144500,
        175933066161800,
        175950066125200,
        180000066125200,
        180006066123700,
        180012066125500,
        180017066132100,
        180023066175400
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
def month_year_timeseries(df, height=1.5, aspect=2, col_wrap=5, style="whitegrid", filtered=False, dfprint=False):
    df2 = df.copy()
    df2.index = pd.to_datetime(df["lev_dt"])
    df2 = df2.groupby(["site_no"]).resample("M").lev_va.mean()
    df2 = df2.reset_index()

    # Filtering out sites by site_no...
    filter_sites = [
        175708066163000,
        175748066160400,
        175837066165400,
        175921066144500,
        175933066161800,
        175950066125200,
        180000066125200,
        180006066123700,
        180012066125500,
        180017066132100,
        180023066175400
    ]
    # ~ indicates is not in.  Check pandas doc
    if filtered:
        df2 = df2[~df2["site_no"].isin(filter_sites)]
    if dfprint:
        print("Exporting dataframe to current directory.")
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))

        df2.to_csv(os.path.join(
                    CURR_DIR, "mean_gwlevels_per_month_year.csv"), index=False)
        return
    site_no_list = df2.site_no.unique().tolist()

    print(df2)
    loop = math.ceil(len(site_no_list)/(col_wrap*2))
    sns.set(style=style, font_scale=0.85)
    for i in range(loop):
        sliced_list = site_no_list[:col_wrap*2]
        g = sns.FacetGrid(df2[df2["site_no"].isin(sliced_list)], col="site_no", col_wrap=col_wrap, height=height, aspect=aspect, sharex=True, sharey=False)
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
        del site_no_list[:col_wrap*2]

    plt.show()


def calculate_gw_elevation():
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(
        CURR_DIR, "filtered_salinas_wells_reprojected.csv")

    df = pd.read_csv(DATA_PATH, index_col=None)

    # get mean columns
    mean_col = [s for s in list(df.columns) if "mean" in s]

    def get_elevation(alt, depth):
        return alt-depth

    for col in mean_col:
        col_name = col.replace("mean_", "alt_")
        df[col_name] = get_elevation(df.alt_va, df[col])

    df.to_csv(os.path.join(
                CURR_DIR, "salinas_wells_reprojected.csv"), index=False)
    return df



def get_data():

    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(
        CURR_DIR, "salinas-gwlevels/by_month/oct_2015_gwlevels.tsv")

    df = pd.read_csv(DATA_PATH, sep='\t', index_col=None, header=125)
    df.lev_dt = pd.to_datetime(df["lev_dt"])
    df2 = df.copy()
    df2.index = pd.to_datetime(df["lev_dt"])
    df2 = df2.groupby(["site_no"]).resample("M").lev_va.mean()
    df2 = df2.reset_index()

    df2.to_csv(os.path.join(
            CURR_DIR, "salinas-gwlevels/by_month/oct_2015_gwlevels.csv"
        ), index=False)
    return df, df2


def normal_test(x):
    # http://dataunderthehood.com/2018/01/15/box-cox-transformation-with-python/
    nm_value, nm_p = stats.normaltest(x)
    jb_value, jb_p = stats.jarque_bera(x)
    data_rows = {
        'Test Name': ["D’Agostino-Pearson", "Jarque-Bera"],
        "Statistics": [nm_value, jb_value], "p-value": [nm_p, jb_p]
    }

    t = pd.DataFrame(data_rows)
    print(t)


def normalize(df_att, shift_num=0):
    # shift dataset in case of negative numbers
    # scipy boxcox doesn't take negative values or 0
    print("Values used...")
    print(df_att)
    print(f'Min value of original set: {df_att.min()}')

    if df_att.min() <= 0 and shift_num == 0:
        shift_num = abs(df_att.min()) + 1

    shifted_vals = df_att + shift_num
    print(f'Min value of shifted_vals: {shifted_vals.min()}')
    print(f'Values shifted by: {str(shift_num)}')

    xt, maxlog, interval = stats.boxcox(shifted_vals, alpha=0.05)
    print("lambda = {:g}".format(maxlog))
    # print(xt)
    normal_test(xt)
    # return new values with original data
    return xt