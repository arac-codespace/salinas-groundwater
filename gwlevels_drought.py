import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def get_data(month_year=None):

    path = {
        "jul_2002": {"name": "jul_2002", "path": "salinas-gwlevels/by_month/july_2002_gwlevels.tsv", "header": 118},
        "oct_2015": {"name": "oct_2015", "path": "salinas-gwlevels/by_month/oct_2015_gwlevels.tsv", "header": 125}
    }

    if month_year is None:
        print("Error: Please provide a month_year")
        print("The following data is available...")
        print(path.keys())
        return

    month_info = path.get(month_year, "Invalid year")

    return groupby_month(month_info)


    # return normalize(df2)

def groupby_month(month_info):

    print(month_info)
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(
        CURR_DIR, month_info["path"])

    ALLSITES_PATH = os.path.join(
        CURR_DIR, "salinas-gwlevels/by_month/USGS_all_site_info.tsv")

    sdf = pd.read_csv(ALLSITES_PATH, sep='\t', index_col=None, header=40)

    df = pd.read_csv(DATA_PATH, sep='\t', index_col=None, header=month_info["header"])

    df.lev_dt = pd.to_datetime(df["lev_dt"])
    df2 = df.copy()
    df2.index = pd.to_datetime(df["lev_dt"])
    df2 = df2.groupby(["site_no"]).resample("M").lev_va.mean()
    df2 = df2.reset_index()

    # Add alt_va and calculate gw_elev
    df2["alt_va"] = df2.site_no.map(sdf.set_index("site_no")["alt_va"])
    df2["gw_elev"] = df2["alt_va"]-df2["lev_va"]

    # Normalize gw_elev...
    bxc_gw_elev = normalize(df2.gw_elev)

    df2["bxc_gw_elev"] = bxc_gw_elev

    df2.to_csv(os.path.join(
            CURR_DIR, f'salinas-gwlevels/by_month/{month_info["name"]}_mean_gwlevels.csv'
        ), index=False)

    print("Returns original df, grouped df")
    return df, sdf, df2

def normal_test(x, title_date=""):
    # https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
    # http://dataunderthehood.com/2018/01/15/box-cox-transformation-with-python/
    nm_value, nm_p = stats.normaltest(x)
    jb_value, jb_p = stats.jarque_bera(x)
    shp_value, shp_p = stats.shapiro(x)
    data_rows = {
        'Test Name': ["D’Agostino-Pearson", "Jarque-Bera", "Shapiro"],
        "Statistics": [nm_value, jb_value, shp_value], "p-value": [nm_p, jb_p, shp_p]
    }

    t = pd.DataFrame(data_rows)
    print(t)

    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm
    ad_test = stats.anderson(x, dist="norm")
    print("Anderson Test: val and p")
    print(ad_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    title_param = {
        "lev_va": {"name": "groundwater depth"},
        "gw_elev": {"name": "groundwater elevation"},
        "bxc_gw_elev": {"name": "normalized groundwater elevation"}
    }

    title = title_param.get(x.name)

    stats.probplot(x, plot=ax)
    ax.set_title(f'QQ-Plot for {title["name"]}. ({title_date})')

    plt.figure()
    plt.hist(x, bins='auto')
    plt.title(f'Histogram for {title["name"]}. ({title_date})')
    plt.show()

    plt.show()


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

    return xt