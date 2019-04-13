import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def get_data(filename=None, get_stats=False):

    INFO = {
        "jul_2002": {
            "name": "jul_2002",
            "path": "./salinas-gwlevels/by_month/july_2002_gwlevels.tsv",
            "header": 118,
            "title_date": "July 2002",
            "test_path": "./salinas-gwlevels/by_month/test_results/jul_2002"},
        "oct_2015": {
            "name": "oct_2015",
            "path": "./salinas-gwlevels/by_month/oct_2015_gwlevels.tsv",
            "header": 125,
            "title_date": "October 2015",
            "test_path": "./salinas-gwlevels/by_month/test_results/oct_2015"}
    }

    # Handle invalid month_year input
    if filename is None or filename not in list(INFO.keys()):
        print("Error: Please provide a month_year")
        print("The following data is available...")
        print(INFO.keys())
        return

    # Get constant info
    fileinfo = INFO.get(filename, "Invalid year")

    # Fetch processed data
    df, df2 = groupby_month(fileinfo)

    # Get normal test stats
    test_results = []
    if get_stats is True:
        normal_tests = []
        print(f'Getting normal test dataframes...')
        test_params = ["lev_va", "gw_elev", "bxc_gw_elev"]

        for x in test_params:
            normal_tests.append(normal_test(df2[x], fileinfo["title_date"], fileinfo))

        plt.show()
        test_results = pd.concat(normal_tests, axis=1, sort=False)
        test_results.to_csv(os.path.join(fileinfo["test_path"], f'normal-{fileinfo["name"]}.csv'), index=False)

    results = {
        "df": df,
        "df2": df2,
        "tests": test_results
    }
    # result = pd.concat(normal_test_array, axis=1, sort=False)
    print("Returning original df, df2 and test results.")
    return results


def groupby_month(fileinfo):
    # Create the processed file with all the relevant info/parameters

    # Get all info sites for calculating gw_elev
    ALLSITES_PATH = "./salinas-gwlevels/by_month/USGS_all_site_info.tsv"
    sdf = pd.read_csv(ALLSITES_PATH, sep='\t', index_col=None, header=40)

    # Get month-year file
    df = pd.read_csv(
        fileinfo["path"], sep='\t', index_col=None, header=fileinfo["header"])

    df.lev_dt = pd.to_datetime(df["lev_dt"])
    df2 = df.copy()
    df2.index = pd.to_datetime(df["lev_dt"])
    df2 = df2.groupby(["site_no"]).resample("M").lev_va.mean()
    df2 = df2.reset_index()

    # Add latlong, alt_va and calculate gw_elev
    df2["alt_va"] = df2.site_no.map(sdf.set_index("site_no")["alt_va"])
    df2["dec_lat_va"] = df2.site_no.map(sdf.set_index("site_no")["dec_lat_va"])
    df2["dec_long_va"] = df2.site_no.map(sdf.set_index("site_no")["dec_long_va"])
    df2["gw_elev"] = df2["alt_va"]-df2["lev_va"]

    # Normalize gw_elev and add to df...
    bxc_gw_elev = normalize(df2.gw_elev)
    df2["bxc_gw_elev"] = bxc_gw_elev["xt"]
    df2["shifted_by"] = bxc_gw_elev["shifted_by"]
    df2["alpha"] = bxc_gw_elev["alpha"]
    df2["lambda"] = bxc_gw_elev["lambda"]

    df2.to_csv(os.path.join(
            fileinfo["test_path"], f'{fileinfo["name"]}_mean_gwlevels.csv'
        ), index=False)

    print("Returns original df, grouped df")
    return df, df2


def normal_test(x, title="", month_info=False):
    # https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
    # http://dataunderthehood.com/2018/01/15/box-cox-transformation-with-python/

    nm_value, nm_p = stats.normaltest(x)
    jb_value, jb_p = stats.jarque_bera(x)
    shp_value, shp_p = stats.shapiro(x)
    data_rows = {
        'Test Name': ["D'Agostino-Pearson", "Jarque-Bera", "Shapiro"],
        "Statistics": [nm_value, jb_value, shp_value], "p-value": [nm_p, jb_p, shp_p]
    }

    dfstat = pd.DataFrame(data_rows)
    dfstat["param"] = x.name
    print(dfstat)

    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm
    # ad_test = stats.anderson(x, dist="norm")
    # print("Anderson Test: val and p")
    # print(ad_test)

    # Histogram and QQ plotting

    title_param = {
        "lev_va": {"desc": "groundwater depth"},
        "gw_elev": {"desc": "groundwater elevation"},
        "bxc_gw_elev": {"desc": "normalized groundwater elevation"}
    }

    title_desc = title_param.get(x.name)

    ax = plt.subplot(121)
    stats.probplot(x, plot=plt)

    ax2 = plt.subplot(122)
    ax2.hist(x, bins='auto')
    ax2.set_xlabel(f'{(title_desc["desc"]).capitalize()}')
    ax2.set_ylabel("Frequency")
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.suptitle(f'Dataset of {month_info["title_date"]}', y=0.95, weight="bold")
    if month_info is not False:
        ax.set_title(f'QQ-Plot for {title_desc["desc"]}')
        ax2.set_title(f'Histogram for {title_desc["desc"]}')
        fig = plt.gcf()
        fig.set_size_inches(11,8)
        plt.savefig(os.path.join(
                month_info["test_path"], f'qq-hist-{month_info["name"]}{x.name}.png'
            ))
        # , bbox_inches='tight'
        plt.close()
    else:
        ax.set_title(title)
        ax2.set_title(title)
        plt.show()

    return dfstat


def normalize(df_att, shift_num=0):
    # shift dataset in case of negative numbers
    # scipy boxcox doesn't take negative values or 0
    print(f'Normalizing... {df_att.name}')
    print(f'Min value of original set: {df_att.min()}')

    if df_att.min() <= 0 and shift_num == 0:
        shift_num = abs(df_att.min()) + 1

    shifted_vals = df_att + shift_num
    print(f'Min value of shifted_vals: {shifted_vals.min()}')
    print(f'Values shifted by: {str(shift_num)}')

    xt, maxlog, interval = stats.boxcox(shifted_vals, alpha=0.05)
    print("lambda = {:g}".format(maxlog))

    normalstats = {
        "shifted_by": shift_num,
        "lambda": maxlog,
        "alpha": 0.05,
        "xt": xt
    }
    return normalstats
