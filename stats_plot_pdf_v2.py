
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, norm, gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class Stats_Plot_PDF:
    """"""
    plt.rcParams['font.size'] = 16
    plt.rcParams["figure.autolayout"] = True

    def __init__(self, asset, data, factor, other_factor_list, output_path):
        self.asset = asset
        self.data = data
        self.factor = factor
        self.other_factor_list = other_factor_list
        self.output_path = output_path

    def stats(self):
        """"""
        stats = {}
        factor_values = self.data[self.factor].values
        if abs(np.mean(factor_values)) >= 1:
            stats['mean'] = np.round(np.mean(factor_values), decimals=2)
            stats['std'] = np.round(np.std(factor_values), decimals=2)
            stats['median'] = np.round(np.median(factor_values), decimals=2)
            stats['kurtosis'] = np.round(kurtosis(factor_values), decimals=2)
            stats['skewness'] = np.round(skew(factor_values), decimals=2)
        else:
            stats['mean'] = np.format_float_positional(np.mean(factor_values), precision=4, unique=False,
                                                       fractional=False)
            stats['std'] = np.format_float_positional(np.std(factor_values), precision=4, unique=False,
                                                      fractional=False)
            stats['median'] = np.format_float_positional(np.median(factor_values), precision=4, unique=False,
                                                         fractional=False)
            stats['kurtosis'] = np.format_float_positional(kurtosis(factor_values), precision=4, unique=False,
                                                           fractional=False)
            stats['skewness'] = np.format_float_positional(skew(factor_values), precision=4, unique=False,
                                                           fractional=False)

        df = pd.DataFrame.from_dict(stats, orient='index', columns=[self.factor])
        df.index.name = 'Stats'
        df.reset_index(inplace=True)

        fig0, ax0 = plt.subplots(figsize=(20, 6))
        ax0.axis('off')  # Hide the axis
        df = self.stats()
        table = ax0.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(True)
        table.scale(2, 2)
        return df, fig0

    def trend_plot(self):
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d')
        x = self.data['date']
        y1 = self.data[self.factor]
        y2 = self.data['close']

        fig1 = plt.figure(figsize=(20, 10))
        plt.plot(x, y1, 'y-', label=self.factor)
        plt.ylabel(self.factor)
        # plt.tick_params('y',colors='y')
        plt.xticks(rotation=45)

        plt.twinx()
        plt.plot(x, y2, 'r-', label='close_price')
        plt.ylabel('close_price')
        # plt.tick_params('y',colors='r')
        plt.xticks(rotation=45)
        plt.legend(loc='best')

        plt.title(self.factor + '_trend')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
        plt.grid(True)
        plt.tight_layout()
        return fig1

    def distribution_plot(self):
        factor_values = self.data[self.factor].values
        fig2 = plt.figure(figsize=(20, 6))
        sns.histplot(factor_values, color='blue', alpha=0.5, label='Histogram')

        kde = gaussian_kde(factor_values)
        x = np.linspace(np.min(factor_values), np.max(factor_values), 100)
        mean = np.mean(factor_values)
        std = np.std(factor_values)

        y1 = kde(x)
        y2 = norm.pdf(x, mean, std)

        plt.twinx()
        plt.plot(x, y1, color='blue', label='KDE')
        plt.plot(x, y2, color='black', label='Normal Distribution')

        median = np.median(factor_values)
        mean = np.mean(factor_values)
        plt.axvline(median, color='r', linestyle='-', label='median')
        plt.axvline(mean, color='orange', linestyle='-', label='mean')
        plt.legend()

        # plt.xlabel(factor)
        # plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.grid(True)
        return fig2

    def box_plot(self):
        factor_values = self.data[self.factor].values
        df1 = self.data[[self.factor] + self.other_factor_list].describe().T.round(4).head(1)
        fig3 = plt.figure(figsize=(20, 5))
        gs = GridSpec(2, 1, figure=fig3, height_ratios=[3, 1])
        ax1 = fig3.add_subplot(gs[1, 0])
        ax1.axis('off')  # Hide the axis
        table = ax1.table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(True)
        table.scale(1, 2)

        # factor_data[factor].plot.box(title=factor +"_Boxplot", whis=2.0)
        ax2 = fig3.add_subplot(gs[0, 0])
        sns.boxplot(x=factor_values, ax=ax2)
        plt.subplots_adjust(wspace=0.3)
        # plt.ylabel('factor_values')
        plt.grid(True)
        return fig3

    def corr_matrix_plot(self):
        fig5 = pd.plotting.scatter_matrix(self.data[[self.factor] + self.other_factor_list], figsize=(20, 20))
        return fig5

    def heat_map(self):
        correlations = self.data[[self.factor, 'return'] + self.other_factor_list].corr()
        fig6 = plt.figure(figsize=(20, 10))
        sns.heatmap(correlations, mask=correlations == 1, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation')
        plt.xticks(rotation=45)
        return fig6

    def plot_pdf(self):
        with (PdfPages(self.output_path + f'{self.asset}_{self.factor}_stats_plot.pdf') as pdf):
            pdf.savefig(self.stats()[1])
            pdf.savefig(self.trend_plot())
            pdf.savefig(self.distribution_plot())
            pdf.savefig(self.box_plot())
            pdf.savefig(self.corr_matrix_plot()[0][0].figure)
            pdf.savefig(self.heat_map())
