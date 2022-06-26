# -*- coding: utf-8 -*-"
"""
Created on 09/24/2021  4:13 PM


@author: Zhuo
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from scipy import stats

from core.autoencoder.models import Linear_AE, CNN_AE, VanillaVAE
from core.dataset.dataset import AEDataset
from core.visualize import plot_loss_curve, plot_comparison_AE_results
from core.utils import save_checkpoint, EarlyStopping
from core.statistic_analysis import center_filter, label_crt_response, crt_criteria, set_columns, data_structure, \
    preprocessing_AEfeatures, univariate_analysis, pearson_corr, multivariate_analysis, get_parameters


class AEtrainer(object):
    def __init__(
            self,
            args,
    ):
        self.args = args
        self.device = torch.device("cuda:{}".format(args.device))
        self._get_model()
        if self.args.resume:
            self.model.load_state_dict(torch.load(self.args.resume)['model_state_dict'])
            self.init_epoch = torch.load(self.args.resume)['epoch']
        else:
            self.init_epoch = 0
        self.AE_min_p = np.nan
        self.AE_max_auc = np.nan

    def _get_model(self):
        if self.args.model == 'Linear_AE':
            self.model = Linear_AE(out_features=self.args.out_features).to(self.device)
        # elif self.args.model == 'CNN_AE':
        #     self.model = CNN_AE(
        #         in_channels=1,
        #         latent_dim=256,
        #     ).to(self.device)
        elif self.args.model == 'VanillaVAE':
            self.model = VanillaVAE(
                in_channels=1,
                latent_dim=256,
            ).to(self.device)
        else:
            raise ValueError("Unknown model: %s" % self.args.model)

    def get_loss(self):
        if self.args.loss == 'mse':
            loss_fn = nn.MSELoss()
        elif self.args.loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        elif self.args.loss == 'bce':
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown loss: %s" % self.args.loss)
        return loss_fn

    def get_tsfm(self):
        if self.args.transform == 1:
            train_tsfm = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.7),
                transforms.RandomVerticalFlip(p=0.7),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ToTensor(),
            ])
        else:
            train_tsfm = transforms.Compose([
                transforms.ToTensor()
            ])
        val_tsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        return train_tsfm, val_tsfm

    def training(self, train_paths=None, val_paths=None):
        train_tsfm, val_tsfm = self.get_tsfm()

        train_ds = AEDataset(train_paths, train_tsfm)
        val_ds = AEDataset(val_paths, val_tsfm)

        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

        loss_fn = self.get_loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001)
        if self.args.lr_update == 'milestones':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000], gamma=0.1)
        elif self.args.lr_update == 'auto':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                   patience=self.args.patience)
        else:
            raise ValueError("Unsupported lr_update methods: %.", self.args.lr_update)

        # Training
        val_interval = 1
        best_mse = 20
        best_mse_epoch = -1
        # Save results
        train_loss_values = list()
        val_loss_values = list()
        print("Start training...")

        early_stopping = EarlyStopping(patience=5 * self.args.patience, verbose=True)
        for epoch in range(self.init_epoch, self.args.num_epochs):
            self.model.train()
            epoch_loss = 0
            step = 0
            val_results = []

            # Training
            for batch_data in train_loader:
                step += 1
                img = batch_data['img']
                if self.args.model == 'Linear_AE':
                    img = Variable(img.float().view(img.size(0), -1))
                img = img.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(img)

                train_loss = loss_fn(outputs, img)
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item()

            epoch_loss /= step
            train_loss_values.append(epoch_loss)

            # Validation
            val_loss = 0
            step = 0
            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        step += 1
                        val_img = val_data['img'].float()
                        if self.args.model == 'Linear_AE':
                            val_img = val_img.view(val_img.size(0), -1)
                            val_img = Variable(val_img)
                        val_img = val_img.to(self.device)

                        val_outputs = self.model(val_img)
                        if self.args.loss == 'mse':
                            val_loss += loss_fn(val_outputs, val_img).item()
                        else:
                            val_loss += loss_fn(val_outputs, val_img)

                        result = [
                            val_img.cpu().numpy().reshape(-1),
                            "",
                            val_outputs.cpu().numpy().reshape(-1),
                            None
                        ]
                        val_results.append(result)

                    val_loss /= step
                    val_loss_values.append(val_loss)

                    if val_loss < best_mse:
                        is_best = True
                        best_mse, best_epoch = save_checkpoint(epoch, self.args.ckpt_interval, self.model, optimizer,
                                                               val_loss, is_best, self.args.results_path)
                    else:
                        is_best = False
                        # save checkpoint for each #ckpt_interval epoch
                        # save_checkpoint(epoch, self.args.ckpt_interval, self.model, optimizer, val_loss, is_best,
                        #                 self.args.results_path)

                    print("epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, best_loss: {:.4f}, best_epoch: {}"
                          .format(epoch, epoch_loss, val_loss, best_mse, best_epoch))

                if self.args.lr_update == 'auto':
                    scheduler.step(val_loss)
                elif self.args.lr_update == 'milestones':
                    scheduler.step()
                else:
                    raise ValueError("learning rate update not support.")

                # plot loss curve
                plot_loss_curve(train_loss_values, val_loss_values, val_interval, epoch,
                                self.args.results_path)

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Save training figures.
            # if (epoch + 1) % (val_interval * 50) == 0:
            #     plot_comparison_AE_results(epoch, val_results, os.path.join(self.args.results_path, 'imgs'))
        print("Training done.")

    def infer(self, infer_paths, multi=True):
        _, infer_tsfm = self.get_tsfm()

        infer_ds = AEDataset(infer_paths, infer_tsfm)

        infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

        loss_fn = self.get_loss()

        # Infer
        print("Inference...")
        results = []
        infer_loss = 0
        step = 0
        self.model.eval()
        with torch.no_grad():
            for data in infer_loader:
                step += 1
                img = data['img'].float()
                if self.args.entropy == 1:
                    params_values, params_names = get_parameters(img[0, 0, :, :].cpu().numpy(), radiomic=True)

                if self.args.model == 'Linear_AE':
                    img = img.view(img.size(0), -1)
                    img = Variable(img)
                img = img.to(self.device)
                name = data['name'][0]

                output = self.model(img)
                AEfeatures = self.model.hook(img)
                infer_loss += loss_fn(output, img).item()

                if self.args.entropy == 1:
                    result = [
                        img.cpu().numpy().reshape(-1),
                        name,
                        output.cpu().numpy().reshape(-1),
                        AEfeatures.cpu().numpy().reshape(-1),
                        params_values,
                        params_names,
                    ]
                else:
                    result = [
                        img.cpu().numpy().reshape(-1),
                        name,
                        output.cpu().numpy().reshape(-1),
                        AEfeatures.cpu().numpy().reshape(-1),
                    ]
                results.append(result)

            # Save AE features to a excel files for the combination analysis.

            # plot_comparison_AE_results(self.args.model, results, os.path.join(self.args.results_path, 'imgs'))
            self.evaluate_result(results, multi)

    def evaluate_result(self, results, multi):
        # Organize AE features
        df = preprocessing_AEfeatures(self.args.out_features, self.args.feature_name, results,
                                      self.args.results_path, self.args.clinic_path, self.args.center,
                                      self.args.entropy)

        clinic_cols, cat_cols, ignore_col_list = set_columns(self.args.center, params=True)

        # AE features list
        df_cols = df.columns.values.tolist()
        AE_cols = [col for col in df_cols if "{}_".format(self.args.feature_name) in col]
        params_cols = [col for col in df_cols if "original" in col]

        # set the label
        df, label_str = label_crt_response(df, self.args.response_definer, self.args.response_source)

        # data CRT criteria Preprocessing
        if self.args.center == 'VISION':
            df = crt_criteria(df, LBBB=True, death=True)
        elif self.args.center == 'two':
            df = crt_criteria(df, LBBB=True, death=True)
        else:
            df = crt_criteria(df, LBBB=False, death=False)

        # data structure
        data_structure(df[clinic_cols + [label_str]], [label_str], cat_cols, sort=True, save_dir=self.args.results_path,
                       tab_title='baseline', ignore_col_list=ignore_col_list, group1_name='Response', rename_dic=None,
                       group0_name='Non-response')

        # Statistic analysis
        # Univariate statistic_analysis in all AE features.
        _, sig05_cols, sig10_AE_cols, self.AE_min_p, self.AE_max_auc = univariate_analysis(df[AE_cols + [label_str]],
                                                                                           label_str,
                                                                                           sort_col='P_value',
                                                                                           tab_title='AE_{}'.format(
                                                                                               self.args.feature_name),
                                                                                           save_dir=self.args.results_path,
                                                                                           ignore_col_list=ignore_col_list)

        # Baseline t-test for the AE features
        data_structure(df[sig10_AE_cols + [label_str]], [label_str], cat_cols=[], sort=True,
                       save_dir=self.args.results_path,
                       tab_title='AE_baseline', ignore_col_list=None, group1_name='Response', rename_dic=None,
                       group0_name='Non-response')

        # univariate_analysis of all significant p >= 0.1 features.
        if len(sig05_cols) > 0:
            uni_cols = sig05_cols + clinic_cols + [label_str]
        else:
            uni_cols = sig10_AE_cols + clinic_cols + [label_str]
        uni_cols = [col for col in uni_cols if "Echo" not in col]
        uni_cols = [col for col in uni_cols if "post" not in col]
        uni_cols = [col for col in uni_cols if "date" not in col]
        _, sig05_cols, _, _, _ = univariate_analysis(df[uni_cols], label_str, sort_col='P_value',
                                                     tab_title='AE_and_clinic_{}'.format(self.args.feature_name),
                                                     save_dir=self.args.results_path, ignore_col_list=ignore_col_list)
        print('Significant variables: ', sig05_cols)

        # pearson_corr to see the correlation with LVMD
        sig_AE_cols = [col for col in sig05_cols if "{}_".format(self.args.feature_name) in col]
        sig_clinic_cols = [col for col in sig05_cols if "{}_".format(self.args.feature_name) not in col]
        # sig_clinic_cols = [col for col in sig_clinic_cols if "Echo".format(self.args.feature_name) not in col]

        # Pearson correlation between AE and other parameters.
        pearson_corr(df[sig_AE_cols + params_cols],
                     filepath=os.path.join(self.args.results_path, 'pearsonCorr_AEandOthers.png'),
                     xrotation=90, xticks_fontsize=4.5, xticks_ha='center',
                     annot=False, annot_kws={'size': 10}, part_cor=True)
        sig_otherParams = [
            # 'original_shannon_entropy',
            'original_firstorder_10Percentile',
            'original_firstorder_90Percentile',
            'original_firstorder_Energy',
            # 'original_firstorder_Entropy',
            'original_firstorder_InterquartileRange',
            'original_firstorder_Kurtosis',
            'original_firstorder_Maximum',
            'original_firstorder_MeanAbsoluteDeviation',
            'original_firstorder_Mean',
            'original_firstorder_Median',
            'original_firstorder_Range',
            'original_firstorder_RobustMeanAbsoluteDeviation',
            'original_firstorder_RootMeanSquared',
            'original_firstorder_Skewness',
            'original_firstorder_TotalEnergy',
            'original_firstorder_Variance']

        # Paired t test
        paired_t_results = []
        # for ae in sig_AE_cols:
        ae = 'AE_12'
        for param in params_cols:
            paired_t_result = {
                "AE": ae,
                "param": param,
                "AE_shapiro_p": stats.shapiro(df[ae]).pvalue,
                "param_shapiro_p": stats.shapiro(df[param]).pvalue,
                "paired_t_p": stats.ttest_rel(df[ae], df[param]).pvalue,
                "wilcoxon_p": stats.wilcoxon(df[ae], df[param]).pvalue,
            }
            paired_t_results.append(paired_t_result)
        df_paired_t = pd.DataFrame(paired_t_results)
        df_paired_t.to_csv(os.path.join(self.args.results_path, "paired_t_results.csv"), index=False)

        # univariate_analysis(df[sig_AE_cols + sig_otherParams], "AE_12", sort_col='P_value',
        #                     tab_title='otherParams_to_predict_AE', save_dir=self.args.results_path,
        #                     method='gls', fit_method='pinv')

        univariate_analysis(df[sig_AE_cols + sig_otherParams + [label_str]], label_str, sort_col='P_value',
                            tab_title='AE_and_otherParams', save_dir=self.args.results_path)

        feature_name_list = [
            'Clinic (QRSd + LVESV)',
            'Clinic + PBW',
            'Clinic + LVMD AE',
            'Clinic + PBW + LVMD AE',
        ]
        # sig_clinic_cols = ['ECG_pre_QRSd', 'Echo_pre_ESV']
        # sig_clinic_cols = ['NYHA', 'Echo_pre_ESV']
        sig_clinic_cols = ['ECG_pre_QRSd', 'SPECT_pre_ESV']  # todo: check this

        if len(sig_AE_cols) > 0:

            pearson_corr(df[sig_AE_cols + clinic_cols], xrotation=60,
                         filepath=os.path.join(self.args.results_path, 'pearsonCorr_AE.png'))

            if multi:
                # Multivariate analysis
                feature_list = [
                    sig_clinic_cols,
                    sig_clinic_cols + ['SPECT_pre_PBW'],
                    sig_clinic_cols + [sig_AE_cols[0]],
                    sig_clinic_cols + [sig_AE_cols[0], 'SPECT_pre_PBW'],
                ]
                multivariate_analysis(df[sig05_cols + clinic_cols + [label_str]], feature_list, feature_name_list,
                                      label_str,
                                      fig_title='AE', save_dir=self.args.results_path, ignore_col_list=ignore_col_list,
                                      y_axis='aic', line_index_list=[[0, 1], [0, 2], [1, 3]],
                                      line_h_list=[170.1, 169.3, 168.5], rotation=0,
                                      )
                # feature_list = [
                #     [sig_AE_cols[0]],
                #     ['ECG_pre_QRSd', sig_AE_cols[0]],
                # ]
                # feature_name_list = [
                #     'AE',
                #     'QRSd + AE'
                # ]
                # multivariate_analysis(df[sig05_cols + clinic_cols + [label_str]], feature_list, feature_name_list,
                #                       label_str,
                #                       fig_title='reviewer_add_QRSd', save_dir=self.args.results_path, ignore_col_list=ignore_col_list,
                #                       y_axis='aic',
                #                       rotation=0,
                #                       )

        else:
            if len(sig10_AE_cols) > 0:
                clinic_cols = [col for col in clinic_cols if "date" not in col]
                pearson_corr(df[sig10_AE_cols + clinic_cols], xrotation=60,
                             filepath=os.path.join(self.args.results_path, 'pearsonCorr_AE_sig10.png'))
            else:
                clinic_cols = [col for col in clinic_cols if "date" not in col]
                pearson_corr(df[sig10_AE_cols + clinic_cols], xrotation=60,
                             filepath=os.path.join(self.args.results_path, 'pearsonCorr_AE_noSig.png'))

            # Manually select AE features.
            if multi:
                # sig_cols = ['ECG_pre_QRSd', 'Echo_pre_ESV']
                feature_list = [
                    sig_clinic_cols,
                    sig_clinic_cols + ['SPECT_pre_PBW'],
                    sig_clinic_cols + ['AE_31'],
                    sig_clinic_cols + ['AE_31', 'SPECT_pre_PBW'],
                ]
                multivariate_analysis(df[AE_cols + clinic_cols + [label_str]], feature_list, feature_name_list,
                                      label_str,
                                      fig_title='AE', save_dir=self.args.results_path, ignore_col_list=ignore_col_list,
                                      y_axis='aic', line_index_list=[[0, 1], [0, 2], [1, 3]])
