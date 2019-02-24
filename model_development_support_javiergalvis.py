def display_importances(feature_importance_df):
    '''
        
    '''
    cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features, order=cols, color='blue')
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        See full source and example: 
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import itertools
    
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, 
                 format(round(cm[i, j],2) if normalize else cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def lgbModel_Regression(df, id_col, target_col, num_folds = 0, file_name='subm_gbm.csv', debug= False):
    '''
        
    '''
    import lightgbm as lgb        
    train_df = df[df[target_col].notnull()]
    test_df = df[df[target_col].isnull()].copy()
    model = lgb.LGBMRegressor(objective='regression',
                            nthread= -1, # how_many_cores()
                            num_leaves=31,
                            learning_rate=0.01,
                            n_estimators=45000,
                            colsample_bytree= 0.9, ## 0.8
                            subsample=0.8715623,
                            max_depth=7, ## 8
                            gamma=1,  ##
                            reg_alpha=0.041545473,
                            reg_lambda=0.0735294,
                            min_split_gain=0.0222415,
                            min_child_weight=39.3259775,
                            silent=-1,
                            verbose=-1, )
    
    # Create arrays and dataframes to store results
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [target_col, id_col]]
    
    if num_folds > 1:
        from sklearn.model_selection import KFold
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target_col])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df[target_col].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target_col].iloc[valid_idx]

            model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 150)

            oof_preds[valid_idx] = model.predict(valid_x, num_iteration=model.best_iteration_)
            sub_preds += model.predict(test_df[feats], num_iteration=model.best_iteration_) / folds.n_splits

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = model.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d rmse : %.6f' % (n_fold + 1, mean_squared_error(valid_y, oof_preds[valid_idx])**0.5))
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        print('Full rmse score %.6f' % mean_squared_error(train_df[target_col], oof_preds)**0.5)
        
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df[target_col], random_state=1, test_size=0.2)
        oof_preds = np.zeros(valid_x.shape[0])    
        
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 150)

        oof_preds = model.predict(valid_x, num_iteration=model.best_iteration_)
        sub_preds += model.predict(test_df[feats], num_iteration=model.best_iteration_) 
    
        feature_importance_df = pd.DataFrame()
        feature_importance_df["feature"] = feats
        feature_importance_df["importance"] = model.feature_importances_
        print(valid_y.shape)
        print(oof_preds.shape)
        print('rmse: %.6f' % (mean_squared_error(valid_y, oof_preds)**0.5))
        del train_x, train_y, valid_x, valid_y
        gc.collect()
        
    # Write submission file and plot feature importance
    if not debug:
        test_df[target_col] = sub_preds
        subm = test_df[[id_col, target_col]]
        subm.to_csv(file_name, index= False)
    display_importances(feature_importance_df)
    return model, subm, feature_importance_df

def XGBmodel_Regression(df, id_col, target_col, num_folds = 0, file_name = 'sub_xgb.csv', debug = False):
    '''
        
    '''
    import xgboost as xgb    
    import operator

    train_df = df[df[target_col].notnull()]
    test_df = df[df[target_col].isnull()].copy()
    
    params = {'objective':'reg:linear','eval_metric':'rmse','colsample_bytree':0.9, 'gamma':1,'max_depth': 7}
    
    # Create arrays and dataframes to store results
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [target_col, id_col]]
    
    matrix_test = xgb.DMatrix(test_df[feats])
    
    if num_folds > 1:
        from sklearn.model_selection import KFold
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target_col])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df[target_col].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target_col].iloc[valid_idx]
            
            matrix_train = xgb.DMatrix(train_x, label=train_y)
            matrix_validation = xgb.DMatrix(valid_x, label=valid_y)
            
            model=xgb.train(params=params,
                            dtrain=matrix_train, num_boost_round=25000, 
                            early_stopping_rounds=100,evals=[(matrix_train,'train'),(matrix_validation,'validation')],
                           )
                        
            oof_preds[valid_idx] = model.predict(matrix_validation, ntree_limit = model.best_ntree_limit)
            sub_preds += model.predict(matrix_test, ntree_limit = model.best_ntree_limit) / folds.n_splits
            
            fold_fimportance = model.get_fscore()
            fold_importance_df = pd.DataFrame(sorted(fold_fimportance.items(), key=operator.itemgetter(1), reverse=True),
                                              columns=['feature','importance'])
            zero_importance = pd.DataFrame({'feature':[x for x in model.feature_names if x not in list(fold_importance_df.feature)],'importance':0})
            fold_importance_df = pd.concat([fold_importance_df,zero_importance], axis=0,ignore_index=True)
            fold_importance_df['fold'] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0, ignore_index=True)

            print('Fold %2d rmse : %.6f' % (n_fold + 1, mean_squared_error(valid_y, oof_preds[valid_idx])**0.5))
        print('Full rmse score %.6f' % mean_squared_error(train_df[target_col], oof_preds)**0.5)
    
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df[target_col],
                                                              random_state=1, test_size=0.2)
        oof_preds = np.zeros(valid_x.shape[0])
        
        matrix_train = xgb.DMatrix(train_x, label=train_y)
        matrix_validation = xgb.DMatrix(valid_x, label=valid_y)
        
        model=xgb.train(params=params,
                        dtrain=matrix_train, num_boost_round=25000, 
                        early_stopping_rounds=100,evals=[(matrix_train,'train'),(matrix_validation,'validation')],
                       )
        
        oof_preds = model.predict(matrix_validation, ntree_limit = model.best_ntree_limit)
        sub_preds = model.predict(matrix_test, ntree_limit = model.best_ntree_limit)
        
        feature_importance = model.get_fscore()
        feature_importance_df = pd.DataFrame(sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True), 
                                             columns=['feature','importance'])
        zero_importance = pd.DataFrame({'feature':[x for x in model.feature_names if x not in list(feature_importance_df.feature)],'importance':0})
        feature_importance_df = pd.concat([feature_importance_df,zero_importance], axis=0,ignore_index=True)

        print('rmse: %.6f' % (mean_squared_error(valid_y, oof_preds)**0.5))
    
    del train_x, train_y, valid_x, valid_y, matrix_train, matrix_validation,
    gc.collect()
    
    # Write submission file and plot feature importance
    test_df[target_col] = sub_preds
    subm = test_df[[id_col, target_col]]
    if not debug:
        subm.to_csv(file_name, index= False)
    display_importances(feature_importance_df)
    
    return model, subm, feature_importance_df


# Permutation Importance
def show_permutation_importance(model, val_X, val_y):
    '''
        Takes the model and dataframes for validation set (X and y)
        then calculates and shows the permutation importance.
        
        For more on permutation importance, check https://www.kaggle.com/dansbecker/permutation-importance
    '''
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
    eli5.show_weights(perm, feature_names = val_X.columns.tolist())
    
def show_partial_dependence(model, val_X, features):
    '''
        Takes the model and dataframe for validation set (X)
        then plots the partial dependence plot
        For more on this, check https://www.kaggle.com/dansbecker/partial-plots?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights 
    '''
    from matplotlib import pyplot as plt
    from pdpbox import pdp, get_dataset, info_plots # Do I need get_dataset and info_plots???
    
    feature_names = [i for i in val_X.columns if val_X[i].dtype in [np.int64]]
    
    if (not type(features) == list):
        # Create the data that we will plot
        pdp_feature = pdp.pdp_isolate(model=model, dataset=val_X, model_features=feature_names, feature=features)

        # plot it
        pdp.pdp_plot(pdp_feature, feature)
        plt.show()
    
    elif (type(features) == list) & (len(features) == 2):
        # Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
        inter1  =  pdp.pdp_interact(model=model, dataset=val_X, model_features=feature_names, features=features)

        pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
        plt.show()
    else:
        print('Error, check input and also think of a better error message....  don\'t be lazy')
        
def show_shap_values(model, input_vector):
    '''
    Takes the model and the record vector (X)
    then calculates and shows the shap value for record.
    
    For more on SHAP values, check https://www.kaggle.com/dansbecker/shap-values?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights
    '''
    
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_vector)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], input_vector)    
    
    
def show_summary_plots(model, val_X):
    '''
        https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights 
    '''
    import shap  # package used to calculate Shap values

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(val_X)

    # Make plot. we call shap_values[1]. For classification problems, there is a separate array of SHAP values for each possible outcome. In this case, we index in to get the SHAP values for the prediction of "True".
    shap.summary_plot(shap_values[1], val_X)

def show_dependence_contribution_plot(model, val_X, feature_of_interest, interaction_index = None):
    '''
        https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights 
    '''
    import shap  # package used to calculate Shap values

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(val_X)

    # make plot.
    shap.dependence_plot(feature_of_interest, shap_values[1], X, interaction_index=interaction_index)

