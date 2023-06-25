import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import itertools
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc, ClassifierAdapter
from sklearn.model_selection import KFold

class Experiment:

    def __init__(self, df, protected_attribute, task_types, model_types, random_thresholds, n_train, n_test, random_seed, data_scale, conformal_pred=False):
        self.df = df
        self.protected_attribute = protected_attribute
        self.random_thresholds = random_thresholds
        self.task_types = task_types 
        self.model_types = model_types
        self.n_train = n_train
        self.n_test = n_test
        self.random_seed = random_seed
        self.data_scale = data_scale
        self.conformal_pred = conformal_pred

        self.pretrained_scores = None
        self.pretrained_pvalues = None
        self.pretrained_partitions = None

        random.Random(self.random_seed).shuffle(self.model_types)

    def format_data(self, task):
        df_train, df_test = sklearn.model_selection.train_test_split(self.df[task], test_size=0.2, random_state=self.random_seed)

        X_train, X_test = df_train.drop(columns=[task, self.protected_attribute]).to_numpy(), df_test.drop(columns=[task, self.protected_attribute]).to_numpy()
        y_train, y_test = df_train[task].to_numpy(), df_test[task].to_numpy()
        z_train, z_test = df_train[self.protected_attribute].to_numpy(), df_test[self.protected_attribute].to_numpy()
        
        return {'X_tr' : X_train, 'X_test' : X_test, 'y_tr': y_train, 'y_test' : y_test, 'z_tr': z_train, 'z_test': z_test}

    def get_partition(self, data, rows_index=None, features_seed=None, feature_subset=None):
        rng = random.Random(self.random_seed)
        if not rows_index:
        	rows_index = rng.randint(0, int(self.data_scale) - 1)

        X_train, y_train, z_train = data['X_tr'], data['y_tr'], data['z_tr']
        X_test, y_test, z_test = data['X_test'], data['y_test'], data['z_test']
        y_train = np.expand_dims(y_train, axis = 1)
        z_train = np.expand_dims(z_train, axis = 1)
        data = np.concatenate((X_train, y_train, z_train), axis = 1)
        np.random.RandomState(seed=self.random_seed).shuffle(data)
        X_train, y_train, z_train = data[:, : -2], data[:, -2], data[:, -1]

        N = len(y_train)
        block_length = int(N // self.data_scale)

        start, end = block_length * rows_index, block_length * rows_index + block_length

        if features_seed:
            ncols = X_train.shape[1]
            cols = np.arange(ncols)
            np.random.RandomState(seed=features_seed).shuffle(cols)
            cols = cols[:int(ncols*feature_subset)]
            X_train = X_train[:, cols]
            X_test = X_test[:, cols]

        return {'X_tr' : X_train[start : end], 'X_test' : X_test, 
        		'y_tr': y_train[start : end], 'y_test' : y_test, 
        		'z_tr': z_train[start: end], 'z_test' : z_test}

    def train_model(self, data, method="logistic"):
        if method == 'logistic':
            base_clf = LogisticRegression
            base_clf = make_pipeline(StandardScaler(), base_clf(max_iter=100, random_state=self.random_seed))
        elif method == 'gbm':
            base_clf = ensemble.GradientBoostingClassifier
            if self.n_train > 10000:
                base_clf = make_pipeline(StandardScaler(), base_clf(min_samples_split=100, random_state=self.random_seed))
            else:
                base_clf = make_pipeline(StandardScaler(), base_clf(min_samples_split=2, random_state=self.random_seed))
        elif method == 'svm':
            base_clf = svm.SVC
            base_clf = make_pipeline(StandardScaler(), base_clf(max_iter=2500, random_state=self.random_seed))
        elif method == 'nn':
            base_clf = MLPClassifier
            base_clf = make_pipeline(StandardScaler(), base_clf(solver='lbfgs', max_iter=100, random_state=self.random_seed))
        elif method == "tree":
            base_clf = DecisionTreeClassifier
            if self.n_train > 10000:
                base_clf = make_pipeline(StandardScaler(), base_clf(min_samples_split=100, random_state=self.random_seed))
            else:
                base_clf = make_pipeline(StandardScaler(), base_clf(min_samples_split=2, random_state=self.random_seed))

        model = CalibratedClassifierCV(base_clf)
        model.fit(data['X_tr'], data['y_tr'])

        return model

    def get_risk_scores(self, data, model):
        scores = model.predict_proba(data['X_test'])
        scores = [i[1] for i in scores]
        return scores

    def get_conformal_pvalues(self, data, model):
        icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model), MarginErrFunc()))
        icp.calibrate(data["X_cal"], data["y_cal"])
        return icp.predict(data["X_test"], None)


    def train_conformal_model(self, data, model_type):
        kf = KFold(n_splits=5, shuffle=False)

        risk_scores = []
        conformal_pvalues = []
        for k, (train_idx, cal_idx) in enumerate(kf.split(data["X_tr"])):
            partition = data.copy()
            partition["X_tr"] = data["X_tr"][train_idx]
            partition["X_cal"] = data["X_tr"][cal_idx]
            partition["y_tr"] = data["y_tr"][train_idx]
            partition["y_cal"] = data["y_tr"][cal_idx]
            partition["z_tr"] = data["z_tr"][train_idx]
            partition["z_cal"] = data["z_tr"][cal_idx]

            model = self.train_model(partition, model_type)
            risk_scores.append(self.get_risk_scores(partition, model))
            conformal_pvalues.append(self.get_conformal_pvalues(partition, model))

        risk_scores = np.mean(np.array(risk_scores), axis=0)
        conformal_pvalues = np.mean(np.array(conformal_pvalues), axis=0)

        return risk_scores, conformal_pvalues

    def pretrain_models(self):            
        data_partitions = {}
        risk_scores = {}
        conformal_pvalues = {}

        for task in self.task_types:
            risk_scores[task] = {}
            conformal_pvalues[task] = {}
            
            data = self.format_data(task)
            partition = self.get_partition(data)
            data_partitions[task] = partition

            for model_type in self.model_types:
                print(task, model_type)

                if self.conformal_pred:
                    risk_scores[task][model_type], conformal_pvalues[task][model_type] = self.train_conformal_model(partition, model_type)
                else:
                    model = self.train_model(partition, model_type)
                    risk_scores[task][model_type] = self.get_risk_scores(partition, model)
                    conformal_pvalues[task][model_type] = None

        self.pretrained_scores = risk_scores
        self.pretrained_partitions = data_partitions
        self.pretrained_pvalues = conformal_pvalues

    def pretrain_diff_models(self, features_subset=(2/3)):
        data_partitions = {}
        risk_scores = {}
        conformal_pvalues = {}

        for task in self.task_types:
            risk_scores[task] = {}
            conformal_pvalues[task] = {}

            data = self.format_data(task)
            k = 0
            for model_type in self.model_types:
                print(task, model_type)

                partition = self.get_partition(data, k, k, features_subset)
                data_partitions[task] = partition               # X_values in partition not important for tracking

                if self.conformal_pred:
                    risk_scores[task][model_type], conformal_pvalues[task][model_type] = self.train_conformal_model(partition, model_type)
                else:
                    model = self.train_model(partition, model_type)
                    risk_scores[task][model_type] = self.get_risk_scores(partition, model)
                    conformal_pvalues[task][model_type] = None
                k += 1

        self.pretrained_scores = risk_scores
        self.pretrained_partitions = data_partitions
        self.pretrained_pvalues = conformal_pvalues


    def experiment_risk_scores(self):
        results = []
        for task in self.task_types:
            for model_type in self.model_types:
                for t in self.random_thresholds:
                    for i in range(self.n_test):
                        r = self.pretrained_scores[task][model_type][i]
                        p = self.pretrained_pvalues[task][model_type][i]
                        result = {"risk_score": r, "threshold": t, "random": 0}

                        if t>0 and p[0]>t and p[1]>t:
                            result["random"]=1
                        elif t>0 and p[0]<=t and p[1]<=t:
                            result["random"]=1
                        results.append(result)
        return pd.DataFrame(results)

    def experiment_baseline(self, num_models=10, iterative=True):
        print("Running Baseline Experiment")
        results = []
        for task in self.task_types:
            for model_type in self.model_types:
                models = ModelGroup("baseline", self.random_thresholds, 0, self.data_scale, self.random_seed, model_type, task, self.n_test)              
                for k in range(num_models):
                    models.update_metrics(self.pretrained_partitions[task], 
                            self.pretrained_scores[task][model_type], self.pretrained_pvalues[task][model_type])
                    models.update_num_models(k+1)
                    if iterative:
                        results += models.final_metrics()
                        
                if not iterative:
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_tasks(self):
        print("Running Tasks Experiment")
        results = []
        for model_type in self.model_types:
            for num_models in range(1, len(self.task_types)+1):
                task_groups = list(itertools.combinations(self.task_types, num_models))
                for task_group in task_groups:
                    models = ModelGroup("tasks", self.random_thresholds, num_models, self.data_scale, self.random_seed, model_type, task_group, self.n_test)  
                    for task in task_group:
                        models.update_metrics(self.pretrained_partitions[task], 
                            self.pretrained_scores[task][model_type], self.pretrained_pvalues[task][model_type])
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_models(self, exp_type="models"):
        print("Running Models Experiment")
        results = []
        for task in self.task_types:
            for num_models in range(1, len(self.model_types)+1):
                model_groups = list(itertools.combinations(self.model_types, num_models))
                for model_group in model_groups:
                    models = ModelGroup(exp_type, self.random_thresholds, num_models, self.data_scale, self.random_seed, model_group, task, self.n_test)  
                    for model_type in model_group:
                        models.update_metrics(self.pretrained_partitions[task], 
                            self.pretrained_scores[task][model_type], self.pretrained_pvalues[task][model_type])
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_partitions(self, num_models=5, iterative=True):
        print("Running Data Partitions Experiment")
        if (self.data_scale < num_models):
            print("Data scale too small for unique partitions")
            return

        results = []
        for task in self.task_types:
            data = self.format_data(task)
            for model_type in self.model_types:
                print(task, model_type)

                models = ModelGroup("data_partitions", self.random_thresholds, 0, self.data_scale, self.random_seed, model_type, task, self.n_test)            
                for k in range(num_models):
                    partition = self.get_partition(data, k)
                    if self.conformal_pred:
                        risk_scores, conformal_pvalues = self.train_conformal_model(partition, model_type)
                    else:
                        model = self.train_model(partition, model_type)
                        risk_scores = self.get_risk_scores(partition, model)
                        conformal_pvalues = None

                    models.update_metrics(partition, risk_scores, conformal_pvalues)
                    models.update_num_models(k+1)
                    if iterative:
                        results += models.final_metrics()
                        
                if not iterative:
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_features(self, features_subset=(2/3), num_models=5, iterative=True):
        print("Running Features Experiment")
        results = []
        for task in self.task_types:
            data = self.format_data(task)
            for model_type in self.model_types:
                print(task, model_type)

                models = ModelGroup("features", self.random_thresholds, 0, self.data_scale, self.random_seed, model_type, task, self.n_test)
                for k in range(num_models):
                    partition = self.get_partition(data, None, k, features_subset)
                    if self.conformal_pred:
                        risk_scores, conformal_pvalues = self.train_conformal_model(partition, model_type)
                    else:
                        model = self.train_model(partition, model_type)
                        risk_scores = self.get_risk_scores(partition, model)
                        conformal_pvalues = None

                    models.update_metrics(partition, risk_scores, conformal_pvalues)
                    models.update_num_models(k+1)
                    if iterative:
                        results += models.final_metrics()
                        
                if not iterative:
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_all(self, features_subset=(2/3)):
        print("Running All Variations Experiment")
        if (self.data_scale < len(self.model_types)):
            print("Data scale too small for unique partitions")
            return
        self.pretrain_diff_models(features_subset)
        return self.experiment_models("all")


class Homogenization:
    def __init__(self, exp_type, random_distance, num_models, data_scale, random_seed, model_type, task_type, size):
        self.exp_type = exp_type
        self.random_distance = random_distance
        self.num_models = num_models
        self.data_scale = data_scale
        self.random_seed = random_seed
        self.model_type = model_type
        self.task_type = task_type
        
        self.accuracy = []
        self.acceptance = []
        
        self.systemic_success_lockout = (np.ones(size)==1)
        self.systemic_failure_lockout = (np.ones(size)==1)
        self.failure_rate_lockout = 1

        self.systemic_success_inaccurate = (np.ones(size)==1)
        self.systemic_failure_inaccurate = (np.ones(size)==1)
        self.failure_rate_inaccurate = 1

        self.pred_pos_0 = []
        self.pred_pos_1 = []

        self.true_pos_0 = []
        self.true_pos_1 = []

        self.random_0 = []
        self.random_1 = []

        self.conf_coverage_0 = []
        self.conf_coverage_1 = []

        self.conf_set_size_0 = []
        self.conf_set_size_1 = []

        self.size = size

    def get_predictions(self, risk_scores, conformal_pvalues=None):
        pred = []
        for i in range(self.size):
            r = risk_scores[i]

            if conformal_pvalues is not None:
                p = conformal_pvalues[i]
                if self.random_distance>0 and p[0]>self.random_distance and p[1]>self.random_distance:
                    pred.append(np.random.binomial(1, r))
                elif self.random_distance>0 and p[0]<=self.random_distance and p[1]<=self.random_distance:
                    pred.append(np.random.binomial(1, r))
                elif (r >= 0.5):
                    pred.append(1)
                else:
                    pred.append(0)
            else:
                if (r > 0.5-self.random_distance) and (r < 0.5+self.random_distance):
                    pred.append(np.random.binomial(1, r))
                elif (r >= 0.5):
                    pred.append(1)
                else:
                    pred.append(0)
        return np.array(pred)

    def update_fairness_metrics(self, partition, pred, risk_scores, conformal_pvalues=None):
        df = pd.DataFrame(partition["z_test"])
        df["y_true"] = partition["y_test"]
        df["y_pred"] = pred
        df["risk"] = risk_scores

        if conformal_pvalues is not None:
            df["p0"] = [p[0] for p in conformal_pvalues]
            df["p1"] = [p[1] for p in conformal_pvalues]

            df["set_size"] = (df["p0"]>self.random_distance).astype(int) + (df["p1"]>self.random_distance).astype(int)
            df["coverage"] = ((df["y_true"]==1)*(df["p1"]>self.random_distance)).astype(int) + ((df["y_true"]==0)*(df["p0"]>self.random_distance)).astype(int)

        
        a = df[df[0]==0]
        b = df[df[0]==1]

        self.pred_pos_0.append(a["y_pred"].sum()/len(a))
        self.pred_pos_1.append(b["y_pred"].sum()/len(b))

        if conformal_pvalues is not None:
            self.conf_coverage_0.append(a["coverage"].sum()/len(a))
            self.conf_coverage_1.append(b["coverage"].sum()/len(b))

            self.conf_set_size_0.append(a["set_size"].sum()/len(a))
            self.conf_set_size_1.append(b["set_size"].sum()/len(b))

            self.random_0.append(((a["set_size"]==0).sum() + (a["set_size"]==2).sum())/len(a))
            self.random_1.append(((b["set_size"]==0).sum() + (b["set_size"]==2).sum())/len(b))

        else:
            u = 0.5+self.random_distance
            l = 0.5-self.random_distance
            self.random_0.append(len(a[(a["risk"]>l)&(a["risk"]<u)])/len(a))
            self.random_1.append(len(b[(b["risk"]>l)&(b["risk"]<u)])/len(b))

        df = df[df["y_true"]==1]

        a = df[df[0]==0]
        b = df[df[0]==1]

        self.true_pos_0.append(a["y_pred"].sum()/len(a))
        self.true_pos_1.append(b["y_pred"].sum()/len(b))

    def update_metrics(self, partition, scores, conformal_pvalues=None):
        pred = self.get_predictions(scores, conformal_pvalues)
        self.update_fairness_metrics(partition, pred, scores, conformal_pvalues)
        
        self.accuracy.append(np.sum(pred==partition["y_test"])/len(pred))
        self.acceptance.append(np.sum(pred)/len(pred))
        
        self.failure_rate_lockout *= np.sum(pred==0)/len(pred)
        self.systemic_success_lockout *= (pred==1)
        self.systemic_failure_lockout *= (pred==0)

        self.failure_rate_inaccurate *= np.sum(pred!=partition["y_test"])/len(pred)
        self.systemic_success_inaccurate *= (pred==partition["y_test"])
        self.systemic_failure_inaccurate *= (pred!=partition["y_test"])
    
    def update_num_models(self, num_models):
        self.num_models = num_models
    
    def homogenization_metrics(self, r, method, systemic_success, systemic_failure, failure_rate):
        r["method"] = method
        r["systemic_success"] = np.sum(systemic_success)/len(systemic_success)
        r["systemic_failure"] = np.sum(systemic_failure)/len(systemic_failure)
        r["multiplicity"] = 1-r["systemic_success"]-r["systemic_failure"]
        
        r["failure_rate"] = failure_rate
        r["homogenization_expected_failure"] = r["systemic_failure"]/failure_rate
        r["homogenization_avg_failure"] = r["systemic_failure"]/(1-r["acceptance"])
        
        return r
    
    def final_metrics(self):
        r = {}
        r["exp_type"] = self.exp_type
        r["random_distance"] = self.random_distance
        r["num_models"] = self.num_models
        r["data_scale"] = self.data_scale
        r["random_seed"] = self.random_seed
        r["model_type"] = self.model_type
        r["task_type"] = self.task_type

        r["accuracy"] = np.mean(self.accuracy)
        r["acceptance"] = np.mean(self.acceptance)

        r["pred_pos_0"] = np.mean(self.pred_pos_0)
        r["pred_pos_1"] = np.mean(self.pred_pos_1)
        r["true_pos_0"] = np.mean(self.true_pos_0)
        r["true_pos_1"] = np.mean(self.true_pos_1)
        r["random_0"] = np.mean(self.random_0)
        r["random_1"] = np.mean(self.random_1)

        r["conf_coverage_0"] = np.mean(self.conf_coverage_0)
        r["conf_coverage_1"] = np.mean(self.conf_coverage_1)
        r["conf_set_size_0"] = np.mean(self.conf_set_size_0)
        r["conf_set_size_1"] = np.mean(self.conf_set_size_1)

        r_lockout = self.homogenization_metrics(r.copy(), "lockout",
                        self.systemic_success_lockout, self.systemic_failure_lockout, self.failure_rate_lockout)

        r_inaccurate = self.homogenization_metrics(r.copy(), "inaccurate",
                        self.systemic_success_inaccurate, self.systemic_failure_inaccurate, self.failure_rate_inaccurate)

        return [r_inaccurate, r_lockout] 

class ModelGroup: 
    def __init__(self, exp_type, thresholds, num_models, data_scale, random_seed, model_type, task_type, size):
        self.models = []
        for t in thresholds:
            model = Homogenization(exp_type, t, num_models, data_scale, random_seed, model_type, task_type, size)
            self.models.append(model)
        
    def update_metrics(self, partition, risk_scores, conformal_pvalues=None):
        for m in self.models:
            m.update_metrics(partition, risk_scores, conformal_pvalues)
    
    def update_num_models(self, num_models):
        for m in self.models:
            m.update_num_models(num_models)
    
    def final_metrics(self):
        results = []
        for m in self.models:
            results.append(m.final_metrics())
        return results
