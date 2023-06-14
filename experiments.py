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


class Experiment:

    def __init__(self, df, features, protected_attribute, task_types, model_types, random_thresholds, n_train, n_test, random_seed, data_scale):
        self.df = df
        self.features = features
        self.protected_attribute = protected_attribute
        self.random_thresholds = random_thresholds
        self.task_types = task_types 
        self.model_types = model_types
        self.n_train = n_train
        self.n_test = n_test
        self.random_seed = random_seed
        self.data_scale = data_scale

        self.pretrained_scores = None
        self.pretrained_partitions = None

    def format_data(self, task):
        df_train, df_test = sklearn.model_selection.train_test_split(self.df, test_size=0.2, random_state=self.random_seed)

        X_train, X_test = df_train[self.features].to_numpy(), df_test[self.features].to_numpy()
        y_train, y_test = df_train[task].to_numpy(), df_test[task].to_numpy()
        z_train, z_test = df_train[self.protected_attribute].to_numpy(), df_test[self.protected_attribute].to_numpy()
        
        return {'X_tr' : X_train, 'X_test' : X_test, 'y_tr': y_train, 'y_test' : y_test, 'z_tr': z_train, 'z_test': z_test}

    def get_partition(self, data, index=None, cal_split=0):
        rng = random.Random(self.random_seed)
        if not index:
        	index = rng.randint(0, int(self.data_scale) - 1)

        X_train, y_train, z_train = data['X_tr'], data['y_tr'], data['z_tr']
        X_test, y_test, z_test = data['X_test'], data['y_test'], data['z_test']
        y_train = np.expand_dims(y_train, axis = 1)
        z_train = np.expand_dims(z_train, axis = 1)
        data = np.concatenate((X_train, y_train, z_train), axis = 1)
        np.random.RandomState(seed=self.random_seed).shuffle(data)
        X_train, y_train, z_train = data[:, : -2], data[:, -2], data[:, -1]

        N = len(y_train)
        block_length = int(N // self.data_scale)

        if cal_split>0:
            train_length = int(block_length * (1-cal_split))
            train_start, train_end = index * block_length, index * block_length + train_length
            cal_start, cal_end = index * block_length + train_length, index * block_length + block_length

            return {'X_tr' : X_train[train_start : train_end], 'X_cal': X_train[cal_start : cal_end], 'X_test' : X_test, 
    	            'y_tr': y_train[train_start : train_end], 'y_cal': y_train[cal_start : cal_end], 'y_test' : y_test, 
    	            'z_tr': z_train[train_start : train_end], 'z_cal': z_train[cal_start : cal_end], 'z_test' : z_test}
        else:
            start, end = block_length * index, block_length * index + block_length

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

    def pretrain_models(self):            
        data_partitions = {}
        risk_scores = {}
        for task in self.task_types:
            risk_scores[task] = {}
            
            data = self.format_data(task)
            partition = self.get_partition(data)
            data_partitions[task] = partition

            for model_type in self.model_types:
                print(task, model_type)
                model = self.train_model(partition, model_type)
                scores = self.get_risk_scores(partition, model)
                risk_scores[task][model_type] = scores

        self.pretrained_scores = risk_scores
        self.pretrained_partitions = data_partitions

    def experiment_baseline(self, num_models=10, iterative=True):
        print("Running Baseline Experiment")
        results = []
        for task in self.task_types:
            for model_type in self.model_types:
                models = ModelGroup("baseline", self.random_thresholds, 0, self.data_scale, self.random_seed, model_type, task, self.n_test)              
                for k in range(num_models):
                    models.update_metrics(self.pretrained_partitions[task], self.pretrained_scores[task][model_type])
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
                        models.update_metrics(self.pretrained_partitions[task], self.pretrained_scores[task][model_type])
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_models(self):
        print("Running Models Experiment")
        results = []
        for task in self.task_types:
            for num_models in range(1, len(self.model_types)+1):
                model_groups = list(itertools.combinations(self.model_types, num_models))
                for model_group in model_groups:
                    models = ModelGroup("models", self.random_thresholds, num_models, self.data_scale, self.random_seed, model_group, task, self.n_test)  
                    for model_type in model_group:
                        models.update_metrics(self.pretrained_partitions[task], self.pretrained_scores[task][model_type])
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

    def experiment_partitions(self, num_models=5, iterative=True):
        print("Running Data Partitions Experiment")
        results = []
        for task in self.task_types:
            for model_type in self.model_types:
                print(task, model_type)
                data = self.format_data(task)

                models = ModelGroup("data_partitions", self.random_thresholds, 0, self.data_scale, self.random_seed, model_type, task, self.n_test)            
                for k in range(num_models):
                    partition = self.get_partition(data, k)
                    model = self.train_model(partition, model_type)
                    risk_scores = self.get_risk_scores(partition, model)

                    models.update_metrics(partition, risk_scores)
                    models.update_num_models(k+1)
                    if iterative:
                        results += models.final_metrics()
                        
                if not iterative:
                    results += models.final_metrics()
        return pd.DataFrame([i for res in results for i in res])

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
        
        self.fairness_spd = []
        self.fairness_eop = []

    def get_predictions(self, risk_scores):
        pred = []
        for r in risk_scores:
            if (r > 0.5-self.random_distance) and (r < 0.5+self.random_distance):
                pred.append(np.random.binomial(1, r))
            elif (r >= 0.5):
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)

    def get_fairness_metrics(self, partition, pred):
        df = pd.DataFrame(partition["z_test"])
        df["y_true"] = partition["y_test"]
        df["y_pred"] = pred
        
        m = df[df[0]==1]
        f = df[df[0]==0]
        
        spd = (m["y_pred"].sum()/len(m)) - (f["y_pred"].sum()/len(f))
        
        df = df[df["y_true"]==1]
        m = df[df[0]==1]
        f = df[df[0]==0]

        eod = (m["y_pred"].sum()/len(m)) - (f["y_pred"].sum()/len(f))
    
        return spd, eod
    
    def update_metrics(self, partition, scores):
        pred = self.get_predictions(scores)
        spd, eop = self.get_fairness_metrics(partition, pred)
        
        self.accuracy.append(np.sum(pred==partition["y_test"])/len(pred))
        self.acceptance.append(np.sum(pred)/len(pred))
        
        self.fairness_spd.append(spd)
        self.fairness_eop.append(eop)
        
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
        r["fairness_spd"] = np.mean(self.fairness_spd)
        r["fairness_eop"] = np.mean(self.fairness_eop)
        
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
        
    def update_metrics(self, partition, risk_scores):
        for m in self.models:
            m.update_metrics(partition, risk_scores)
    
    def update_num_models(self, num_models):
        for m in self.models:
            m.update_num_models(num_models)
    
    def final_metrics(self):
        results = []
        for m in self.models:
            results.append(m.final_metrics())
        return results
