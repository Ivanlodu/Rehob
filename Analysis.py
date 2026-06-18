import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier 
import numpy as np

class Analysis:
    drop_cols = [ # post fight leakage
    'finish', 'finish_details', 'finish_round',
    'finish_round_time', 'total_fight_time_secs',
    # identifiers not useful for prediction
    'R_fighter', 'B_fighter', 'location', 'country']
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
    
    def clean(self):
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.drop(columns=self.drop_cols)
    
        # encode target
        df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0})
        df = pd.get_dummies(df, columns=['R_Stance', 'B_Stance', 'weight_class', 'gender','better_rank'])
        
        # fill nulls
        rank_cols = [col for col in df.columns if '_rank' in col]
        df[rank_cols] = df[rank_cols].fillna(99)
        
        odds_cols = [col for col in df.columns if '_odds' in col or '_ev' in col]
        df[odds_cols] = df[odds_cols].fillna(df[odds_cols].median())
        
        df = df.fillna(0)
        
        self.data = df

    
    def split(self):
        df = self.data
        self.train = df[df['date'] < '2020-01-01']
        self.validate = df[(df['date'] >= '2020-01-01') & (df['date'] < '2022-01-01')]
        self.test = df[df['date'] >= '2022-01-01']

    def train_model(self):
        x_train = self.train.drop(columns=['Winner', 'date'])
        y_train = self.train['Winner']

        x_val = self.validate.drop(columns=['Winner', 'date'])
        y_val = self.validate['Winner']

        self.model = RandomForestClassifier(n_estimators=500, random_state=42,max_depth=10, min_samples_split=5, max_features='sqrt')
        self.model.fit(x_train, y_train)

        score = self.model.score(x_val, y_val)
        print(f'Validation Accuracy: {score:.4f}')

        #importance = pd.Series(self.model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
        #print("Top 20 Feature Importances:")
        #print(importance.head(20))


    def train_model2(self):
        
        x_train = self.train.drop(columns=['Winner', 'date'])
        y_train = self.train['Winner']

        x_val = self.validate.drop(columns=['Winner', 'date'])
        y_val = self.validate['Winner']

        odds_to_drop = ['R_odds', 'B_odds', 'R_ev', 'B_ev', 
                'r_dec_odds', 'b_dec_odds', 'r_ko_odds', 
                'b_ko_odds', 'r_sub_odds', 'b_sub_odds']

        x_train = x_train.drop(columns=odds_to_drop)
        x_val = x_val.drop(columns=odds_to_drop)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        self.lr_model = LogisticRegression(max_iter=1000)
        self.lr_model.fit(x_train_scaled, y_train)

        self.ada_model = AdaBoostClassifier(
            estimator=self.lr_model,  
            n_estimators=100, 
            random_state=42,
            algorithm='SAMME'  
        )

        self.ada_model.fit(x_train_scaled, y_train)

        score = self.ada_model.score(x_val_scaled, y_val)
        print(f'LR + AdaBoost Accuracy: {score:.4f}')

analysis = Analysis('ufc-master.csv')
analysis.clean()
analysis.split()
analysis.train_model()
analysis.train_model2()
