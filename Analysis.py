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
        #print(f'Validation Accuracy: {score:.4f}')


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
        self.scaler = scaler

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
        #print(f'LR + AdaBoost Accuracy: {score:.4f}')

    def ensemble(self):
        self.train_model()
        self.train_model2()
        x_val = self.validate.drop(columns=['Winner', 'date'])
        y_val = self.validate['Winner']

        odds_to_drop = ['R_odds', 'B_odds', 'R_ev', 'B_ev', 
                        'r_dec_odds', 'b_dec_odds', 'r_ko_odds', 
                        'b_ko_odds', 'r_sub_odds', 'b_sub_odds']

        # RF uses full data with odds
        x_val_rf = x_val

        # LR/Ada uses scaled data without odds
        x_val_lr = x_val.drop(columns=odds_to_drop)
        x_val_lr_scaled = self.scaler.transform(x_val_lr)

        # get probabilities from each model
        rf_proba  = self.model.predict_proba(x_val_rf)[:, 1]
        ada_proba = self.ada_model.predict_proba(x_val_lr_scaled)[:, 1]

        # combine
        combined = (0.6 * rf_proba) + (0.4 * ada_proba)

        predictions = (combined >= 0.5).astype(int)
        accuracy = (predictions == y_val).mean()
        print(f'Ensemble Accuracy: {accuracy:.4f}')
    
    def testcase(self):
        self.train_model()
        self.train_model2()
        x_test = self.test.drop(columns=['Winner', 'date'])
        y_test = self.test['Winner']

        odds_to_drop = ['R_odds', 'B_odds', 'R_ev', 'B_ev', 
                        'r_dec_odds', 'b_dec_odds', 'r_ko_odds', 
                        'b_ko_odds', 'r_sub_odds', 'b_sub_odds']

        # RF uses full data with odds
        x_test_rf = x_test

        # LR/Ada uses scaled data without odds
        x_test_lr = x_test.drop(columns=odds_to_drop)
        x_test_lr_scaled = self.scaler.transform(x_test_lr)

        # get probabilities from each model
        rf_proba  = self.model.predict_proba(x_test_rf)[:, 1]
        ada_proba = self.ada_model.predict_proba(x_test_lr_scaled)[:, 1]

        # combine
        combined = (0.6 * rf_proba) + (0.4 * ada_proba)

        predictions = (combined >= 0.5).astype(int)
        accuracy = (predictions == y_test).mean()
        print(f'Test Accuracy: {accuracy:.4f}')

    def predict_fight(self, fighter1, fighter2):
        raw = pd.read_csv('ufc-master.csv')
        
        # search both corner combinations
        f1_as_red = raw[raw['R_fighter'] == fighter1].sort_values('date')
        f1_as_blue = raw[raw['B_fighter'] == fighter1].sort_values('date')
        f2_as_red = raw[raw['R_fighter'] == fighter2].sort_values('date')
        f2_as_blue = raw[raw['B_fighter'] == fighter2].sort_values('date')

        if f1_as_red.empty and f1_as_blue.empty:
            print(f'{fighter1} not found in dataset')
            return
        if f2_as_red.empty and f2_as_blue.empty:
            print(f'{fighter2} not found in dataset')
            return

        # get most recent fight regardless of corner
        f1_latest = pd.concat([f1_as_red, f1_as_blue]).sort_values('date').iloc[-1]
        f2_latest = pd.concat([f2_as_red, f2_as_blue]).sort_values('date').iloc[-1]

        # build a synthetic fight row using fighter1 as Red, fighter2 as Blue
        fight_row = {}

        # pull Red stats from fighter1's last fight
        for col in raw.columns:
            if col.startswith('R_') and col not in ['R_fighter']:
                if fighter1 in f1_as_red['R_fighter'].values:
                    fight_row[col] = f1_latest[col]
                elif col.replace('R_', 'B_') in raw.columns:
                    fight_row[col] = f1_latest[col.replace('R_', 'B_')]

        # pull Blue stats from fighter2's last fight
        for col in raw.columns:
            if col.startswith('B_') and col not in ['B_fighter']:
                if fighter2 in f2_as_blue['B_fighter'].values:
                    fight_row[col] = f2_latest[col]
                elif col.replace('B_', 'R_') in raw.columns:
                    fight_row[col] = f2_latest[col.replace('B_', 'R_')]

        fight_row['date'] = pd.Timestamp.now()
        fight_row['Winner'] = 0
        fight_row['title_bout'] = 1
        fight_row['no_of_rounds'] = 5
        fight_row['empty_arena'] = 0
        r_rank = f1_latest.get('R_match_weightclass_rank', 99)
        b_rank = f2_latest.get('B_match_weightclass_rank', 99)
        # fight_row['R_odds'] = -450 
        # fight_row['R_ev'] = -450
        # fight_row['B_ev'] = +340
        # fight_row['r_dec_odds'] = 0
        # fight_row['b_dec_odds'] = 0
        #fight_row['r_ko_odds'] = 0
        #fight_row['b_ko_odds'] = 0
        #fight_row['r_sub_odds'] = 0
        #fight_row['b_sub_odds'] = 0

        if r_rank == b_rank:
            fight_row['better_rank'] = 'neither'
        else:
            fight_row['better_rank'] = 'Red' if r_rank < b_rank else 'Blue'

        
        fight_row['weight_class'] = f1_latest['weight_class']
        fight_row['gender'] = f1_latest['gender']

        # fill any dif columns
        for col in raw.columns:
            if col.endswith('_dif') and col not in fight_row:
                fight_row[col] = 0

        df = pd.DataFrame([fight_row])
        df['date'] = pd.to_datetime(df['date'])

        # apply same cleaning as training data
        df = pd.get_dummies(df, columns=['R_Stance', 'B_Stance', 'weight_class', 'gender', 'better_rank'])

        # align columns to match training data
        model_cols = self.model.feature_names_in_
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0  # add missing columns as 0
        df = df[model_cols]  # reorder to match exactly

        # fill nulls
        df = df.fillna(0)

        # get RF probability
        rf_proba = self.model.predict_proba(df)[:, 1][0]

        print(f'\nPredicting: {fighter1} (Red) vs {fighter2} (Blue)')
        print(f'{fighter1} win probability: {rf_proba:.2%}')
        print(f'{fighter2} win probability: {1 - rf_proba:.2%}')
        print(f'Predicted winner: {fighter1 if rf_proba >= 0.5 else fighter2}')


analysis = Analysis('ufc-master.csv')
analysis.clean()
analysis.split()
analysis.ensemble()
analysis.testcase()
#analysis.predict_fight('Mauricio Ruffy', 'Michael Chandler')