import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
from steps.train import Trainer

class HyperparameterTuner:
    def __init__(self):
        self.config = self.load_config()
        self.trainer = Trainer()
        
    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
    
    def get_param_grid(self, model_name):
        """Define parameter grids for different models"""
        param_grids = {
            'DecisionTreeClassifier': {
                'model__criterion': ['gini', 'entropy'],
                'model__max_depth': [3, 5, 10, 15, 20, None],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__max_features': ['sqrt', 'log2', None]
            },
            'RandomForestClassifier': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [5, 10, 15, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2']
            },
            'GradientBoostingClassifier': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        }
        return param_grids.get(model_name, {})
    
    def tune_hyperparameters(self, X_train, y_train, search_type='grid', cv=5):
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            cv: Number of cross-validation folds
        """
        model_name = self.config['model']['name']
        param_grid = self.get_param_grid(model_name)
        
        if not param_grid:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Use the existing pipeline from trainer
        pipeline = self.trainer.pipeline
        
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',  # You can change this to other metrics
                n_jobs=-1,  # Use all available cores
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=50,  # Number of parameter settings sampled
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        print(f"Starting {search_type} search for {model_name}...")
        search.fit(X_train, y_train)
        
        return search
    
    def update_config_with_best_params(self, best_params, update_original=True):
        """Update config file with best parameters found"""
        # Extract only model parameters (remove 'model__' prefix)
        model_params = {}
        for key, value in best_params.items():
            if key.startswith('model__'):
                param_name = key.replace('model__', '')
                model_params[param_name] = value
        
        # Update config
        self.config['model']['params'] = model_params
        
        if update_original:
            # Update the original config.yml file
            with open('config.yml', 'w') as config_file:
                yaml.dump(self.config, config_file, default_flow_style=False)
            print("Best parameters updated in original 'config.yml'")
        else:
            # Save to a separate file
            with open('config_best_params.yml', 'w') as config_file:
                yaml.dump(self.config, config_file, default_flow_style=False)
            print("Best parameters saved to 'config_best_params.yml'")
        
        print("Best parameters found:")
        for param, value in model_params.items():
            print(f"  {param}: {value}")
    
    def run_tuning_workflow(self, data=None, data_path=None, search_type='random'):
        """Complete workflow: load data -> tune -> save best params"""
        
        # Handle both dataframe and file path inputs
        if data is not None:
            # Use provided dataframe directly
            train_data = data
        elif data_path is not None:
            # Load data from file path (original functionality)
            train_data = pd.read_csv(data_path)
        else:
            raise ValueError("Either 'data' dataframe or 'data_path' must be provided")
        
        # Rest of the method remains the same
        X, y = self.trainer.feature_target_separator(train_data)
        
        # Split data for tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config['train']['test_size'],
            random_state=self.config['train']['random_state'],
            shuffle=self.config['train']['shuffle']
        )
        
        # Perform hyperparameter tuning
        search = self.tune_hyperparameters(X_train, y_train, search_type=search_type)
        
        if search:
            print(f"\nBest score: {search.best_score_:.4f}")
            print(f"Best parameters: {search.best_params_}")
            
            # Evaluate on validation set
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Update config with best parameters
            self.update_config_with_best_params(search.best_params_, update_original=True)
            
            return search
        
        return None