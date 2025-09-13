"""Model explainability utilities using SHAP and LIME."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available. Install with: pip install lime")

class ModelExplainer:
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model
            X_train: Training data used to fit the explainer
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers if libraries are available
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
        
        if LIME_AVAILABLE:
            self._init_lime_explainer()
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer."""
        try:
            # Choose appropriate explainer based on model type
            model_name = type(self.model).__name__.lower()
            
            if 'tree' in model_name or 'forest' in model_name or 'gradient' in model_name:
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif 'linear' in model_name or 'logistic' in model_name:
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                # Use KernelExplainer for other models (slower but universal)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(self.X_train, 100)
                )
            
            print("✅ SHAP explainer initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _init_lime_explainer(self):
        """Initialize LIME explainer."""
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['Class_0', 'Class_1', 'Class_2'],  # Adjust based on your classes
                mode='classification'
            )
            print("✅ LIME explainer initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_with_shap(self, X_test=None, sample_size=100, plot_type='summary'):
        """
        Generate SHAP explanations.
        
        Args:
            X_test: Test data to explain
            sample_size: Number of samples to use for explanation
            plot_type: Type of plot ('summary', 'waterfall', 'force', 'dependence')
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            print("❌ SHAP explainer not available")
            return None
        
        if X_test is None:
            X_test = self.X_train[:sample_size]
        
        print(f"🔍 Generating SHAP explanations for {len(X_test)} samples...")
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_test)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_class = shap_values[1]  # Use class 1 for binary/first class for multi-class
            else:
                shap_values_class = shap_values
            
            # Create visualizations
            if plot_type == 'summary':
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_class, X_test, feature_names=self.feature_names, show=False)
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            elif plot_type == 'waterfall' and len(X_test) > 0:
                # Waterfall plot for first instance
                if hasattr(shap, 'waterfall_plot'):
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values_class[0],
                            base_values=self.shap_explainer.expected_value,
                            data=X_test[0],
                            feature_names=self.feature_names
                        )
                    )
                    plt.title('SHAP Waterfall Plot (First Instance)')
                    plt.tight_layout()
                    plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
                    plt.show()
            
            elif plot_type == 'force' and len(X_test) > 0:
                # Force plot for first instance
                shap.force_plot(
                    self.shap_explainer.expected_value,
                    shap_values_class[0],
                    X_test[0],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.title('SHAP Force Plot (First Instance)')
                plt.tight_layout()
                plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            print("📊 SHAP plots saved successfully")
            
            return shap_values
            
        except Exception as e:
            print(f"❌ Error generating SHAP explanations: {e}")
            return None
    
    def explain_with_lime(self, instance, num_features=5):
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            print("❌ LIME explainer not available")
            return None
        
        print(f"🔍 Generating LIME explanation for instance...")
        
        try:
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba, 
                num_features=num_features
            )
            
            # Create visualization
            fig = explanation.as_pyplot_figure()
            fig.suptitle('LIME Feature Importance')
            plt.tight_layout()
            plt.savefig('lime_explanation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print explanation
            print("\nLIME Explanation:")
            for feature, importance in explanation.as_list():
                print(f"  {feature}: {importance:.4f}")
            
            print("📊 LIME explanation saved successfully")
            
            return explanation
            
        except Exception as e:
            print(f"❌ Error generating LIME explanation: {e}")
            return None
    
    def permutation_importance_analysis(self, X_test, y_test, n_repeats=5):
        """
        Calculate permutation importance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_repeats: Number of permutation repeats
        """
        print(f"🔍 Calculating permutation importance...")
        
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42,
                n_jobs=-1
            )
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df['importance_mean'], 
                    xerr=importance_df['importance_std'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Permutation Importance')
            plt.title('Permutation Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Permutation importance analysis completed")
            print("\nTop 5 most important features:")
            for i in range(min(5, len(importance_df))):
                row = importance_df.iloc[i]
                print(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
            
            return importance_df
            
        except Exception as e:
            print(f"❌ Error calculating permutation importance: {e}")
            return None
    
    def feature_interaction_analysis(self, X_test, feature1_idx, feature2_idx, sample_size=100):
        """
        Analyze feature interactions using SHAP.
        
        Args:
            X_test: Test data
            feature1_idx: Index of first feature
            feature2_idx: Index of second feature
            sample_size: Number of samples to use
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            print("❌ SHAP explainer not available for interaction analysis")
            return None
        
        print(f"🔍 Analyzing interaction between {self.feature_names[feature1_idx]} and {self.feature_names[feature2_idx]}")
        
        try:
            # Calculate SHAP values
            X_sample = X_test[:sample_size]
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1
            
            # Create dependence plot
            shap.dependence_plot(
                feature1_idx, 
                shap_values, 
                X_sample,
                feature_names=self.feature_names,
                interaction_index=feature2_idx,
                show=False
            )
            plt.title(f'Feature Interaction: {self.feature_names[feature1_idx]} vs {self.feature_names[feature2_idx]}')
            plt.tight_layout()
            plt.savefig(f'interaction_{feature1_idx}_{feature2_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Feature interaction analysis completed")
            
        except Exception as e:
            print(f"❌ Error analyzing feature interactions: {e}")
    
    def global_feature_importance(self, X_test, method='shap'):
        """
        Calculate global feature importance.
        
        Args:
            X_test: Test data
            method: Method to use ('shap', 'permutation', or 'model_native')
        """
        print(f"🔍 Calculating global feature importance using {method}...")
        
        if method == 'shap' and SHAP_AVAILABLE and self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(X_test)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use class 1
                
                # Calculate mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': mean_shap
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
            except Exception as e:
                print(f"❌ Error with SHAP importance: {e}")
                return None
        
        elif method == 'model_native' and hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            print(f"❌ Method {method} not available or supported")
            return None
    
    def create_explainability_report(self, X_test, y_test, instance_idx=0, save_plots=True):
        """
        Create a comprehensive explainability report.
        
        Args:
            X_test: Test features
            y_test: Test targets
            instance_idx: Index of instance to explain individually
            save_plots: Whether to save plots
        """
        print("📊 Creating comprehensive explainability report...")
        
        report = {
            'global_importance': None,
            'permutation_importance': None,
            'shap_values': None,
            'lime_explanation': None
        }
        
        # Global feature importance
        if hasattr(self.model, 'feature_importances_'):
            report['global_importance'] = self.global_feature_importance(X_test, method='model_native')
            print("✅ Model native importance calculated")
        
        # Permutation importance
        report['permutation_importance'] = self.permutation_importance_analysis(X_test, y_test)
        
        # SHAP analysis
        if SHAP_AVAILABLE and self.shap_explainer:
            report['shap_values'] = self.explain_with_shap(X_test, plot_type='summary')
            print("✅ SHAP analysis completed")
        
        # LIME analysis for single instance
        if LIME_AVAILABLE and self.lime_explainer and len(X_test) > instance_idx:
            report['lime_explanation'] = self.explain_with_lime(X_test[instance_idx])
            print("✅ LIME analysis completed")
        
        print("📋 Explainability report completed!")
        
        return report
